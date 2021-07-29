import logging
import numpy as np
import os

import data, io_dss

import pdb

from cpc_ssl_dss_flies import CPCModel
from typing import List

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable = True)

try:
    from tensorflow.python.framework.ops import disable_eager_execution
    disable_eager_execution()
except Exception as e:
    logging.exception(e)
       
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

def train_ssl_dss(*, data_dir: str, y_suffix: str = '',
                  save_dir: str = './', save_prefix: str = None,
                  model_name: str = 'tcn', nb_filters: int = 16, kernel_size: int = 16,
                  nb_conv: int = 3, use_separable: List[bool] = False, nb_hist: int = 1024,
                  ignore_boundaries: bool = True, batch_norm: bool = True,
                  nb_pre_conv: int = 0,
                  pre_kernel_size: int = 3, pre_nb_filters: int = 16, pre_nb_conv: int = 2,
                  verbose: int = 2, batch_size: int = 32,
                  nb_epoch: int = 20,
                  learning_rate: float = None, reduce_lr: bool = False, reduce_lr_patience: int = 5,
                  fraction_data: float = 1.0, seed: int = None, batch_level_subsampling: bool = False,
                  tensorboard: bool = False, log_messages: bool = False,
                  nb_stacks: int = 2, with_y_hist: bool = True, x_suffix: str = '',
                  dilations: List[int] = [1, 2, 4, 8, 16], activation: str = 'norm_relu', use_skip_connections: bool = True, dropout_rate: float = 0.00, padding: str = 'same',
                  early_stop_epoch: int = 20, gru_units: int = 32, dropout: float = 0.2, neg: int = 12, steps: int = 10):
    
    """Train a DeepSS network using Contrastive Predictive Coding (CPC) model.
    
    Args:
       data_dir (str): Path to the directory or file with the dataset for training.
                       Accepts npy-dirs (recommended), h5 files or zarr files.
                       See documentation for how the dataset should be organized.
       y_suffix (str): Select training target by suffix.
                       Song-type specific targets can be created with a training dataset.
                       Defaults to '' (will use the standard target 'y').
       save_dir (str): Directory to save training outputs.
                       The path of output files will constructed from the SAVE_DIR, an optional prefix, and the time stamp of the start of training.
                       Defaults to current directory ('./').
       save_prefix (str): Prepend to timestamp.
                          Name of files created will be SAVE_DIR/SAVE_PREFIX + "_" + TIMESTAMP or SAVE_DIR/ TIMESTAMP if SAVE_PREFIX is empty.
                          Defaults to '' (empty).
       model_name (str): Network architecture to use.
                         Use "tcn" (TCN) or "tcn_stft" (TCN with STFT frontend).
                         See dss.models for a description of all models.
                         Defaults to 'tcn'.
       nb_filters (int): Number of filters per layer.
                         Defaults to 16.
       kernel_size (int): Duration of the filters (=kernels) in samples.
                          Defaults to 16.
       nb_conv (int): Number of TCN blocks in the network.
                      Defaults to 3.
       use_separable (List[bool]): Specify which TCN blocks should use separable convolutions.
                                   Provide as a space-separated sequence of "False" or "True.
                                   For instance: "True False False" will set the first block in a
                                   three-block (as given by nb_conv) network to use separable convolutions.
                                   Defaults to False (no block uses separable convolution).
       nb_hist (int): Number of samples processed at once by the network (a.k.a chunk size).
                      Defaults to 1024.
       ignore_boundaries (bool): Minimize edge effects by discarding predictions at the edges of chunks.
                                 Defaults to True.
       batch_norm (bool): Batch normalize.
                          Defaults to True.
       nb_pre_conv (int): Adds downsampling frontend.
                          TCN: adds a frontend of N conv blocks (conv-relu-batchnorm-maxpool2) to the TCN - useful for reducing the sampling rate for USV.
                          TCN_STFT: stft.
                          Defaults to 0 (no frontend).
       pre_nb_filters (int): [description].
                             Defaults to 16.
       pre_kernel_size (int): [description].
                              Defaults to 3.
       pre_nb_conv (int): [description].
                          Defaults to 3.
       verbose (int): Verbosity of training output (0 - no output(?), 1 - progress bar, 2 - one line per epoch).
                      Defaults to 2.
       batch_size (int): Batch size.
                         Defaults to 32.
       nb_epoch (int): Maximal number of training epochs.
                       Training will stop early if validation loss did not decrease in the last 20 epochs.
                       Defaults to 20.
       learning_rate (float): Learning rate of the model. Defaults should work in most cases.
                              Values typically range between 0.1 and 0.00001.
                              If None, uses per model defaults: "tcn" 0.0001, "tcn_stft" 0.0005).
                              Defaults to None.
       reduce_lr (bool): Reduce learning rate on plateau.
                         Defaults to False.
       reduce_lr_patience (int): Number of epochs w/o a reduction in validation loss after which to trigger a reduction in learning rate.
                                 Defaults to 5.
       fraction_data (float): Fraction of training and validation to use for training.
                              Defaults to 1.0.
       seed (int): Random seed to reproducible select fractions of the data.
                   Defaults to None (no seed).
       batch_level_subsampling (bool): Select fraction of data for training from random subset of shuffled batches.
                                       If False, select a continuous chunk of the recording.
                                       Defaults to False.
       tensorboard (bool): Write tensorboard logs to save_dir.
                           Defaults to False.
       log_messages (bool): Sets logging level to INFO.
                            Defaults to False (will follow existing settings).
       nb_stacks (int): Unused if model name is "tcn" or "tcn_stft". 
                        Defaults to 2.
       with_y_hist (bool): Unused if model name is "tcn" or "tcn_stft".
                           Defaults to True.
       x_suffix (str): Select specific training data based on suffix (e.g. x_suffix).
                       Defaults to '' (will use the standard data 'x').
       dilations (List[int], optional): [description].
                                        Defaults to [1, 2, 4, 8, 16].
       activation (str, optional): [description].
                                   Defaults to 'norm_relu'.
       use_skip_connections (bool, optional): [description]. 
                                              Defaults to True.
       dropout_rate (float, optional): [description]. 
                                       Defaults to 0.00.
       padding (str, optional): [description].
                                Defaults to 'same'.
       early_stop_epoch (int): An epoch number for early stopping of training.
                               Defaults to 20.
       gru_units (int): Number of units of the context representation (GRU) for Contrastive Predictive Coding (CPC) model.
                        Defaults to 256.
       dropout (float): A dropout rate for Contrastive Predictive Coding (CPC) model.
                        Defaults to 0.2.
       neg (int): Number of negative samples for Contrastive Predictive Coding (CPC) model.
                  Defaults to 12.
       steps (int): Number of steps to predict for Contrastive Predictive Coding (CPC) model.
                    Defaults to 10.
       """
    
    if log_messages:
        logging.basicConfig(level = logging.INFO)

    sample_weight_mode = None
    data_padding = 0
    
    if with_y_hist:  # Regression
        return_sequences = True
        stride = nb_hist
        y_offset = 0
        sample_weight_mode = 'temporal'
        
        if ignore_boundaries:
            data_padding = int(np.ceil(kernel_size * nb_conv))  # This does not completely avoid boundary effects but should minimize them sufficiently
            stride = stride - 2 * data_padding
    
    else:  # Classification
        return_sequences = False
        stride = 1  # Should take every sample, since sampling rates of both x and y are now the same
        y_offset = int(round(nb_hist / 2))

    output_stride = 1  # Since we upsample output to original sampling rate. w/o upsampling: `output_stride = int(2 ** nb_pre_conv)` since each pre-conv layer does 2x max pooling

    if save_prefix is None:
        save_prefix = ''

    if len(save_prefix):
        save_prefix = save_prefix + '_'
        
    params = locals()

    if stride <= 0:
        raise ValueError('Stride <= 0 - needs to be > 0. Possible solutions: reduce kernel_size, increase nb_hist parameters, uncheck ignore_boundaries.')

    # Remove learning rate param if not set so the value from the model def is used
    if params['learning_rate'] is None:
        del params['learning_rate']
    
    params['unpack_channels'] = None
    
    if '_multi' in model_name:
        params['unpack_channels'] = True
        
    logging.info('Loading data')
    d = io_dss.load(data_dir, x_suffix = x_suffix, y_suffix = y_suffix)
    params.update(d.attrs)  # Add metadata from data.attrs to params for saving

    if fraction_data is not None:
        if fraction_data > 1.0:  # Seconds
            logging.info(f"{fraction_data} seconds corresponds to {fraction_data / (d['train']['x'].shape[0] / d.attrs['samplerate_x_Hz']):1.4f} of the training data.")
            fraction_data = np.min((fraction_data / (d['train']['x'].shape[0] / d.attrs['samplerate_x_Hz']), 1.0))
        elif fraction_data < 1.0:
            logging.info(f"Using {fraction_data:1.4f} of data for training and validation.")

    if fraction_data is not None and not batch_level_subsampling:  # Train on a subset
        min_nb_samples = nb_hist * (batch_size + 2)  # Ensure the generator contains at least one full batch
        first_sample_train, last_sample_train = data.sub_range(d['train']['x'].shape[0], fraction_data, min_nb_samples, seed = seed)
        first_sample_val, last_sample_val = data.sub_range(d['val']['x'].shape[0], fraction_data, min_nb_samples, seed = seed)
    else:
        first_sample_train, last_sample_train = 0, None
        first_sample_val, last_sample_val = 0, None
        
    # TODO clarify nb_channels and nb_freq semantics - always [nb_samples,..., nb_channels] -  nb_freq is ill-defined for 2D data
    params.update({'nb_freq': d['train']['x'].shape[1], 'nb_channels': d['train']['x'].shape[-1], 'nb_classes': len(params['class_names']),
                   'first_sample_train': first_sample_train, 'last_sample_train': last_sample_train,
                   'first_sample_val': first_sample_val, 'last_sample_val': last_sample_val})
           
    logging.info('Parameters')
    logging.info(params)

    logging.info('Preparing data')
    if fraction_data is not None and batch_level_subsampling:  # Train on a subset
        np.random.seed(seed)
        shuffle_subset = fraction_data
    else:
        shuffle_subset = None

    data_gen = data.AudioSequence(d['train']['x'], d['train']['y'],
                                  shuffle = True, shuffle_subset = shuffle_subset,
                                  first_sample = first_sample_train, last_sample = last_sample_train, nb_repeats = 1,
                                  **params)
    val_gen = data.AudioSequence(d['val']['x'], d['val']['y'],
                                 shuffle = False, shuffle_subset = shuffle_subset,
                                 first_sample = first_sample_val, last_sample = last_sample_val,
                                 **params)

    logging.info('Training data')
    logging.info(data_gen)
    logging.info('Validation data')
    logging.info(val_gen)

    logging.info('Building network')  
    model_cpc = CPCModel()
    
    # pdb.set_trace()       
    model_cpc.load_training_configuration(config = params, data_gen = data_gen, val_gen = val_gen)
    model_cpc.train(config = params)
    
if __name__ == '__main__':
    
    for gru_units in [16, 32, 64, 128, 256]:
    
        train_ssl_dss(data_dir = 'dat/dmel_multi_murthy_raw_manual_sine.npy', y_suffix = '',
                      save_dir = 'parameter_search_flies_multichannels_steps=1', save_prefix = 'flies_' + str(gru_units),
                      model_name = 'tcn', nb_filters = 32, kernel_size = 32,
                      nb_conv = 3, use_separable = False, nb_hist = 2048,
                      ignore_boundaries = False, batch_norm = True,
                      nb_pre_conv = 0,
                      pre_kernel_size = 3, pre_nb_filters = 16, pre_nb_conv = 2,
                      verbose = 1, batch_size = 32,
                      nb_epoch = 100,
                      learning_rate = 0.0001, reduce_lr = False, reduce_lr_patience = 5,
                      fraction_data = None, seed = 1, batch_level_subsampling = False,
                      tensorboard = False, log_messages = False,
                      nb_stacks = 2, with_y_hist = True, x_suffix = '',
                      dilations = [1, 2, 4, 8, 16], activation = 'norm_relu', use_skip_connections = True, dropout_rate = 0.00, padding = 'same',
                      early_stop_epoch = 20, gru_units = gru_units, dropout = 0.2, neg = 10, steps = 1)
        
        # train_ssl_dss(data_dir = 'dat/dmel_single_raw.npy', y_suffix = '', 
        #               save_dir = 'parameter_search_flies_singlechannel_steps=1', save_prefix = 'flies_' + str(gru_units),
        #               model_name = 'tcn', nb_filters = 16, kernel_size = 16,
        #               nb_conv = 3, use_separable = False, nb_hist = 2048,
        #               ignore_boundaries = False, batch_norm = True,
        #               nb_pre_conv = 0,
        #               pre_kernel_size = 3, pre_nb_filters = 16, pre_nb_conv = 2,
        #               verbose = 1, batch_size = 32,
        #               nb_epoch = 100,
        #               learning_rate = 0.0001, reduce_lr = False, reduce_lr_patience = 5,
        #               fraction_data = None, seed = 1, batch_level_subsampling = False,
        #               tensorboard = False, log_messages = False,
        #               nb_stacks = 2, with_y_hist = True, x_suffix = '',
        #               dilations = [1, 2, 4, 8, 16], activation = 'norm_relu', use_skip_connections = True, dropout_rate = 0.00, padding = 'same',
        #               early_stop_epoch = 20, gru_units = gru_units, dropout = 0.2, neg = 10, steps = 1)