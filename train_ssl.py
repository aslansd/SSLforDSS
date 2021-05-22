import copy
import logging
import numpy as np
import os
# import pdb

from typing import List

import data, io_dss

from cpc_dss import CPCModel

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable = True)

try:
    from tensorflow.python.framework.ops import disable_eager_execution
    disable_eager_execution()
except Exception as e:
    logging.exception(e)
       
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

def train_ssl(*, data_dir: str, y_suffix: str = '',
              save_dir: str = './', save_prefix: str = None,
              model_name: str = 'tcn', nb_filters: int = 16, kernel_size: int = 16,
              nb_conv: int = 3, use_separable: List[bool] = False, nb_hist: int = 512,
              ignore_boundaries: bool = True, batch_norm: bool = True,
              nb_pre_conv: int = 0,
              pre_kernel_size: int = 3, pre_nb_filters: int = 16, pre_nb_conv: int = 2,
              verbose: int = 2, batch_size: int = 32,
              nb_epoch: int = 10,
              learning_rate: float = None, reduce_lr: bool = False, reduce_lr_patience: int = 5,
              fraction_data: float = 1.0, seed: int = None, batch_level_subsampling: bool = False,
              tensorboard: bool = False, log_messages: bool = False,
              nb_stacks: int = 2, with_y_hist: bool = True, x_suffix: str = '',
              early_stop_epoch: int = 20, gru_units: int = 256, dropout: float = 0.2, neg: int = 10, steps: int = 12):
    
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
                      Defaults to 512.
       ignore_boundaries (bool): Minimize edge effects by discarding predictions at the edges of chunks.
                                 Defaults to True.
       batch_norm (bool): Batch normalize.
                          Defaults to True.
       nb_pre_conv (int): Adds downsampling frontend.
                          TCN: adds a frontend of N conv blocks (conv-relu-batchnorm-maxpool2) to the TCN - useful for reducing the sampling rate for USV.
                          TCN_STFT: stft.
                          Defaults to 0 (no frontend).
       pre_nb_filters (int): [description]. Defaults to 16.
       pre_kernel_size (int): [description]. Defaults to 3.
       pre_nb_conv (int): [description]. Defaults to 3.
       verbose (int): Verbosity of training output (0 - no output(?), 1 - progress bar, 2 - one line per epoch).
                      Defaults to 2.
       batch_size (int): Batch size.
                         Defaults to 32.
       nb_epoch (int): Maximal number of training epochs.
                       Training will stop early if validation loss did not decrease in the last 20 epochs.
                       Defaults to 10.
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
       nb_stacks (int): Unused if model name is "tcn" or "tcn_stft". Defaults to 2.
       with_y_hist (bool): Unused if model name is "tcn" or "tcn_stft". Defaults to True.
       x_suffix (str): Select specific training data based on suffix (e.g. x_suffix).
                       Defaults to '' (will use the standard data 'x').
       early_stop_epoch (int): An epoch number for early stopping of training.
                               Defaults to 20.
       gru_units (int): Number of units of the context representation (GRU) for Contrastive Predictive Coding (CPC) model.
                        Defaults to 256.
       dropout (float): A dropout rate for Contrastive Predictive Coding (CPC) model.
                        Defaults to 0.2.
       neg (int): Number of negative samples for Contrastive Predictive Coding (CPC) model.
                  Defaults to 10.
       steps (int): Number of steps to predict for Contrastive Predictive Coding (CPC) model.
                    Defaults to 12.
       """
    
    # Configuration
    params = locals()
    
    params['FeatureEncoder_model'] = model_name
    params['nb_hist'] = nb_hist
    params['nb_filters'] = nb_filters
    params['kernel_size'] = kernel_size
    params['nb_conv'] = nb_conv 
    params['learning_rate'] = learning_rate
    params['use_separable'] = use_separable
    params['nb_pre_conv'] = nb_pre_conv
    params['pre_kernel_size'] = pre_kernel_size
    params['pre_nb_filters'] = pre_nb_filters
    params['pre_nb_conv'] = pre_nb_conv
        
    params['data_dir'] = data_dir
    params['y_suffix'] = y_suffix
    params['save_dir'] = save_dir
    params['save_prefix'] = save_prefix
    params['ignore_boundaries'] = ignore_boundaries
    params['batch_norm'] = batch_norm
    params['verbose'] = verbose 
    params['batch_size'] = batch_size
    params['nb_epoch'] = nb_epoch
    params['reduce_lr'] = reduce_lr
    params['reduce_lr_patience'] = reduce_lr_patience
    params['fraction_data'] = fraction_data
    params['seed'] = seed
    params['batch_level_subsampling'] = batch_level_subsampling
    params['tensorboard'] = tensorboard 
    params['log_messages'] = log_messages
    params['nb_stacks'] = nb_stacks 
    params['with_y_hist'] = with_y_hist
    params['x_suffix'] = x_suffix
    
    params['early_stop_epoch'] = early_stop_epoch
    params['gru_units'] = gru_units
    params['dropout'] = dropout
    params['neg'] = neg
    params['steps'] = steps
    
    if log_messages:
        logging.basicConfig(level = logging.INFO)

    params['sample_weight_mode'] = None
    data_padding = 0
    
    if with_y_hist:  # Regression
        params['return_sequences'] = True
        stride = nb_hist
        y_offset = 0
        params['sample_weight_mode'] = 'temporal'
        
        if ignore_boundaries:
            data_padding = int(np.ceil(kernel_size * nb_conv))  # This does not completely avoid boundary effects but should minimize them sufficiently
            stride = stride - 2 * data_padding
    
    else:  # Classification
        params['return_sequences'] = False
        stride = 1  # Should take every sample, since sampling rates of both x and y are now the same
        y_offset = int(round(nb_hist / 2))

    output_stride = 1  # Since we upsample output to original sampling rate. w/o upsampling: `output_stride = int(2 ** nb_pre_conv)` since each pre-conv layer does 2x max pooling

    if save_prefix is None:
        save_prefix = ''

    if len(save_prefix):
        save_prefix = save_prefix + '_'

    if stride <= 0:
        raise ValueError('Stride <= 0 - needs to be > 0. Possible solutions: reduce kernel_size, increase nb_hist parameters, uncheck ignore_boundaries.')

    # Remove learning rate param if not set so the value from the model def is used
    if params['learning_rate'] is None:
        del params['learning_rate']
    
    params['unpack_channels'] = None
    
    if '_multi' in model_name:
        params['unpack_channels'] = True
        
    logging.info('loading data')
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
        
    # TODO clarify nb_channels, nb_freq semantics - always [nb_samples,..., nb_channels] -  nb_freq is ill-defined for 2D data
    params.update({'nb_freq': d['train']['x'].shape[1], 'nb_channels': d['train']['x'].shape[-1], 'nb_classes': len(params['class_names']),
                   'first_sample_train': first_sample_train, 'last_sample_train': last_sample_train,
                   'first_sample_val': first_sample_val, 'last_sample_val': last_sample_val})
           
    logging.info('Parameters:')
    logging.info(params)

    logging.info('preparing data')
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

    logging.info('Training data:')
    logging.info(data_gen)
    logging.info('Validation data:')
    logging.info(val_gen)

    logging.info('building network')  
    model = CPCModel()
    
    # pdb.set_trace()
    fraction_ratio = 1
    data_size = data_gen.x.shape[0]
    val_size = val_gen.x.shape[0]    
    
    for i in range(fraction_ratio):
        print('fraction ' + str(i))
        data_gen_fraction = copy.deepcopy(data_gen)
        
        data_start_index = int(data_size // fraction_ratio * i)
        data_end_index = int(data_size // fraction_ratio * (i + 1))
        
        data_gen_fraction.x = data_gen.x[data_start_index:data_end_index]
        data_gen_fraction.y = data_gen.y[data_start_index:data_end_index]
        
        data_gen_fraction.first_sample = data_gen_fraction.first_sample
        data_gen_fraction.last_sample = data_gen_fraction.x.shape[0]
        data_gen_fraction.nb_samples = data_gen_fraction.last_sample - data_gen_fraction.first_sample

        s0 = data_gen_fraction.first_sample / data_gen_fraction.stride
        s1 = (data_gen_fraction.last_sample - data_gen_fraction.x_hist - 1) / data_gen_fraction.stride
        data_gen_fraction.allowed_batches = np.arange(s0, s1, dtype = np.uintp)
        
        val_gen_fraction = copy.deepcopy(val_gen)
        
        val_start_index = int(val_size // fraction_ratio * i)
        val_end_index = int(val_size // fraction_ratio * (i + 1))
        
        val_gen_fraction.x = val_gen.x[val_start_index:val_end_index]
        val_gen_fraction.y = val_gen.y[val_start_index:val_end_index]
        
        val_gen_fraction.first_sample = 0
        val_gen_fraction.last_sample = val_gen_fraction.x.shape[0]
        val_gen_fraction.nb_samples = val_gen_fraction.last_sample - val_gen_fraction.first_sample

        s0 = val_gen_fraction.first_sample / val_gen_fraction.stride
        s1 = (val_gen_fraction.last_sample - val_gen_fraction.x_hist - 1) / val_gen_fraction.stride
        val_gen_fraction.allowed_batches = np.arange(s0, s1, dtype = np.uintp)
        
        # pdb.set_trace()
        model.load_training_configuration(params, data_gen_fraction, val_gen_fraction)
        model.train()
    
if __name__ == '__main__':
    
    train_ssl(data_dir = 'dat/dmel_single_raw.npy', y_suffix = '', # dat/dmel_single_stern_raw_manualp.npy
              save_dir = 'res', save_prefix = '',
              model_name = 'tcn', nb_filters = 16, kernel_size = 16,
              nb_conv = 3, use_separable = False, nb_hist = 512,
              ignore_boundaries = True, batch_norm = True,
              nb_pre_conv = 0,
              pre_kernel_size = 3, pre_nb_filters = 16, pre_nb_conv = 2,
              verbose = 1, batch_size = 32,
              nb_epoch = 20,
              learning_rate = 0.0001, reduce_lr = False, reduce_lr_patience = 5,
              fraction_data = None, seed = None, batch_level_subsampling = False,
              tensorboard = False, log_messages = False,
              nb_stacks = 2, with_y_hist = True, x_suffix = '',
              early_stop_epoch = 20, gru_units = 32, dropout = 0.2, neg = 10, steps = 10)