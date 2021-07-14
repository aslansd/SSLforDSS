import flammkuchen as fl
import gc
import logging
import make_dataset
import npy_dir
import numpy as np
import os
import time
import zarr
    
import data, evaluate, io_dss, predict, utils

# import pdb 

from cpc_ssl_dss import CPCModel
from glob import glob
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
from typing import List

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable = True)

try:
    from tensorflow.python.framework.ops import disable_eager_execution
    disable_eager_execution()
except Exception as e:
    logging.exception(e)

def prepare_train_predict_dss(*, data_dir: str, y_suffix: str = '',
                              save_dir: str = './', save_prefix: str = None, save_name_pretrained: str = None,
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
    
    params = locals()
                           
    def load_data(filename: str):
        return np.load(filename)
    
    def load_annotation(filename: str):
        return np.load(filename)['pulsetimes']
        
    for data_idx in [1, 2, 3, 4, 5]:
        
        ### Prepare Data
        
        data_files = glob(data_dir + '*recording.npy')  # List all data files
        annotation_files = [data_file.replace('_recording.npy', '_annotations.npz') for data_file in data_files]  # Generate the names of associated annotation files
        
        data_files = data_files[:data_idx + 2]
        annotation_files = annotation_files[:data_idx + 2]
        
        [print(f'data "{d}" with annotations in "{a}".') for d, a in zip(data_files, annotation_files)];
        
        np.random.seed(1)  # Seed random number generator for reproducible splits
        test_idx, val_idx, train_idx = np.split(np.random.permutation(len(data_files)), (1, 2))  # This will split the recordings into one for testing, one for validation, and the remaining for training  
        print('indices of test file(s):', test_idx, '\nindices of validation file(s):', val_idx, '\nindices of train files:', train_idx)
        
        samplerate = 10_000  # This is the sample rate of your data and the pulse times
        
        root = make_dataset.init_store(nb_channels = 1,  # Number of channels/microphones in the recording
                                       nb_classes = 2,  # Number of classes to predict - [noise, pulse]
                                       samplerate = samplerate,  # Make sure audio data and the annotations are all on the same sampling rate
                                       class_names = ['noise', 'pulse'],
                                       class_types = ['segment', 'event'],
                                       store_type = zarr.DictStore,  # Use DirectoryStore for big data        
                                       store_name = 'dsec_raw.zarr') # Only used with DirectoryStore - this is the path to the directory created
      
        root['test'].create_dataset('eventtimes', shape = (0, 2))
        root['val'].create_dataset('eventtimes', shape = (0, 2))
        root['train'].create_dataset('eventtimes', shape = (0, 2))
        
        for idx, (data_file, annotation_file) in tqdm(enumerate(zip(data_files, annotation_files)), total = len(data_files)):
            # First, load the data
            x = load_data(data_file)  # x should be [nb_samples, nb_channels]
           
            # Second, load the annotations
            pulse_samples = load_annotation(annotation_file)  # y should be a list of pulsetimes in units of samples
            
            # Convert the list of pulse samples to a one-hot-encoded vector of probabilities [times, events]
            y = make_dataset.events_to_probabilities(pulse_samples.astype(np.int), desired_len = x.shape[0])
            
            # Determine whether file is test/val/train
            if idx in test_idx:
                target = 'test'        
            elif idx in val_idx:
                target = 'val'
            elif idx in train_idx:
                target = 'train'
            else:
                continue
        
            offset_seconds = root[target]['x'].shape[0] / samplerate
            
            # Add the recording (x) and the prediction target (y) to the data set
            root[target]['x'].append(x)
            root[target]['y'].append(y)
                
            # Also, save the pulses times for evaluation purposes
            pulse_times = pulse_samples.astype(np.float) / samplerate
            root[target]['eventtimes'].append(np.stack([np.full_like(pulse_times, np.nan), offset_seconds + pulse_times], axis = 1))
               
            # Keep track of which samples in the data come from which file 
            root.attrs[f'filename_{target}'].append(data_file)
            root.attrs[f'filename_startsample_{target}'].append(root[target]['x'].shape[0])
            root.attrs[f'filename_endsample_{target}'].append(root[target]['x'].shape[0] + x.shape[0])
            
        # Save the zarr store as a hierarchy of npy files
        npy_dir.save('dat/dmel_single_raw_' + str(data_idx) + '.npy', root)
        
        if log_messages and data_idx == 1:
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
            
        logging.info('Loading data')
        data_dir_partial = 'dat/dmel_single_raw_' + str(data_idx) + '.npy'
        d = io_dss.load(data_dir_partial, x_suffix = x_suffix, y_suffix = y_suffix)
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
        model_training, feature_encoder, intermediate_encoder = model_cpc.load_training_configuration(config = params, data_gen = data_gen, val_gen = val_gen)
                                
        fraction_index_test_data = np.random.permutation(d['test']['x'].shape[0])[:int(d['test']['x'].shape[0] // 50)]
        
        for train_flag in [1, 2]:
            
            if train_flag == 2:
                logging.info('Re-loading last best model')
                checkpoint_save_name = save_name_pretrained + '_model.h5'
                intermediate_encoder.load_weights(checkpoint_save_name, by_name = True)
                
            # pdb.set_trace()
            model = model_cpc.load_prediction_configuration(model = intermediate_encoder, config = params, test_gen = d['test'])  
                
            ### Train Data
            gc.collect()
        
            # os.makedirs(os.path.abspath(save_dir), exist_ok = True)
            # save_name = '{0}/{1}{2}{3}'.format(save_dir, save_prefix, time.strftime('%Y%m%d_%H%M%S'), '_' + str(data_idx) + '_' + str(train_flag))
            save_name_new =  save_name_pretrained + '_' + str(data_idx) + '_' + str(train_flag)
            checkpoint_save_name = save_name_new + "_model.h5"  # This will overwrite intermediates from previous epochs
            utils.save_params(params, save_name_new)
            
            model.compile(optimizer = Adam(lr = learning_rate, amsgrad = True, clipnorm = 1.0), loss = "categorical_crossentropy", sample_weight_mode = params['sample_weight_mode'])
        
            callbacks = [ModelCheckpoint(checkpoint_save_name, save_best_only = True, save_weights_only = False, monitor = 'val_loss', verbose = 1), EarlyStopping(monitor = 'val_loss', patience = early_stop_epoch)]
            
            if reduce_lr:
                callbacks.append(ReduceLROnPlateau(patience = reduce_lr_patience, verbose = 1))
        
            if tensorboard:
                callbacks.append(TensorBoard(log_dir = save_dir))
                
            # pdb.set_trace()
            logging.info('Start training')        
            fit_hist = model.fit(data_gen,                                 
                                 steps_per_epoch = len(data_gen) // batch_size,
                                 validation_data = val_gen,
                                 validation_steps = len(val_gen) // batch_size,
                                 epochs = nb_epoch,
                                 verbose = verbose,
                                 callbacks = callbacks)
            
            ### Predict Data           
            gc.collect()
            
            # pdb.set_trace()    
            logging.info('Predicting')
            x_test, y_test, y_pred = evaluate.evaluate_probabilities(x = d['test']['x'][fraction_index_test_data], y = d['test']['y'][fraction_index_test_data], model = model, params = params)
        
            # pdb.set_trace()
            labels_test = predict.labels_from_probabilities(y_test)
            labels_pred = predict.labels_from_probabilities(y_pred)
            
            # pdb.set_trace()
            logging.info('Evaluating')
            conf_mat, report = evaluate.evaluate_segments(labels_test, labels_pred, params['class_names'])
            
            logging.info(conf_mat)
            logging.info(report)
            
            print(conf_mat)
            print(report)
            
            save_filename = "{0}_results.h5".format(save_name_new)
            logging.info('saving to ' + save_filename)
            dd = {'confusion_matrix': conf_mat,
                 'classification_report': report,
                 'x_test': x_test,
                 'y_test': y_test,
                 'y_pred': y_pred,
                 'labels_test': labels_test,
                 'labels_pred': labels_pred}
        
            fl.save(save_filename, dd)
    
if __name__ == '__main__':
    
    prepare_train_predict_dss(data_dir = 'dat.raw/', y_suffix = '',
                              save_dir = 'parameter_search', save_prefix = '', save_name_pretrained = 'parameter_search_new_2/128_5_20210630_134536',  # 32_5_20210630_134103 # 64_5_20210630_134536 # 128_5_20210630_134536 # 256_5_20210630_134536
                              model_name = 'tcn', nb_filters = 16, kernel_size = 16,
                              nb_conv = 3, use_separable = False, nb_hist = 4096,
                              ignore_boundaries = True, batch_norm = True,
                              nb_pre_conv = 0,
                              pre_kernel_size = 3, pre_nb_filters = 16, pre_nb_conv = 2,
                              verbose = 1, batch_size = 32,
                              nb_epoch = 10,
                              learning_rate = 0.0001, reduce_lr = False, reduce_lr_patience = 5,
                              fraction_data = None, seed = None, batch_level_subsampling = False,
                              tensorboard = False, log_messages = False,
                              nb_stacks = 2, with_y_hist = True, x_suffix = '',
                              dilations = [1, 2, 4, 8, 16], activation = 'norm_relu', use_skip_connections = True, dropout_rate = 0.00, padding = 'same',
                              early_stop_epoch = 20, gru_units = 128, dropout = 0.2, neg = 10, steps = 5)