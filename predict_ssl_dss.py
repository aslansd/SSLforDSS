import logging

import io_dss, utils

from cpc_ssl_dss import CPCModel

# import pdb

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable = True)

try:
    from tensorflow.python.framework.ops import disable_eager_execution
    disable_eager_execution()
except Exception as e:
    logging.exception(e)
    
import copy
import flammkuchen as fl
import gc
import numpy as np
import pandas as pd
import sklearn

import evaluate, predict

from tensorflow.keras.optimizers import Adam

def predict_ssl_dss(save_name: str = None):
    
    """Predict a test set using a pre-trained DeepSS network by self-supervised Contrastive Predictive Coding (CPC) model.
        
    Args:
        save_name (str): Path to the directory or file with the parameters for training or prediction.
                         Accepts npy-dirs (recommended), h5 files or zarr files.
                         See documentation for how the dataset should be organized.
    """

    logging.basicConfig(level = logging.INFO)
        
    logging.info('Loading parameters')
    params = utils.load_params(save_name)
    params['save_name'] = save_name
    
    logging.info('Loading data')
    d = io_dss.load(params['data_dir'], x_suffix = params['x_suffix'], y_suffix = params['y_suffix'])

    logging.info('Building network')  
    model_cpc = CPCModel()
    
    # pdb.set_trace()       
    model_training, feature_encoder, intermediate_encoder = model_cpc.load_training_configuration(config = params, data_gen = d['train'], val_gen = d['val'])   
           
    # pdb.set_trace()
    logging.info('Re-loading last best model')
    checkpoint_save_name = save_name + '_model.h5'
    intermediate_encoder.load_weights(checkpoint_save_name, by_name = True)
    
    # pdb.set_trace()
    model_prediction = model_cpc.load_prediction_configuration(model = intermediate_encoder, config = params, test_gen = d['test'])   
    
    # pdb.set_trace()
    # model_cpc.predict(config = params)
    
    model_prediction.compile(optimizer = Adam(lr = params['learning_rate'], amsgrad = True, clipnorm = 1.0), loss = "categorical_crossentropy", sample_weight_mode = params['sample_weight_mode'])
    
    fraction_rate = 200
    data_size = d['test']['x'].shape[0]
    conf_mat = np.zeros((2, 2))
   
    for i in range(1):
        gc.collect()
        print('Fraction ' + str(i + 1))
        
        data_start_index = int(data_size // fraction_rate * i)
        data_end_index = int(data_size // fraction_rate * (i + 1))
                        
        # pdb.set_trace()    
        logging.info('Predicting')        
        x_test_temp, y_test_temp, y_pred_temp = evaluate.evaluate_probabilities(x = d['test']['x'][data_start_index:data_end_index], y = d['test']['y'][data_start_index:data_end_index], model = model_prediction, params = params)
           
        # pdb.set_trace()
        labels_test_temp = predict.labels_from_probabilities(y_test_temp)
        labels_pred_temp = predict.labels_from_probabilities(y_pred_temp)
        
        # pdb.set_trace()
        logging.info('Evaluating')
        print(labels_test_temp.shape, labels_pred_temp.shape)
        
        conf_mat_temp = sklearn.metrics.confusion_matrix(labels_test_temp, labels_pred_temp)       
        conf_mat = conf_mat + conf_mat_temp
        logging.info(conf_mat_temp)
            
        if i == 0:
            x_test = copy.deepcopy(x_test_temp) 
            y_test = copy.deepcopy(y_test_temp)
            y_pred = copy.deepcopy(y_pred_temp)
            
            labels_test = copy.deepcopy(labels_test_temp)
            labels_pred = copy.deepcopy(labels_pred_temp)
        
        elif i > 0:
            x_test = np.concatenate((x_test, x_test_temp)) 
            y_test = np.concatenate((y_test, y_test_temp))
            y_pred = np.concatenate((y_pred, y_pred_temp))
            
            labels_test = np.concatenate((labels_test, labels_test_temp))
            labels_pred = np.concatenate((labels_pred, labels_pred_temp))
    
    # pdb.set_trace()
    labels = np.arange(len(params['class_names']))
    report = sklearn.metrics.classification_report(labels_test, labels_pred, labels = labels, target_names = params['class_names'], digits = 3)
    
    logging.info(conf_mat)
    logging.info(report)
        
    save_filename = "{0}_results.h5".format(save_name)
    logging.info('saving to ' + save_filename)
    d = {'confusion_matrix': conf_mat,
          'classification_report': report,
          'x_test': x_test,
          'y_test': y_test,
          'y_pred': y_pred,
          'labels_test': labels_test,
          'labels_pred': labels_pred}

    fl.save(save_filename, d)
        
if __name__ == '__main__':
  
    predict_ssl_dss(save_name = 'parameter_search_new/256_5_20210630_134536')  # 32_5_20210630_134103 # 64_5_20210630_134536 # 128_5_20210630_134536 # 256_5_20210630_134536