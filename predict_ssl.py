import flammkuchen as fl
import logging
import numpy as np
# import pdb

import evaluate, io_dss, predict, utils

from cpc_dss import ContrastiveLoss, FeatureEncoder_tcn_seq, FeatureEncoder_tcn, FeatureEncoder_tcn_tcn, FeatureEncoder_tcn_small, FeatureEncoder_tcn_stft, FeatureEncoder_tcn_multi
from tensorflow.keras.layers import Dropout, GRU, Input
from tensorflow.keras.models import Model, Sequential

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable = True)

try:
    from tensorflow.python.framework.ops import disable_eager_execution
    disable_eager_execution()
except Exception as e:
    logging.exception(e)

def predict_ssl(model_name: str = 'FeatureEncoder_tcn', save_name: str = None):
    """Predict a test set using a pre-trained DeepSS network by self-supervised Contrastive Predictive Coding (CPC) model.
        
    Args:
        model_name (str): Network architecture to use.
                          Use "FeatureEncoder_tcn" (TCN) or "FeatureEncoder_tcn_stft" (TCN with STFT frontend).
                          See dss.models for a description of all models.
                          Defaults to 'FeatureEncoder_tcn'.
        save_name (str): Path to the directory or file with the parameters for training or prediction.
                         Accepts npy-dirs (recommended), h5 files or zarr files.
                         See documentation for how the dataset should be organized.
    """

    logging.basicConfig(level = logging.INFO)
        
    logging.info('loading parameters')
    params = utils.load_params(save_name)
    
    logging.info('loading data')
    d = io_dss.load(params['data_dir'], x_suffix = params['x_suffix'], y_suffix = params['y_suffix'])
    
    logging.info('building network')
    
    if model_name == 'FeatureEncoder_tcn_seq' or model_name == 'FeatureEncoder_tcn':
        feature_encoder = eval(model_name)(**params)
                      
    elif model_name == 'FeatureEncoder_tcn_tcn' or model_name == 'FeatureEncoder_tcn_small' or model_name == 'FeatureEncoder_tcn_stft':
        feature_encoder = eval(model_name)(**params)
        
    elif model_name == 'FeatureEncoder_tcn_multi':            
        feature_encoder = eval(model_name)(**params)

    # Input shape
    input_shape = (params['nb_hist'], params['nb_freq'])

    # Input tensor
    input_feats = Input(shape = input_shape, name = 'input_layer')

    # Dropout layer
    dropout_layer = Dropout(params['dropout'], name = 'dropout_block')

    # Feature Encoder
    encoder_features = feature_encoder.model(input_feats)
    encoder_output = dropout_layer(encoder_features)

    # Autoregressive model
    autoregressive_model = GRU(params['gru_units'], return_sequences = True, name = 'autoregressive_layer')
    autoregressive_output = autoregressive_model(encoder_output)
    autoregressive_output = dropout_layer(autoregressive_output)

    # Contrastive loss
    contrastive_loss = ContrastiveLoss(params['gru_units'], params['neg'], params['steps'])
    contrastive_loss_output = contrastive_loss([encoder_features, autoregressive_output])

    # Self-supervised Contrastive Predictive Coding Model
    old_model = Model(input_feats, contrastive_loss_output)    
    print(old_model.summary())
           
    checkpoint_save_name = save_name + '_model.h5'
    
    logging.info('re-loading last best model')
    old_model.load_weights(checkpoint_save_name, by_name = True)
    
    # pdb.set_trace()    
    model = Sequential(old_model.layers[:-3])
    print(model.summary())

    # pdb.set_trace()    
    print(logging.info('predicting'))
    x_test, y_test, y_pred = evaluate.evaluate_probabilities(x = d['test']['x'], y = d['test']['y'], model = model, params = params)

    # pdb.set_trace()
    labels_test = predict.labels_from_probabilities(y_test)
    labels_pred = predict.labels_from_probabilities(y_pred)

    # pdb.set_trace()
    logging.info('evaluating')
    conf_mat, report = evaluate.evaluate_segments(labels_test, labels_pred, params['class_names'])
    
    logging.info(conf_mat)
    logging.info(report)
        
    save_filename = "{0}_results.h5".format(save_name)
    logging.info('saving to ' + save_filename + '.')
    d = {'confusion_matrix': conf_mat,
         'classification_report': report,
         'params': params}

    fl.save(save_filename, d)
    
if __name__ == '__main__':
  
    predict_ssl(model_name = 'FeatureEncoder_tcn', save_name = 'res/20210520_171831')