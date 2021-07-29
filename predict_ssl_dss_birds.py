import h5py
import logging
import numpy as np

import io_dss, utils

from cpc_ssl_dss_birds import CPCModel

import pdb

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable = True)

try:
    from tensorflow.python.framework.ops import disable_eager_execution
    disable_eager_execution()
except Exception as e:
    logging.exception(e)

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
    
    # model_weights_all = []
    
    # for layer in intermediate_encoder.layers:
    #     model_weights = layer.get_weights()
    
    #     for element in model_weights:
    #         model_weights_all.append(element)
        
    # pdb.set_trace()
    logging.info('Re-loading last best model')
    checkpoint_save_name = save_name + '_model.h5'
    f = h5py.File(checkpoint_save_name, 'r')
    layer_pretrained = {dataset: f.get('model_weights/TCN_Intermediate')[dataset] for dataset in f.get('model_weights/TCN_Intermediate')}

    # pdb.set_trace()
    counter = -1
    for layer_1 in intermediate_encoder.layers:
        for layer_2 in layer_pretrained:
            if layer_2 == layer_1.name:
                counter = counter + 1
                
                # pdb.set_trace()
                if layer_2 == 'trainable_stft':
                    intermediate_encoder.get_layer(layer_1.name).set_weights([np.array(f['model_weights/TCN_Intermediate/' + layer_2 + '/imag_kernels:0']), np.array(f['model_weights/TCN_Intermediate/' + layer_2 + '/real_kernels:0'])])
                else:
                    intermediate_encoder.get_layer(layer_1.name).set_weights([np.array(f['model_weights/TCN_Intermediate/' + layer_2 + '/kernel:0']), np.array(f['model_weights/TCN_Intermediate/' + layer_2 + '/bias:0'])])
                    
                # print('********************')               
                # print((intermediate_encoder.get_layer(layer_1.name).get_weights()[0]).all() == (np.array(f['model_weights/TCN_Intermediate/' + layer_2 + '/kernel:0'])).all())
                # print((intermediate_encoder.get_layer(layer_1.name).get_weights()[1]).all() == (np.array(f['model_weights/TCN_Intermediate/' + layer_2 + '/bias:0'])).all())                
                # print((intermediate_encoder.get_layer(layer_1.name).get_weights()[0]).all() == model_weights_all[2 * counter].all())
                # print((intermediate_encoder.get_layer(layer_1.name).get_weights()[1]).all() == model_weights_all[2 * counter +  1].all())               
                # print(np.array(f['model_weights/TCN_Intermediate/' + layer_2 + '/kernel:0']).all() == model_weights_all[2 * counter].all())
                # print(np.array(f['model_weights/TCN_Intermediate/' + layer_2 + '/bias:0']).all() == model_weights_all[2 * counter +  1].all())
                            
    print('**********' + str(counter + 1) + '**********')
                
    # pdb.set_trace()
    # logging.info('Re-loading last best model')
    # checkpoint_save_name = save_name + '_model.h5'
    # intermediate_encoder.load_weights(checkpoint_save_name, by_name = True)
               
    # pdb.set_trace()
    params['batch_size'] = 1
    model_cpc.load_prediction_configuration(model = intermediate_encoder, config = params, test_gen = d['test'])
    
    # pdb.set_trace()
    model_cpc.predict(config = params)

if __name__ == '__main__':
    
    for gru_units in [16, 32, 64, 128]:        
        predict_ssl_dss(save_name = 'parameter_search_birds_steps=100/birds_' + str(gru_units))