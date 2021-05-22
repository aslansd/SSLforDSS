import flammkuchen as fl
import logging
import os
import tensorflow as tf
import time

from abc import ABC, abstractmethod
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.layers import Conv1D, Conv2DTranspose, Dropout, GRU, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from typing import List

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable = True)

try:
    from tensorflow.python.framework.ops import disable_eager_execution
    disable_eager_execution()
except Exception as e:
    logging.exception(e)
    
import evaluate, predict, utils

from models import tcn_seq, tcn, tcn_tcn, tcn_small, tcn_stft, tcn_multi

class ModelBase(ABC):
    """
    Abstract class for neural networks models using Keras.
    New models should inherit from this class so we can run training 
    and prediction the same way for all the models independently.
    """
    
    @abstractmethod
    def load_training_configuration(self, config, data_gen, val_gen):
        """
        It loads the configuration parameters from the configuration dictionary and the input/output features for training.
        :param config: a dictionary with the configuration for training.
        :param data_gen: a dictionary with the training set features.
        :param val_gen: a dictionary with the validation set features.
        :return: instance will have the configuration parameters.
        """

        # Configuration        
        self.model_name = config['FeatureEncoder_model']
        self.nb_hist = config['nb_hist']
        self.nb_filters = config['nb_filters']
        self.kernel_size = config['kernel_size']
        self.nb_conv = config['nb_conv'] 
        self.learning_rate = config['learning_rate']
        self.use_separable = config['use_separable']
        self.nb_pre_conv = config['nb_pre_conv']
        self.pre_kernel_size = config['pre_kernel_size']
        self.pre_nb_filters = config['pre_nb_filters']
        self.pre_nb_conv = config['pre_nb_conv']
        
        self.data_dir = config['data_dir']
        self.y_suffix = config['y_suffix']
        self.save_dir = config['save_dir']
        self.save_prefix = config['save_prefix']
        self.ignore_boundaries = config['ignore_boundaries']
        self.batch_norm = config['batch_norm']
        self.verbose = config['verbose'] 
        self.batch_size = config['batch_size']
        self.nb_epoch = config['nb_epoch'] 
        self.reduce_lr = config['reduce_lr']
        self.reduce_lr_patience = config['reduce_lr_patience']
        self.fraction_data = config['fraction_data'] 
        self.seed = config['seed'] 
        self.batch_level_subsampling = config['batch_level_subsampling']
        self.tensorboard = config['tensorboard']
        self.log_messages = config['log_messages'] 
        self.nb_stacks = config['nb_stacks'] 
        self.with_y_hist = config['with_y_hist'] 
        self.x_suffix = config['x_suffix']
        
        self.early_stop_epoch = config['early_stop_epoch']
        self.gru_units = config['gru_units']
        self.dropout = config['dropout']
        self.neg = config['neg']
        self.steps = config['steps']
        
        self.sample_weight_mode = config['sample_weight_mode']
        self.return_sequences = config['return_sequences']
        self.unpack_channels = config['unpack_channels']
        self.nb_freq = config['nb_freq'] 
        self.nb_channels = config['nb_channels']  
        self.nb_classes = config['nb_classes']
        self.first_sample_train = config['first_sample_train'] 
        self.last_sample_train = config['last_sample_train']
        self.first_sample_val = config['first_sample_val']
        self.last_sample_val = config['last_sample_val']     
               
        # Create folder structure to save the model
        os.makedirs(os.path.abspath(self.save_dir), exist_ok = True)
        
        self.save_name = '{0}/{1}{2}'.format(self.save_dir, self.save_prefix, time.strftime('%Y%m%d_%H%M%S'))
        self.checkpoint_save_name = self.save_name + '_model.h5'       
               
        config['save_name'] = self.save_name
        config['checkpoint_save_name'] = self.checkpoint_save_name
        
        utils.save_params(config, self.save_name)
        
        self.data_gen = data_gen
        self.val_gen = val_gen

    @abstractmethod
    def load_prediction_configuration(self, config):
        """
        It loads the configuration parameters for the prediction.
        :param config: a dictionary with the configuration parameters.
        :return: instance will have the configuration parameters.
        """
        
        # Configuration
        self.model_name = config['FeatureEncoder_model']
        self.nb_hist = config['nb_hist']
        self.nb_filters = config['nb_filters']
        self.kernel_size = config['kernel_size']
        self.nb_conv = config['nb_conv'] 
        self.learning_rate = config['learning_rate']
        self.use_separable = config['use_separable']
        self.nb_pre_conv = config['nb_pre_conv']
        self.pre_kernel_size = config['pre_kernel_size']
        self.pre_nb_filters = config['pre_nb_filters']
        self.pre_nb_conv = config['pre_nb_conv']
        
        self.data_dir = config['data_dir']
        self.y_suffix = config['y_suffix']
        self.save_dir = config['save_dir']
        self.save_prefix = config['save_prefix']
        self.ignore_boundaries = config['ignore_boundaries']
        self.batch_norm = config['batch_norm']
        self.verbose = config['verbose'] 
        self.batch_size = config['batch_size']
        self.nb_epoch = config['nb_epoch'] 
        self.reduce_lr = config['reduce_lr']
        self.reduce_lr_patience = config['reduce_lr_patience']
        self.fraction_data = config['fraction_data'] 
        self.seed = config['seed'] 
        self.batch_level_subsampling = config['batch_level_subsampling']
        self.tensorboard = config['tensorboard']
        self.log_messages = config['log_messages'] 
        self.nb_stacks = config['nb_stacks'] 
        self.with_y_hist = config['with_y_hist'] 
        self.x_suffix = config['x_suffix']
        
        self.early_stop_epoch = config['early_stop_epoch']
        self.gru_units = config['gru_units']
        self.dropout = config['dropout']
        self.neg = config['neg']
        self.steps = config['steps']
        
        self.sample_weight_mode = config['sample_weight_mode']
        self.return_sequences = config['return_sequences']
        self.unpack_channels = config['unpack_channels']
        self.nb_freq = config['nb_freq'] 
        self.nb_channels = config['nb_channels']  
        self.nb_classes = config['nb_classes']
        self.first_sample_train = config['first_sample_train'] 
        self.last_sample_train = config['last_sample_train']
        self.first_sample_val = config['first_sample_val']
        self.last_sample_val = config['last_sample_val']        
        
        self.save_name = config['save_name']
        self.checkpoint_save_name = config['checkpoint_save_name']

    @abstractmethod
    def train(self):
        raise NotImplementedError('The model needs to overwrite the train method. The method should configure the learning process and callbacks, then fit the model.')

    @abstractmethod
    def predict(self, x_test, y_test, model, config):
        """
        It predicts the output for the test set.
        :param x_test: a numpy array with the input of the test set.
        :param y_test: a numpy array with the labels of the test set.
        :param model: a string with the name of the neural network model.
        :param config: a dictionary with the configuration parameters.
        :return: returns a full model.
        """
        
        logging.info('re-loading last best model')
        model.load_weights(config['checkpoint_save_name'])
        
        self.model = model
        
        return model

class Block(Model):
    """
    Super class for all the blocks so they have get_layer method. The method is used in prediction to extract features of the CPC encoder.
    """
    
    def __init__(self, name):
        super(Block, self).__init__(name = name)

    def get_layer(self, name = None, index = None):
        """
        Keras sourcecode for Model.
        :param name: name of the layer.
        :param index: index of the layer.
        :return: the layer if name or index is found, error otherwise.
        """
        
        if index is not None:
            if len(self.model.layers) <= index:
                raise ValueError('Was asked to retrieve layer at index ' + str(index) +
                                 ' but model only has ' + str(len(self.model.layers)) +
                                 ' layers.')
            
            else:
                return self.model.layers[index]
        
        else:
            if not name:
                raise ValueError('Provide either a layer name or layer index.')
            
            for layer in self.model.layers:
                if layer.name == name:
                    return layer
            
            raise ValueError('No such layer: ' + name)
            
class ContrastiveLoss(Block):
    """
    It creates the block that calculates the contrastive loss for given latent representation and context representations.
    Implementation from wav2vec (https://github.com/pytorch/fairseq/blob/master/fairseq/models/wav2vec.py).
    [wav2vec: Unsupervised Pre-training for Speech Recognition](https://arxiv.org/abs/1904.05862)
    """
    
    def __init__(self, context_units, neg, steps, name = 'Contrastive_Loss'):
        """
        :param context_units: number of units of the context representation.
        :param neg: number of negative samples.
        :param steps: number of steps to predict.
        :param name: name of the block, by default Contrastive_Loss.
        """
        
        super(ContrastiveLoss, self).__init__(name = name)
        
        self.context_units = context_units
        self.neg = neg
        self.steps = steps
        
        with K.name_scope(name):
            self.project_steps = Conv2DTranspose(self.steps, kernel_size = 1, strides = 1, name = 'project_steps')
            self.project_latent = Conv1D(self.context_units, kernel_size = 1, strides = 1, name = 'project_latent')
            self.cross_entropy = tf.keras.losses.CategoricalCrossentropy(from_logits = True, reduction = tf.keras.losses.Reduction.SUM)

    def get_negative_samples(self, true_features):
        """
        It calculates the negative samples re-ordering the time-steps of the true features.
        :param true_features: a tensor with the CPC predictions for the input.
        :return: a tensor with the negative samples.
        """
        
        # Shape S x T x F
        samples = K.shape(true_features)[0]
        timesteps = K.shape(true_features)[1]
        features = K.shape(true_features)[2]

        # New shape F x S x T
        true_features = K.permute_dimensions(true_features, pattern = (2, 0, 1))
        # New shape F x (S x T)
        true_features = K.reshape(true_features, (features, -1))

        high = timesteps

        # New order for time-steps
        indices = tf.repeat(tf.expand_dims(tf.range(timesteps), axis = -1), self.neg)
        neg_indices = tf.random.uniform(shape = (samples, self.neg * timesteps), minval = 0, maxval = high - 1, dtype = tf.dtypes.int32)
        neg_indices = tf.where(tf.greater_equal(neg_indices, indices), neg_indices + 1, neg_indices)

        right_indices = tf.reshape(tf.range(samples), (-1, 1)) * high
        neg_indices = neg_indices + right_indices

        # Reorder for negative samples
        negative_samples = tf.gather(true_features, tf.reshape(neg_indices, [-1]), axis = 1)
        negative_samples = K.permute_dimensions(K.reshape(negative_samples, (features, samples, self.neg, timesteps)), (2, 1, 3, 0))
        
        return negative_samples

    def call(self, inputs, **kwargs):
        """
        :param inputs: a list with two elements, the latent representation and the context representation.
        :return: the contrastive loss calculated.
        """
        
        true_latent, context_latent = inputs

        # Linear transformation of latent representation into the vector space of context representations
        true_latent = self.project_latent(true_latent)

        # Calculate the following steps using context_latent
        context_latent = K.expand_dims(context_latent, -1)
        predictions = self.project_steps(context_latent)

        negative_samples = self.get_negative_samples(true_latent)

        true_latent = K.expand_dims(true_latent, 0)

        targets = K.concatenate([true_latent, negative_samples], 0)
        copies = self.neg + 1  # Total of samples in targets

        # samples, timesteps, features, steps = predictions.shape

        # Logits calculated from predictions and targets
        logits = None
        for i in range(self.steps):
            if i == 0:
                # The time-steps are correspondent as is the first step
                logits = tf.reshape(tf.einsum('stf,cstf->tsc', predictions[:, :, :, i], targets[:, :, :, :]), [-1])
            
            else:
                # We need to match the time-step taking into account the step for which is being predicted
                logits = tf.concat([logits, tf.reshape(tf.einsum('stf,cstf->tsc', predictions[:, :-i, :, i], targets[:, :, i:, :]), [-1])], 0)

        logits = tf.reshape(logits, (-1, copies))
        total_points = tf.shape(logits)[0]

        # Labels, this should be the true value, that is 1.0 for the first copy (positive sample) and 0.0 for the rest
        label_idx = [True] + [False] * self.neg
        labels = tf.where(label_idx, tf.ones((total_points, copies)), tf.zeros((total_points, copies)))

        # The loss is the softmax_cross_entropy_with_logits sum over all steps
        loss = self.cross_entropy(labels, logits)
        loss = tf.reshape(loss, (1,))
        
        return loss

    def get_config(self):
        return {'context_units': self.context_units, 'neg': self.neg, 'steps': self.steps}

class FeatureEncoder_tcn_seq(Block):
    """
    It creates a tcn_seq model for the encoder part (latent representations).
    """
    
    def __init__(self, nb_freq: int, nb_classes: int, nb_hist: int = 1, nb_filters: int = 16, kernel_size: int = 3,
                 nb_conv: int = 1, loss: str = "categorical_crossentropy",
                 dilations: List[int] = [1, 2, 4, 8, 16], activation: str = 'norm_relu',
                 use_skip_connections: bool = True, return_sequences: bool = True,
                 dropout_rate: float = 0.00, padding: str = 'same', sample_weight_mode: str = None,
                 nb_pre_conv: int = 0, learning_rate: float = 0.0001, out_activation: str = 'softmax',
                 use_separable: bool = False, name: str = 'FeatureEncoder_tcn_seq', **kwignored):

        super(FeatureEncoder_tcn_seq, self).__init__(name = name)
        
        self.nb_freq = nb_freq
        self.nb_classes = nb_classes
        self.nb_hist = nb_hist
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.nb_conv = nb_conv
        self.loss = loss
        self.dilations = dilations
        self.activation = activation
        self.use_skip_connections = use_skip_connections
        self.return_sequences = return_sequences
        self.dropout_rate = dropout_rate
        self.padding = padding
        self.sample_weight_mode = sample_weight_mode
        self.nb_pre_conv = nb_pre_conv
        self.learning_rate = learning_rate
        self.out_activation = out_activation
        self.use_separable = use_separable
                
        self.model = tcn_seq(self.nb_freq, self.nb_classes, self.nb_hist, self.nb_filters, self.kernel_size,
                             self.nb_conv, self.loss,
                             self.dilations, self.activation,
                             self.use_skip_connections, self.return_sequences,
                             self.dropout_rate, self.padding, self.sample_weight_mode,
                             self.nb_pre_conv, self.learning_rate, self.out_activation,
                             self.use_separable, **kwignored)

    def call(self, inputs, **kwargs):
        """
        It is executed when an input tensor is passed.
        :param inputs: a tensor with the input features.
        :return: a tensor with the output of the block.
        """
        
        features = inputs
        for layer in self.model.layers:
            features = layer(features)
        
        return features

    def get_config(self):
        
        return {'nb_freq': self.nb_freq, 'nb_classes': self.nb_classes, 'nb_hist': self.nb_hist, 'nb_filters': self.nb_filters, 'kernel_size': self.kernel_size,
                'nb_conv': self.nb_conv, 'loss': self.loss,
                'dilations': self.dilations, 'activation': self.activation,
                'use_skip_connections': self.use_skip_connections, 'return_sequences': self.return_sequences,
                'dropout_rate': self.dropout_rate, 'padding': self.padding, 'sample_weight_mode': self.sample_weight_mode,
                'nb_pre_conv': self.nb_pre_conv, 'learning_rate': self.learning_rate, 'out_activation': self.out_activation,
                'use_separable': self.use_separable}
    
class FeatureEncoder_tcn(Block):
    """
    It creates a tcn model for the encoder part (latent representations).
    """
    
    def __init__(self, nb_freq: int, nb_classes: int, nb_hist: int = 1, nb_filters: int = 16, kernel_size: int = 3,
                 nb_conv: int = 1, loss: str = "categorical_crossentropy",
                 dilations: List[int] = [1, 2, 4, 8, 16], activation: str = 'norm_relu',
                 use_skip_connections: bool = True, return_sequences: bool = True,
                 dropout_rate: float = 0.00, padding: str = 'same', sample_weight_mode: str = None,
                 nb_pre_conv: int = 0, learning_rate: float = 0.0001, out_activation: str = 'softmax',
                 use_separable: bool = False, name: str = 'FeatureEncoder_tcn', **kwignored):

        super(FeatureEncoder_tcn, self).__init__(name = name)
        
        self.nb_freq = nb_freq
        self.nb_classes = nb_classes
        self.nb_hist = nb_hist
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.nb_conv = nb_conv
        self.loss = loss
        self.dilations = dilations
        self.activation = activation
        self.use_skip_connections = use_skip_connections
        self.return_sequences = return_sequences
        self.dropout_rate = dropout_rate
        self.padding = padding
        self.sample_weight_mode = sample_weight_mode
        self.nb_pre_conv = nb_pre_conv
        self.learning_rate = learning_rate
        self.out_activation = out_activation
        self.use_separable = use_separable
                
        self.model = tcn(self.nb_freq, self.nb_classes, self.nb_hist, self.nb_filters, self.kernel_size,
                         self.nb_conv, self.loss,
                         self.dilations, self.activation,
                         self.use_skip_connections, self.return_sequences,
                         self.dropout_rate, self.padding, self.sample_weight_mode,
                         self.nb_pre_conv, self.learning_rate, self.out_activation,
                         self.use_separable, **kwignored)

    def call(self, inputs, **kwargs):
        """
        It is executed when an input tensor is passed.
        :param inputs: a tensor with the input features.
        :return: a tensor with the output of the block.
        """
        
        features = inputs
        for layer in self.model.layers:
            features = layer(features)
        
        return features

    def get_config(self):
        
        return {'nb_freq': self.nb_freq, 'nb_classes': self.nb_classes, 'nb_hist': self.nb_hist, 'nb_filters': self.nb_filters, 'kernel_size': self.kernel_size,
                'nb_conv': self.nb_conv, 'loss': self.loss,
                'dilations': self.dilations, 'activation': self.activation,
                'use_skip_connections': self.use_skip_connections, 'return_sequences': self.return_sequences,
                'dropout_rate': self.dropout_rate, 'padding': self.padding, 'sample_weight_mode': self.sample_weight_mode,
                'nb_pre_conv': self.nb_pre_conv, 'learning_rate': self.learning_rate, 'out_activation': self.out_activation,
                'use_separable': self.use_separable}
    
class FeatureEncoder_tcn_tcn(Block):
    """
    It creates a tcn_tcn model for the encoder part (latent representations).
    """
    
    def __init__(self, nb_freq: int, nb_classes: int, nb_hist: int = 1, nb_filters: int = 16, kernel_size: int = 3,
                 nb_conv: int = 1, loss: str = "categorical_crossentropy",
                 dilations: List[int] = [1, 2, 4, 8, 16], activation: str = 'norm_relu',
                 use_skip_connections: bool = True, return_sequences: bool = True,
                 dropout_rate: float = 0.00, padding: str = 'same', sample_weight_mode: str = None,
                 nb_pre_conv: int = 0, learning_rate: float = 0.0005, upsample: bool = True,
                 use_separable: bool = False, name: str = 'FeatureEncoder_tcn_tcn', **kwignored):

        super(FeatureEncoder_tcn_tcn, self).__init__(name = name)
        
        self.nb_freq = nb_freq
        self.nb_classes = nb_classes
        self.nb_hist = nb_hist
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.nb_conv = nb_conv
        self.loss = loss
        self.dilations = dilations
        self.activation = activation
        self.use_skip_connections = use_skip_connections
        self.return_sequences = return_sequences
        self.dropout_rate = dropout_rate
        self.padding = padding
        self.sample_weight_mode = sample_weight_mode
        self.nb_pre_conv = nb_pre_conv
        self.learning_rate = learning_rate
        self.upsample = upsample
        self.use_separable = use_separable
                
        self.model = tcn_tcn(self.nb_freq, self.nb_classes, self.nb_hist, self.nb_filters, self.kernel_size,
                             self.nb_conv, self.loss,
                             self.dilations, self.activation,
                             self.use_skip_connections, self.return_sequences,
                             self.dropout_rate, self.padding, self.sample_weight_mode,
                             self.nb_pre_conv, self.learning_rate, self.upsample,
                             self.use_separable, **kwignored)

    def call(self, inputs, **kwargs):
        """
        It is executed when an input tensor is passed.
        :param inputs: a tensor with the input features.
        :return: a tensor with the output of the block.
        """
        
        features = inputs
        for layer in self.model.layers:
            features = layer(features)
        
        return features

    def get_config(self):
        
        return {'nb_freq': self.nb_freq, 'nb_classes': self.nb_classes, 'nb_hist': self.nb_hist, 'nb_filters': self.nb_filters, 'kernel_size': self.kernel_size,
                'nb_conv': self.nb_conv, 'loss': self.loss,
                'dilations': self.dilations, 'activation': self.activation,
                'use_skip_connections': self.use_skip_connections, 'return_sequences': self.return_sequences,
                'dropout_rate': self.dropout_rate, 'padding': self.padding, 'sample_weight_mode': self.sample_weight_mode,
                'nb_pre_conv': self.nb_pre_conv, 'learning_rate': self.learning_rate, 'upsample': self.upsample,
                'use_separable': self.use_separable}

class FeatureEncoder_tcn_small(Block):
    """
    It creates a tcn_small model for the encoder part (latent representations).
    """
    
    def __init__(self, nb_freq: int, nb_classes: int, nb_hist: int = 1, nb_filters: int = 16, kernel_size: int = 3,
                 nb_conv: int = 1, loss: str = "categorical_crossentropy",
                 dilations: List[int] = [1, 2, 4, 8, 16], activation: str = 'norm_relu',
                 use_skip_connections: bool = True, return_sequences: bool = True,
                 dropout_rate: float = 0.00, padding: str = 'same', sample_weight_mode: str = None,
                 nb_pre_conv: int = 0, learning_rate: float = 0.0005, upsample: bool = True,
                 use_separable: bool = False, name: str = 'FeatureEncoder_tcn_small', **kwignored):

        super(FeatureEncoder_tcn_small, self).__init__(name = name)
        
        self.nb_freq = nb_freq
        self.nb_classes = nb_classes
        self.nb_hist = nb_hist
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.nb_conv = nb_conv
        self.loss = loss
        self.dilations = dilations
        self.activation = activation
        self.use_skip_connections = use_skip_connections
        self.return_sequences = return_sequences
        self.dropout_rate = dropout_rate
        self.padding = padding
        self.sample_weight_mode = sample_weight_mode
        self.nb_pre_conv = nb_pre_conv
        self.learning_rate = learning_rate
        self.upsample = upsample
        self.use_separable = use_separable
                
        self.model = tcn_small(self.nb_freq, self.nb_classes, self.nb_hist, self.nb_filters, self.kernel_size,
                               self.nb_conv, self.loss,
                               self.dilations, self.activation,
                               self.use_skip_connections, self.return_sequences,
                               self.dropout_rate, self.padding, self.sample_weight_mode,
                               self.nb_pre_conv, self.learning_rate, self.upsample,
                               self.use_separable, **kwignored)

    def call(self, inputs, **kwargs):
        """
        It is executed when an input tensor is passed.
        :param inputs: a tensor with the input features.
        :return: a tensor with the output of the block.
        """
        
        features = inputs
        for layer in self.model.layers:
            features = layer(features)
        
        return features

    def get_config(self):
        
        return {'nb_freq': self.nb_freq, 'nb_classes': self.nb_classes, 'nb_hist': self.nb_hist, 'nb_filters': self.nb_filters, 'kernel_size': self.kernel_size,
                'nb_conv': self.nb_conv, 'loss': self.loss,
                'dilations': self.dilations, 'activation': self.activation,
                'use_skip_connections': self.use_skip_connections, 'return_sequences': self.return_sequences,
                'dropout_rate': self.dropout_rate, 'padding': self.padding, 'sample_weight_mode': self.sample_weight_mode,
                'nb_pre_conv': self.nb_pre_conv, 'learning_rate': self.learning_rate, 'upsample': self.upsample,
                'use_separable': self.use_separable}
    
class FeatureEncoder_tcn_stft(Block):
    """
    It creates a tcn_stft model for the encoder part (latent representations).
    """
    
    def __init__(self, nb_freq: int, nb_classes: int, nb_hist: int = 1, nb_filters: int = 16, kernel_size: int = 3,
                 nb_conv: int = 1, loss: str = "categorical_crossentropy",
                 dilations: List[int] = [1, 2, 4, 8, 16], activation: str = 'norm_relu',
                 use_skip_connections: bool = True, return_sequences: bool = True,
                 dropout_rate: float = 0.00, padding: str = 'same', sample_weight_mode: str = None,
                 nb_pre_conv: int = 0, learning_rate: float = 0.0005, upsample: bool = True,
                 use_separable: bool = False, name: str = 'FeatureEncoder_tcn_stft', **kwignored):

        super(FeatureEncoder_tcn_stft, self).__init__(name = name)
        
        self.nb_freq = nb_freq
        self.nb_classes = nb_classes
        self.nb_hist = nb_hist
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.nb_conv = nb_conv
        self.loss = loss
        self.dilations = dilations
        self.activation = activation
        self.use_skip_connections = use_skip_connections
        self.return_sequences = return_sequences
        self.dropout_rate = dropout_rate
        self.padding = padding
        self.sample_weight_mode = sample_weight_mode
        self.nb_pre_conv = nb_pre_conv
        self.learning_rate = learning_rate
        self.upsample = upsample
        self.use_separable = use_separable
                
        self.model = tcn_stft(self.nb_freq, self.nb_classes, self.nb_hist, self.nb_filters, self.kernel_size,
                              self.nb_conv, self.loss,
                              self.dilations, self.activation,
                              self.use_skip_connections, self.return_sequences,
                              self.dropout_rate, self.padding, self.sample_weight_mode,
                              self.nb_pre_conv, self.learning_rate, self.upsample,
                              self.use_separable, **kwignored)

    def call(self, inputs, **kwargs):
        """
        It is executed when an input tensor is passed.
        :param inputs: a tensor with the input features.
        :return: a tensor with the output of the block.
        """
        
        features = inputs
        for layer in self.model.layers:
            features = layer(features)
        
        return features

    def get_config(self):
        
        return {'nb_freq': self.nb_freq, 'nb_classes': self.nb_classes, 'nb_hist': self.nb_hist, 'nb_filters': self.nb_filters, 'kernel_size': self.kernel_size,
                'nb_conv': self.nb_conv, 'loss': self.loss,
                'dilations': self.dilations, 'activation': self.activation,
                'use_skip_connections': self.use_skip_connections, 'return_sequences': self.return_sequences,
                'dropout_rate': self.dropout_rate, 'padding': self.padding, 'sample_weight_mode': self.sample_weight_mode,
                'nb_pre_conv': self.nb_pre_conv, 'learning_rate': self.learning_rate, 'upsample': self.upsample,
                'use_separable': self.use_separable}
    
class FeatureEncoder_tcn_multi(Block):
    """
    It creates a tcn_multi model for the encoder part (latent representations).
    """
    
    def __init__(self, nb_freq: int, nb_classes: int, nb_hist: int = 1, nb_filters: int = 16, kernel_size: int = 3,
                 nb_conv: int = 1, loss: str = "categorical_crossentropy",
                 dilations: List[int] = [1, 2, 4, 8, 16], activation: str = 'norm_relu',
                 use_skip_connections: bool = True, return_sequences: bool = True,
                 dropout_rate: float = 0.00, padding: str = 'same', sample_weight_mode: str = None,
                 learning_rate: float = 0.0005, use_separable: bool = False, 
                 pre_kernel_size: int = 16, pre_nb_filters: int = 16, pre_nb_conv: int = 2,
                 name: str = 'FeatureEncoder_tcn_multi', **kwignored):

        super(FeatureEncoder_tcn_multi, self).__init__(name = name)
        
        self.nb_freq = nb_freq
        self.nb_classes = nb_classes
        self.nb_hist = nb_hist
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.nb_conv = nb_conv
        self.loss = loss
        self.dilations = dilations
        self.activation = activation
        self.use_skip_connections = use_skip_connections
        self.return_sequences = return_sequences
        self.dropout_rate = dropout_rate
        self.padding = padding
        self.sample_weight_mode = sample_weight_mode
        self.learning_rate = learning_rate
        self.use_separable = use_separable
        self.pre_kernel_size = pre_kernel_size
        self.pre_nb_filters = pre_nb_filters
        self.pre_nb_conv = pre_nb_conv
                
        self.model = tcn_multi(self.nb_freq, self.nb_classes, self.nb_hist, self.nb_filters, self.kernel_size,
                               self.nb_conv, self.loss,
                               self.dilations, self.activation,
                               self.use_skip_connections, self.return_sequences,
                               self.dropout_rate, self.padding, self.sample_weight_mode,
                               self.learning_rate, self.use_separable,
                               self.pre_kernel_size, self.pre_nb_filters, self.pre_nb_conv, 
                               **kwignored)

    def call(self, inputs, **kwargs):
        """
        It is executed when an input tensor is passed.
        :param inputs: a tensor with the input features.
        :return: a tensor with the output of the block.
        """
        
        features = inputs
        for layer in self.model.layers:
            features = layer(features)
        
        return features

    def get_config(self):
        
        return {'nb_freq': self.nb_freq, 'nb_classes': self.nb_classes, 'nb_hist': self.nb_hist, 'nb_filters': self.nb_filters, 'kernel_size': self.kernel_size,
                'nb_conv': self.nb_conv, 'loss': self.loss,
                'dilations': self.dilations, 'activation': self.activation,
                'use_skip_connections': self.use_skip_connections, 'return_sequences': self.return_sequences,
                'dropout_rate': self.dropout_rate, 'padding': self.padding, 'sample_weight_mode': self.sample_weight_mode,
                'learning_rate': self.learning_rate, 'use_separable': self.use_separable,
                'pre_kernel_size': self.pre_kernel_size, 'pre_nb_filters': self.pre_nb_filters, 'pre_nb_conv': self.pre_nb_conv}

class CPCModel(ModelBase):
    
    def load_training_configuration(self, config, data_gen, val_gen):
        """
        It instantiates the model architecture using the configuration parameters.
        :param config: a dictionary with the configuration for training.
        :param data_gen: a dictionary with the training set features.
        :param val_gen: a dictionary with the validation set features.
        :return: instance will have the parameters from configuration and the model architecture.
        """
        
        super(CPCModel, self).load_training_configuration(config, data_gen, val_gen)

        # Model architecture: Feature_Encoder -> Dropout -> GRU -> Dropout
       
        # Configuration
        self.model_name = config['FeatureEncoder_model']
        self.nb_hist = config['nb_hist']
        self.nb_filters = config['nb_filters']
        self.kernel_size = config['kernel_size']
        self.nb_conv = config['nb_conv'] 
        self.learning_rate = config['learning_rate']
        self.use_separable = config['use_separable']
        self.nb_pre_conv = config['nb_pre_conv']
        self.pre_kernel_size = config['pre_kernel_size']
        self.pre_nb_filters = config['pre_nb_filters']
        self.pre_nb_conv = config['pre_nb_conv']
        
        self.data_dir = config['data_dir']
        self.y_suffix = config['y_suffix']
        self.save_dir = config['save_dir']
        self.save_prefix = config['save_prefix']
        self.ignore_boundaries = config['ignore_boundaries']
        self.batch_norm = config['batch_norm']
        self.verbose = config['verbose'] 
        self.batch_size = config['batch_size']
        self.nb_epoch = config['nb_epoch'] 
        self.reduce_lr = config['reduce_lr']
        self.reduce_lr_patience = config['reduce_lr_patience']
        self.fraction_data = config['fraction_data'] 
        self.seed = config['seed'] 
        self.batch_level_subsampling = config['batch_level_subsampling']
        self.tensorboard = config['tensorboard']
        self.log_messages = config['log_messages'] 
        self.nb_stacks = config['nb_stacks'] 
        self.with_y_hist = config['with_y_hist'] 
        self.x_suffix = config['x_suffix']
        
        self.early_stop_epoch = config['early_stop_epoch']
        self.gru_units = config['gru_units']
        self.dropout = config['dropout']
        self.neg = config['neg']
        self.steps = config['steps']
        
        self.sample_weight_mode = config['sample_weight_mode']
        self.return_sequences = config['return_sequences']
        self.unpack_channels = config['unpack_channels']
        self.nb_freq = config['nb_freq'] 
        self.nb_channels = config['nb_channels']  
        self.nb_classes = config['nb_classes']
        self.first_sample_train = config['first_sample_train'] 
        self.last_sample_train = config['last_sample_train']
        self.first_sample_val = config['first_sample_val']
        self.last_sample_val = config['last_sample_val']
        
        self.save_name = config['save_name']
        self.checkpoint_save_name = config['checkpoint_save_name']
        
        self.data_gen = data_gen
        self.val_gen = val_gen
                
        if self.model_name == 'tcn_seq' or self.model_name == 'tcn':
            feature_encoder = eval(self.model_name)(nb_freq = self.nb_freq, nb_classes = self.nb_classes, nb_hist = self.nb_hist, nb_filters = self.nb_filters, kernel_size = self.kernel_size,
                                                    nb_conv = self.nb_conv,
                                                    return_sequences = self.return_sequences,
                                                    sample_weight_mode = self.sample_weight_mode,
                                                    nb_pre_conv = self.nb_pre_conv, learning_rate = self.learning_rate,
                                                    use_separable = self.use_separable)
                      
        elif self.model_name == 'tcn_tcn' or self.model_name == 'tcn_small' or self.model_name == 'tcn_stft':
            feature_encoder = eval(self.model_name)(nb_freq = self.nb_freq, nb_classes = self.nb_classes, nb_hist = self.nb_hist, nb_filters = self.nb_filters, kernel_size = self.kernel_size,
                                                    nb_conv = self.nb_conv,
                                                    return_sequences = self.return_sequences,
                                                    sample_weight_mode = self.sample_weight_mode,
                                                    nb_pre_conv = self.nb_pre_conv, learning_rate = self.learning_rate,
                                                    use_separable = self.use_separable)
            
        elif self.model_name == 'tcn_multi':            
            feature_encoder = eval(self.model_name)(nb_freq = self.nb_freq, nb_classes = self.nb_classes, nb_hist = self.nb_hist, nb_filters = self.nb_filters, kernel_size = self.kernel_size,
                                                    nb_conv = self.nb_conv,
                                                    return_sequences = self.return_sequences,
                                                    sample_weight_mode = self.sample_weight_mode,
                                                    learning_rate = self.learning_rate, use_separable = self.use_separable,
                                                    pre_kernel_size = self.pre_kernel_size, pre_nb_filters = self.pre_nb_filters, pre_nb_conv = self.pre_nb_conv)

        # Autoregressive model params
        self.gru_units = config['gru_units']
        
        # Dropout params
        self.dropout = config['dropout']

        # Contrastive loss params
        self.neg = config['neg']
        self.steps = config['steps']

        # Input shape
        self.input_shape = (self.nb_hist, self.nb_freq)
        self.features = 1

        # Input tensor
        input_feats = Input(shape = self.input_shape, name = 'input_layer')

        # Dropout layer
        dropout_layer = Dropout(self.dropout, name = 'dropout_block')

        # Feature Encoder
        encoder_features = feature_encoder(input_feats)
        encoder_output = dropout_layer(encoder_features)

        # Autoregressive model
        autoregressive_model = GRU(self.gru_units, return_sequences = True, name = 'autoregressive_layer')
        autoregressive_output = autoregressive_model(encoder_output)
        autoregressive_output = dropout_layer(autoregressive_output)

        # Contrastive loss
        contrastive_loss = ContrastiveLoss(self.gru_units, self.neg, self.steps)
        contrastive_loss_output = contrastive_loss([encoder_features, autoregressive_output])

        # Self-supervised Contrastive Predictive Coding Model
        self.model = Model(input_feats, contrastive_loss_output)
        
        print(self.model.summary())
                
        utils.save_params(config, self.save_name)
  
    def load_prediction_configuration(self, config):
        """
        It loads the configuration parameters for the prediction.
        :param config: a dictionary with the configuration parameters.
        :return: instance will have the configuration parameters.
        """
        
        super(CPCModel, self).load_prediction_configuration(config)

    def train(self):
        """
        Train a self-supervised Contrastive Predictive Coding (CPC) model.
        :return: a trained model saved on disk.
        """
        
        adam = Adam(lr = self.learning_rate)
        self.model.compile(optimizer = adam, loss = {'Contrastive_Loss': lambda y_true, y_pred: y_pred})
          
        # Callbacks for training
        callbacks = [ModelCheckpoint(self.checkpoint_save_name, save_best_only = True, save_weights_only = False, monitor = 'val_loss', verbose = self.verbose), EarlyStopping(monitor = 'val_loss', patience = self.early_stop_epoch)]
        
        if self.reduce_lr:
            callbacks.append(ReduceLROnPlateau(patience = self.reduce_lr_patience, verbose = self.verbose))
        if self.tensorboard:
            callbacks.append(TensorBoard(log_dir = self.save_dir))
    
        # Train network
        logging.info('start training')
        fit_hist = self.model.fit(self.data_gen,                                 
                                  # steps_per_epoch = min(len(self.data_gen), 1000),
                                  steps_per_epoch = len(self.data_gen) // self.batch_size,
                                  validation_data = self.val_gen,
                                  validation_steps = len(self.val_gen) // self.batch_size,
                                  epochs = self.nb_epoch,
                                  verbose = self.verbose,
                                  callbacks = callbacks)
        
        save_filename = "{0}_fit_hist.h5".format(self.save_name)
        logging.info('saving to ' + save_filename)
        d = {'fit_hist': fit_hist}
    
        fl.save(save_filename, d)

    def predict(self, x_test, y_test, model, config):        
        """
        It predicts the output for the test set.
        :param x_test: a numpy array with the input of the test set.
        :param y_test: a numpy array with the labels of the test set.
        :param model: a string with the name of the neural network model.
        :param config: a dictionary with the configuration parameters.
        :return: returns a full model.
        """

        self.class_names = config['class_names']        

        logging.info('re-loading last best model')
        model.load_weights(self.checkpoint_save_name)
    
        logging.info('predicting')
        x_test, y_test, y_pred = evaluate.evaluate_probabilities(x = x_test, y = y_test, model = model, params = config)
    
        labels_test = predict.labels_from_probabilities(y_test)
        labels_pred = predict.labels_from_probabilities(y_pred)
    
        logging.info('evaluating')
        conf_mat, report = evaluate.evaluate_segments(labels_test, labels_pred, self.class_names)
        logging.info(conf_mat)
        logging.info(report)
    
        save_filename = "{0}_results.h5".format(self.save_name)
        logging.info('saving to ' + save_filename)
        d = {'confusion_matrix': conf_mat,
             'classification_report': report,
             'x_test': x_test,
             'y_test': y_test,
             'y_pred': y_pred,
             'labels_test': labels_test,
             'labels_pred': labels_pred,
             'params': config}
    
        fl.save(save_filename, d)
        
        return self.model