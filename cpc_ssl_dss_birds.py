import flammkuchen as fl
import logging
import os
import tensorflow as tf
import time

import evaluate, predict, utils

from abc import ABC, abstractmethod
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.layers import Activation, AveragePooling1D, BatchNormalization, concatenate, Conv1D, Conv2DTranspose, Dense, Dropout, GRU, Input, MaxPooling1D, Reshape, UpSampling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import tcn as tcn_layer
import tcn_new as tcn_new_layer
from time_frequency import Spectrogram

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable = True)

try:
    from tensorflow.python.framework.ops import disable_eager_execution
    disable_eager_execution()
except Exception as e:
    logging.exception(e)
    
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
        self.model_name = config['model_name']
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
                           
        self.dilations = config['dilations']
        self.activation = config['activation']
        self.use_skip_connections = config['use_skip_connections']
        self.dropout_rate = config['dropout']
        self.padding = config['padding']
        
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
        self.class_names = config['class_names']
        self.first_sample_train = config['first_sample_train'] 
        self.last_sample_train = config['last_sample_train']
        self.first_sample_val = config['first_sample_val']
        self.last_sample_val = config['last_sample_val']     
                       
        self.data_gen = data_gen
        self.val_gen = val_gen

    @abstractmethod
    def load_prediction_configuration(self, model, config, test_gen):
        """
        It loads the configuration parameters from the configuration dictionary and the input/output features for prediction.
        :param model: a dictionary with the pretrained model.
        :param config: a dictionary with the configuration for predcition.
        :param test_gen: a dictionary with the test set features.
        :return: instance will have the configuration parameters.
        """
        
        # Configuration
        self.model_name = config['model_name']
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
        
        self.dilations = config['dilations']
        self.activation = config['activation']
        self.use_skip_connections = config['use_skip_connections']
        self.dropout_rate = config['dropout']
        self.padding = config['padding']
        
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
        self.class_names = config['class_names']
        self.first_sample_train = config['first_sample_train'] 
        self.last_sample_train = config['last_sample_train']
        self.first_sample_val = config['first_sample_val']
        self.last_sample_val = config['last_sample_val']        
                
        self.model_training = model
        self.test_gen = test_gen
        
    @abstractmethod
    def train(self, config):
        raise NotImplementedError('The model needs to overwrite the train method. The method should configure the learning process and callbacks, then fit the model.')

    @abstractmethod
    def predict(self, config):
        raise NotImplementedError('The model needs to overwrite the predict method. The method should configure the prediction process.')

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
        It calculates the negative samples by re-ordering the time-steps of the true features.
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
        :return: the calculated contrastive loss.
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

class CPCModel(ModelBase):
    
    def load_training_configuration(self, config, data_gen, val_gen):
        """
        It instantiates the model architecture using the configuration parameters.
        :param config: a dictionary with the configuration for training.
        :param data_gen: a dictionary with the training set features.
        :param val_gen: a dictionary with the validation set features.
        :return: a model used for training.
        """
        
        super(CPCModel, self).load_training_configuration(config, data_gen, val_gen)

        # Model architecture: Feature_Encoder -> Dropout -> GRU -> Dropout
       
        # Configuration
        self.model_name = config['model_name']
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
        
        self.dilations = config['dilations']
        self.activation = config['activation']
        self.use_skip_connections = config['use_skip_connections']
        self.dropout_rate = config['dropout']
        self.padding = config['padding']
        
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
        self.class_names = config['class_names']
        self.first_sample_train = config['first_sample_train'] 
        self.last_sample_train = config['last_sample_train']
        self.first_sample_val = config['first_sample_val']
        self.last_sample_val = config['last_sample_val']
        
        self.data_gen = data_gen
        self.val_gen = val_gen
        
        if self.model_name == 'tcn_multi':            
            # Define the per-channel model
            nb_channels = self.nb_freq
            channels_in = []
            for chan in range(self.nb_channels):
                channels_in.append(Input(shape = (self.nb_hist, 1), name = "channel_{0}".format(chan)))
            
        else:
            input_layer = Input(shape = (self.nb_hist, self.nb_freq))
            out = input_layer
            
        if self.model_name == 'tcn_seq' or self.model_name == 'tcn':
            """Create TCN network."""

            for conv in range(self.nb_pre_conv):
                out = Conv1D(self.nb_filters * (2 ** conv), self.kernel_size, padding = 'same', activation = 'relu')(out)
                out = BatchNormalization()(out)
                out = MaxPooling1D(min(int(out.shape[1]), 2))(out)
        
            x = tcn_layer.TCN(nb_filters = self.nb_filters, kernel_size = self.kernel_size, nb_stacks = self.nb_conv, dilations = self.dilations, activation = self.activation,
                              use_skip_connections = self.use_skip_connections, padding = self.padding, dropout_rate = self.dropout_rate, return_sequences = self.return_sequences,
                              use_separable = self.use_separable)(out)
                      
        elif self.model_name == 'tcn_tcn':
            """Create TCN network with TCN layer as pre-processing and downsampling frontend."""
            
            if self.nb_pre_conv > 0:
                out = tcn_layer.TCN(nb_filters = self.nb_filters, kernel_size = self.kernel_size, nb_stacks = self.nb_pre_conv, dilations = self.dilations,
                                    activation = self.activation, use_skip_connections = self.use_skip_connections, padding = self.padding,
                                    dropout_rate = self.dropout_rate, return_sequences = self.return_sequences,
                                    use_separable = self.use_separable, name = 'frontend')(out)
                
                out = MaxPooling1D(pool_size = 2 ** self.nb_pre_conv, strides = 2 ** self.nb_pre_conv)(out)
        
            x = tcn_layer.TCN(nb_filters = self.nb_filters, kernel_size = self.kernel_size, nb_stacks = self.nb_conv, dilations = self.dilations,
                              activation = self.activation, use_skip_connections = self.use_skip_connections, padding = self.padding,
                              dropout_rate = self.dropout_rate, return_sequences = self.return_sequences,
                              use_separable = self.use_separable)(out)
            
        elif self.model_name == 'tcn_small':
            """Create TCN network with TCN layer as pre-processing and downsampling frontend."""
            
            if self.nb_pre_conv > 0:
                out = tcn_layer.TCN(nb_filters = 32, kernel_size = 3, nb_stacks = 1, dilations = self.dilations,
                                    activation = self.activation, use_skip_connections = self.use_skip_connections, padding = self.padding,
                                    dropout_rate = self.dropout_rate, return_sequences = self.return_sequences,
                                    use_separable = self.use_separable, name = 'frontend')(out)
                
                out = MaxPooling1D(pool_size = 2 ** self.nb_pre_conv, strides = 2 ** self.nb_pre_conv)(out)
        
            x = tcn_layer.TCN(nb_filters = self.nb_filters, kernel_size = self.kernel_size, nb_stacks = self.nb_conv, dilations = self.dilations,
                              activation = self.activation, use_skip_connections = self.use_skip_connections, padding = self.padding,
                              dropout_rate = self.dropout_rate, return_sequences = self.return_sequences,
                              use_separable = self.use_separable)(out)
               
        elif self.model_name == 'tcn_stft':
            """Create TCN network with trainable STFT layer as pre-processing and downsampling frontend."""
            
            if self.nb_freq > 1:
                raise ValueError(f'This model only works with single channel data but last dim of inputs has len {self.nb_freq} (should be 1).')
            
            if self.nb_pre_conv > 0:
                out = Spectrogram(n_dft = 64, n_hop = 2 ** self.nb_pre_conv,
                                  return_decibel_spectrogram = True, power_spectrogram = 1.0,
                                  trainable_kernel = True, name = 'trainable_stft', image_data_format = 'channels_last')(out)
                
                out = Reshape((out.shape[1], out.shape[2] * out.shape[3]))(out)
        
            x = tcn_layer.TCN(nb_filters = self.nb_filters, kernel_size = self.kernel_size, nb_stacks = self.nb_conv, dilations = self.dilations,
                              activation = self.activation, use_skip_connections = self.use_skip_connections, padding = self.padding,
                              dropout_rate = self.dropout_rate, return_sequences = self.return_sequences,
                              use_separable = self.use_separable)(out)
                   
        elif self.model_name == 'tcn_multi':
            """Create TCN network with TCN layer as pre-processing and downsampling frontend with weights shared between channels."""
      
            # Channel model will be shared, weights and all
            channel_model = tcn_new_layer.TCN(nb_filters = self.pre_nb_filters, kernel_size = self.pre_kernel_size, nb_stacks = self.pre_nb_conv, dilations = self.dilations,
                                              activation = 'relu', use_skip_connections = self.use_skip_connections, padding = self.padding,
                                              dropout_rate = self.dropout_rate, return_sequences = self.return_sequences,
                                              use_separable = self.use_separable)
        
            channels_out = []
            for chan in channels_in:
                channels_out.append(channel_model(chan))
        
            out = concatenate(channels_out)
        
            x = tcn_layer.TCN(nb_filters = self.nb_filters, kernel_size = self.kernel_size, nb_stacks = self.nb_conv, dilations = self.dilations,
                              activation = self.activation, use_skip_connections = self.use_skip_connections, padding = self.padding,
                              dropout_rate = self.dropout_rate, return_sequences = self.return_sequences, name = 'merge',
                              use_separable = self.use_separable)(out)
            
        if self.model_name == 'tcn_multi':            
            intermediate_encoder = Model(channels_in, x, name = 'TCN_Intermediate')
            
            # x = AveragePooling1D(pool_size = int(self.nb_hist // 128), strides = 16)(x) 
            x = MaxPooling1D(pool_size = int(self.nb_hist // 128), strides = 16)(x)
            
            output_layer = x
            feature_encoder = Model(channels_in, output_layer, name = 'TCN_FeatureEncoder')
        
        else:
            intermediate_encoder = Model(input_layer, x, name = 'TCN_Intermediate')
             
            # x = AveragePooling1D(pool_size = int(self.nb_hist // 128), strides = 16)(x) 
            x = MaxPooling1D(pool_size = int(self.nb_hist // 128), strides = 16)(x)
            
            output_layer = x
            feature_encoder = Model(input_layer, output_layer, name = 'TCN_FeatureEncoder')
            
        print(intermediate_encoder.summary())
        print(feature_encoder.summary())

        # Autoregressive params
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
        intermediate_features = intermediate_encoder(input_feats)
        
        # pooling_features = AveragePooling1D(pool_size = int(self.nb_hist // 128), strides = 16)(intermediate_features) 
        pooling_features = MaxPooling1D(pool_size = int(self.nb_hist // 128), strides = 16)(intermediate_features)
        
        encoder_output = dropout_layer(pooling_features)
        
        # Autoregressive model
        autoregressive_model = GRU(self.gru_units, return_sequences = True, name = 'autoregressive_layer')
        autoregressive_output = autoregressive_model(encoder_output)
        autoregressive_output = dropout_layer(autoregressive_output)

        # Contrastive loss
        contrastive_loss = ContrastiveLoss(self.gru_units, self.neg, self.steps)
        contrastive_loss_output = contrastive_loss([pooling_features, autoregressive_output])

        # Self-supervised Contrastive Predictive Coding Model
        self.model_training = Model(input_feats, contrastive_loss_output)       
        print(self.model_training.summary())
        
        return self.model_training, feature_encoder, intermediate_encoder
  
    def load_prediction_configuration(self, model, config, test_gen):
        """
        It instantiates the model architecture using the configuration parameters.
        :param model: a dictionary with the pretrained model.
        :param config: a dictionary with the configuration for predcition.
        :param test_gen: a dictionary with the test set features.
        :return: a model used for prediction.
        """
        
        super(CPCModel, self).load_prediction_configuration(model, config, test_gen)

        # Model architecture: Feature_Encoder -> Dropout -> GRU -> Dropout
       
        # Configuration
        self.model_name = config['model_name']
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
        
        self.dilations = config['dilations']
        self.activation = config['activation']
        self.use_skip_connections = config['use_skip_connections']
        self.dropout_rate = config['dropout']
        self.padding = config['padding']
        
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
        self.class_names = config['class_names'] 
        self.first_sample_train = config['first_sample_train'] 
        self.last_sample_train = config['last_sample_train']
        self.first_sample_val = config['first_sample_val']
        self.last_sample_val = config['last_sample_val']
        
        self.model_training = model
        self.test_gen = test_gen
                       
        if self.model_name == 'tcn_multi':
            """Create TCN network with TCN layer as pre-processing and downsampling frontend with weights shared between channels."""
            
            input_layer = Input(shape = (self.nb_hist, self.nb_freq))
            
            x = Dense(self.nb_classes)(model(input_layer))
            x = Activation('softmax')(x)
                        
            output_layer = x
            model = Model(input_layer, output_layer, name = 'TCN')
                                 
        else:
            """Create other TCN network."""
    
            input_layer = Input(shape = (self.nb_hist, self.nb_freq))
            
            x = Dense(self.nb_classes)(model(input_layer))
            x = Activation('softmax')(x)
            
            if self.nb_pre_conv > 0:
                x = UpSampling1D(size = 2 ** self.nb_pre_conv)(x)
            
            output_layer = x
            model = Model(input_layer, output_layer, name = 'TCN')

        self.model_prediction = model      
        print(self.model_prediction.summary())
        
        return self.model_prediction

    def train(self, config):
        """
        Train a self-supervised Contrastive Predictive Coding (CPC) model.
        :return: a trained model is saved on disk.
        """
        
        # Create folder structure to save the model
        os.makedirs(os.path.abspath(self.save_dir), exist_ok = True)
        
        self.save_name = '{0}/{1}{2}'.format(self.save_dir, self.save_prefix, time.strftime('%Y%m%d_%H%M%S'))
        self.checkpoint_save_name = self.save_name + '_model.h5'       
               
        config['save_name'] = self.save_name
        config['checkpoint_save_name'] = self.checkpoint_save_name
        
        utils.save_params(config, config['save_name'])
        
        self.model_training.compile(optimizer = Adam(lr = self.learning_rate, amsgrad = True, clipnorm = 1.0), loss = {'Contrastive_Loss': lambda y_true, y_pred: y_pred}, sample_weight_mode = self.sample_weight_mode)
          
        # Callbacks for training
        callbacks = [ModelCheckpoint(self.checkpoint_save_name, save_best_only = True, save_weights_only = False, monitor = 'val_loss', verbose = self.verbose), EarlyStopping(monitor = 'val_loss', patience = self.early_stop_epoch)]
        
        if self.reduce_lr:
            callbacks.append(ReduceLROnPlateau(patience = self.reduce_lr_patience, verbose = self.verbose))
        if self.tensorboard:
            callbacks.append(TensorBoard(log_dir = self.save_dir))
    
        # Train network
        logging.info('Start training')
        fit_hist = self.model_training.fit(self.data_gen,                                 
                                           steps_per_epoch = len(self.data_gen) // self.batch_size, # steps_per_epoch = min(len(self.data_gen), 1000),
                                           validation_data = self.val_gen,
                                           validation_steps = len(self.val_gen) // self.batch_size,
                                           epochs = self.nb_epoch,
                                           verbose = self.verbose,
                                           callbacks = callbacks)
        
        # save_filename = "{0}_fit_hist.h5".format(self.save_name)
        # logging.info('saving to ' + save_filename)
        # d = {'fit_hist': fit_hist}
    
        # fl.save(save_filename, d)

    def predict(self, config):        
        """
        Predict a self-supervised Contrastive Predictive Coding (CPC) model.
        :return: the results is saved on disk.
        """
        
        self.save_name = config['save_name']
        self.model_prediction.compile(optimizer = Adam(lr = self.learning_rate, amsgrad = True, clipnorm = 1.0), loss = "categorical_crossentropy", sample_weight_mode = self.sample_weight_mode)
        
        logging.info('Predicting')
        x_test, y_test, y_pred = evaluate.evaluate_probabilities(x = self.test_gen['x'], y = self.test_gen['y'], model = self.model_prediction, params = config)

        labels_test = predict.labels_from_probabilities(y_test)
        labels_pred = predict.labels_from_probabilities(y_pred)

        logging.info('Evaluating')
        conf_mat, report = evaluate.evaluate_segments(labels_test, labels_pred, config['class_names'])
        logging.info(conf_mat)
        logging.info(report)

        save_filename = "{0}_results.h5".format(self.save_name)
        logging.info('Saving to ' + save_filename + '.')
        d = {'confusion_matrix': conf_mat,
             'classification_report': report,
             'x_test': x_test,
             'y_test': y_test,
             'y_pred': y_pred,
             'labels_test': labels_test,
             'labels_pred': labels_pred}

        fl.save(save_filename, d)