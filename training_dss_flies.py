"""
Created on Wed Jan 27 01:41:02 2021

@author: Aslan
"""

from train import train
import logging

logging.basicConfig(level = logging.INFO)

train(data_dir = 'dat/dmel_multi_murthy_raw_manual_sine.npy', 
      y_suffix = '',
      save_dir = 'res',
      save_prefix = 'flies',
      model_name = 'tcn',
      nb_filters = 32, 
      kernel_size = 32,
      nb_conv = 3, 
      use_separable = False, 
      nb_hist = 2048,
      ignore_boundaries = False, 
      batch_norm = True,
      nb_pre_conv = 0,
      pre_kernel_size = 3, 
      pre_nb_filters = 16, 
      pre_nb_conv = 2,
      verbose = 1, 
      batch_size = 32,
      nb_epoch = 10,
      learning_rate = 0.0001, 
      reduce_lr = False, 
      reduce_lr_patience = 5,
      fraction_data = None, 
      seed = 1, 
      batch_level_subsampling = False,
      tensorboard = False, 
      log_messages = False,
      nb_stacks = 2, 
      with_y_hist = True, 
      x_suffix = '')