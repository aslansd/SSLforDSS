from train_ssl_dss import train_ssl_dss

all_gru_units = [32, 64, 128, 256]
all_steps = [5, 10]

for i in all_gru_units:
    for j in all_steps:
        
        train_ssl_dss(data_dir = 'dat/dmel_single_raw.npy', y_suffix = '', # dat/dmel_single_raw.npy # dat/dmel_single_stern_raw_manualp.npy
                      save_dir = 'parameter_search_new_3', save_prefix = str(i) + '_' + str(j),
                      model_name = 'tcn', nb_filters = 16, kernel_size = 16,
                      nb_conv = 3, use_separable = False, nb_hist = (j + 1) * 100,
                      ignore_boundaries = False, batch_norm = True,
                      nb_pre_conv = 0,
                      pre_kernel_size = 3, pre_nb_filters = 16, pre_nb_conv = 2,
                      verbose = 1, batch_size = 32,
                      nb_epoch = 200,
                      learning_rate = 0.0001, reduce_lr = False, reduce_lr_patience = 5,
                      fraction_data = None, seed = None, batch_level_subsampling = False,
                      tensorboard = False, log_messages = False,
                      nb_stacks = 2, with_y_hist = True, x_suffix = '',
                      dilations = [1, 2, 4, 8, 16], activation = 'norm_relu', use_skip_connections = True, dropout_rate = 0.00, padding = 'same',
                      early_stop_epoch = 20, gru_units = i, dropout = 0.2, neg = 10, steps = j)