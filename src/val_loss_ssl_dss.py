import matplotlib.pyplot as plt
import scipy.io

# parent_folder = 'parameter_search_flies_multichannels_steps=1'
# variable_tag_1 = 'flies'
# variable_tag_2 = 'multichannels'

# parent_folder = 'parameter_search_flies_singlechannel_steps=1'
# variable_tag_1 = 'flies'
# variable_tag_2 = 'singlechannel'

parent_folder = 'parameter_search_birds_steps=100'
variable_tag_1 = 'birds'
variable_tag_2 = 'singlechannel'

if variable_tag_1 == 'flies':
    steps = 10
else:
    steps = 100

for gru_units in [16, 32, 64, 128]:
    
    fig, ax = plt.subplots(1, 1, figsize = (1 * 8, 1 * 8))

    ax.set_title('Validation Loss >>> gru_units = ' + str(gru_units) + ', neg = 10, steps = ' + str(steps) + ' ms', fontsize = 10) 
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val Loss')
    
    color_tag = ['b', 'g', 'r', 'c']
    line_tag = ['-', '--', '+']
    legend_tag = ['fraction_data = 0.25, from scratch', 'fraction_data = 0.25, CPC + training', 'fraction_data = 0.25, CPC + freezing']
    
    counter = 0
    for fraction_ratio in [0.25, 0.5, 0.75, 1.0]:
        
        counter = counter + 1        
        for training_id in [1, 2, 3]:
            file_name_results = parent_folder + '/' + variable_tag_1 + '_' + str(gru_units) + '_' + str(fraction_ratio) + '_' + str(training_id) + '_results.mat'
            
            val_loss = scipy.io.loadmat(file_name_results)['val_loss']
            conf_mat = scipy.io.loadmat(file_name_results)['confusion_matrix']
            report_mat = scipy.io.loadmat(file_name_results)['classification_report']
            
            ax.plot(range(1, 11), val_loss.squeeze(0), line_tag[training_id - 1] + color_tag[counter - 1], label = legend_tag[training_id - 1])
            print(conf_mat)
            print(report_mat)

    ax.legend(loc = 'upper right', fontsize = 'medium')
    fig.savefig('Validation_Loss_' + variable_tag_1 + '_' + variable_tag_2 + '_' + str(gru_units) + '.png')