import matplotlib.pyplot as plt
import scipy.io

########## gru_units = 16 ##########

val_loss_16_1_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_16_0.25_1_results.mat')['val_loss']
conf_mat_16_1_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_16_0.25_1_results.mat')['confusion_matrix']
report_mat_16_1_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_16_0.25_1_results.mat')['classification_report']

val_loss_16_1_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_16_0.25_2_results.mat')['val_loss']
conf_mat_16_1_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_16_0.25_2_results.mat')['confusion_matrix']
report_mat_16_1_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_16_0.25_2_results.mat')['classification_report']

val_loss_16_1_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_16_0.25_3_results.mat')['val_loss']
conf_mat_16_1_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_16_0.25_3_results.mat')['confusion_matrix']
report_mat_16_1_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_16_0.25_3_results.mat')['classification_report']


val_loss_16_2_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_16_0.5_1_results.mat')['val_loss']
conf_mat_16_2_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_16_0.5_1_results.mat')['confusion_matrix']
report_mat_16_2_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_16_0.5_1_results.mat')['classification_report']

val_loss_16_2_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_16_0.5_2_results.mat')['val_loss']
conf_mat_16_2_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_16_0.5_2_results.mat')['confusion_matrix']
report_mat_16_2_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_16_0.5_2_results.mat')['classification_report']

val_loss_16_2_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_16_0.5_3_results.mat')['val_loss']
conf_mat_16_2_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_16_0.5_3_results.mat')['confusion_matrix']
report_mat_16_2_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_16_0.5_3_results.mat')['classification_report']


val_loss_16_3_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_16_0.75_1_results.mat')['val_loss']
conf_mat_16_3_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_16_0.75_1_results.mat')['confusion_matrix']
report_mat_16_3_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_16_0.75_1_results.mat')['classification_report']

val_loss_16_3_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_16_0.75_2_results.mat')['val_loss']
conf_mat_16_3_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_16_0.75_2_results.mat')['confusion_matrix']
report_mat_16_3_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_16_0.75_2_results.mat')['classification_report']

val_loss_16_3_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_16_0.75_3_results.mat')['val_loss']
conf_mat_16_3_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_16_0.75_3_results.mat')['confusion_matrix']
report_mat_16_3_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_16_0.75_3_results.mat')['classification_report']


val_loss_16_4_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_16_1.0_1_results.mat')['val_loss']
conf_mat_16_4_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_16_1.0_1_results.mat')['confusion_matrix']
report_mat_16_4_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_16_1.0_1_results.mat')['classification_report']

val_loss_16_4_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_16_1.0_2_results.mat')['val_loss']
conf_mat_16_4_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_16_1.0_2_results.mat')['confusion_matrix']
report_mat_16_4_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_16_1.0_2_results.mat')['classification_report']

val_loss_16_4_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_16_1.0_3_results.mat')['val_loss']
conf_mat_16_4_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_16_1.0_3_results.mat')['confusion_matrix']
report_mat_16_4_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_16_1.0_3_results.mat')['classification_report']

fig, ax = plt.subplots(1, 1, figsize = (1 * 8, 1 * 8))

ax.set_title('Validation Loss >>> gru_units = 16, neg = 10, steps = 10 ms', fontsize = 10) 
ax.set_xlabel('Epoch')
ax.set_ylabel('Val Loss')
            
ax.plot(range(1, 11), val_loss_16_1_1.squeeze(0), "-b", label = "fraction_data = 0.25, from scratch")
ax.plot(range(1, 11), val_loss_16_1_2.squeeze(0), "--b", label = "fraction_data = 0.25, CPC + training")
ax.plot(range(1, 11), val_loss_16_1_3.squeeze(0), "+b", label = "fraction_data = 0.25, CPC + freezing")

ax.plot(range(1, 11), val_loss_16_2_1.squeeze(0), "-g", label = "fraction_data = 0.5, from scratch")
ax.plot(range(1, 11), val_loss_16_2_2.squeeze(0), "--g", label = "fraction_data = 0.5, CPC + training")
ax.plot(range(1, 11), val_loss_16_2_3.squeeze(0), "+g", label = "fraction_data = 0.5, CPC + freezing")

ax.plot(range(1, 11), val_loss_16_3_1.squeeze(0), "-r", label = "fraction_data = 0.75, from scratch")
ax.plot(range(1, 11), val_loss_16_3_2.squeeze(0), "--r", label = "fraction_data = 0.75, CPC + training")
ax.plot(range(1, 11), val_loss_16_3_3.squeeze(0), "+r", label = "fraction_data = 0.75, CPC + freezing") 

ax.plot(range(1, 11), val_loss_16_4_1.squeeze(0), "-c", label = "fraction_data = 1.0, from scratch")
ax.plot(range(1, 11), val_loss_16_4_2.squeeze(0), "--c", label = "fraction_data = 1.0, CPC + training")
ax.plot(range(1, 11), val_loss_16_4_3.squeeze(0), "+c", label = "fraction_data = 1.0, CPC + freezing")

ax.legend(loc = 'upper right', fontsize = 'medium')

fig.savefig('Validation Loss_Flies_Single Channel_16.png')
       
########## gru_units = 32 ##########

val_loss_32_1_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_32_0.25_1_results.mat')['val_loss']
conf_mat_32_1_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_32_0.25_1_results.mat')['confusion_matrix']
report_mat_32_1_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_32_0.25_1_results.mat')['classification_report']

val_loss_32_1_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_32_0.25_2_results.mat')['val_loss']
conf_mat_32_1_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_32_0.25_2_results.mat')['confusion_matrix']
report_mat_32_1_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_32_0.25_2_results.mat')['classification_report']

val_loss_32_1_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_32_0.25_3_results.mat')['val_loss']
conf_mat_32_1_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_32_0.25_3_results.mat')['confusion_matrix']
report_mat_32_1_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_32_0.25_3_results.mat')['classification_report']


val_loss_32_2_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_32_0.5_1_results.mat')['val_loss']
conf_mat_32_2_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_32_0.5_1_results.mat')['confusion_matrix']
report_mat_32_2_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_32_0.5_1_results.mat')['classification_report']

val_loss_32_2_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_32_0.5_2_results.mat')['val_loss']
conf_mat_32_2_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_32_0.5_2_results.mat')['confusion_matrix']
report_mat_32_2_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_32_0.5_2_results.mat')['classification_report']

val_loss_32_2_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_32_0.5_3_results.mat')['val_loss']
conf_mat_32_2_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_32_0.5_3_results.mat')['confusion_matrix']
report_mat_32_2_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_32_0.5_3_results.mat')['classification_report']


val_loss_32_3_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_32_0.75_1_results.mat')['val_loss']
conf_mat_32_3_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_32_0.75_1_results.mat')['confusion_matrix']
report_mat_32_3_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_32_0.75_1_results.mat')['classification_report']

val_loss_32_3_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_32_0.75_2_results.mat')['val_loss']
conf_mat_32_3_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_32_0.75_2_results.mat')['confusion_matrix']
report_mat_32_3_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_32_0.75_2_results.mat')['classification_report']

val_loss_32_3_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_32_0.75_3_results.mat')['val_loss']
conf_mat_32_3_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_32_0.75_3_results.mat')['confusion_matrix']
report_mat_32_3_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_32_0.75_3_results.mat')['classification_report']


val_loss_32_4_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_32_1.0_1_results.mat')['val_loss']
conf_mat_32_4_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_32_1.0_1_results.mat')['confusion_matrix']
report_mat_32_4_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_32_1.0_1_results.mat')['classification_report']

val_loss_32_4_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_32_1.0_2_results.mat')['val_loss']
conf_mat_32_4_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_32_1.0_2_results.mat')['confusion_matrix']
report_mat_32_4_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_32_1.0_2_results.mat')['classification_report']

val_loss_32_4_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_32_1.0_3_results.mat')['val_loss']
conf_mat_32_4_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_32_1.0_3_results.mat')['confusion_matrix']
report_mat_32_4_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_32_1.0_3_results.mat')['classification_report']

fig, ax = plt.subplots(1, 1, figsize = (1 * 8, 1 * 8))

ax.set_title('Validation Loss >>> gru_units = 32, neg = 10, steps = 10 ms', fontsize = 10) 
ax.set_xlabel('Epoch')
ax.set_ylabel('Val Loss')
            
ax.plot(range(1, 11), val_loss_32_1_1.squeeze(0), "-b", label = "fraction_data = 0.25, from scratch")
ax.plot(range(1, 11), val_loss_32_1_2.squeeze(0), "--b", label = "fraction_data = 0.25, CPC + training")
ax.plot(range(1, 11), val_loss_32_1_3.squeeze(0), "+b", label = "fraction_data = 0.25, CPC + freezing")

ax.plot(range(1, 11), val_loss_32_2_1.squeeze(0), "-g", label = "fraction_data = 0.5, from scratch")
ax.plot(range(1, 11), val_loss_32_2_2.squeeze(0), "--g", label = "fraction_data = 0.5, CPC + training")
ax.plot(range(1, 11), val_loss_32_2_3.squeeze(0), "+g", label = "fraction_data = 0.5, CPC + freezing")

ax.plot(range(1, 11), val_loss_32_3_1.squeeze(0), "-r", label = "fraction_data = 0.75, from scratch")
ax.plot(range(1, 11), val_loss_32_3_2.squeeze(0), "--r", label = "fraction_data = 0.75, CPC + training")
ax.plot(range(1, 11), val_loss_32_3_3.squeeze(0), "+r", label = "fraction_data = 0.75, CPC + freezing") 

ax.plot(range(1, 11), val_loss_32_4_1.squeeze(0), "-c", label = "fraction_data = 1.0, from scratch")
ax.plot(range(1, 11), val_loss_32_4_2.squeeze(0), "--c", label = "fraction_data = 1.0, CPC + training")
ax.plot(range(1, 11), val_loss_32_4_3.squeeze(0), "+c", label = "fraction_data = 1.0, CPC + freezing")

ax.legend(loc = 'upper right', fontsize = 'medium')

fig.savefig('Validation Loss_Flies_Single Channel_32.png')

########## gru_units = 64 ##########

val_loss_64_1_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_64_0.25_1_results.mat')['val_loss']
conf_mat_64_1_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_64_0.25_1_results.mat')['confusion_matrix']
report_mat_64_1_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_64_0.25_1_results.mat')['classification_report']

val_loss_64_1_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_64_0.25_2_results.mat')['val_loss']
conf_mat_64_1_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_64_0.25_2_results.mat')['confusion_matrix']
report_mat_64_1_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_64_0.25_2_results.mat')['classification_report']

val_loss_64_1_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_64_0.25_3_results.mat')['val_loss']
conf_mat_64_1_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_64_0.25_3_results.mat')['confusion_matrix']
report_mat_64_1_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_64_0.25_3_results.mat')['classification_report']


val_loss_64_2_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_64_0.5_1_results.mat')['val_loss']
conf_mat_64_2_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_64_0.5_1_results.mat')['confusion_matrix']
report_mat_64_2_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_64_0.5_1_results.mat')['classification_report']

val_loss_64_2_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_64_0.5_2_results.mat')['val_loss']
conf_mat_64_2_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_64_0.5_2_results.mat')['confusion_matrix']
report_mat_64_2_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_64_0.5_2_results.mat')['classification_report']

val_loss_64_2_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_64_0.5_3_results.mat')['val_loss']
conf_mat_64_2_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_64_0.5_3_results.mat')['confusion_matrix']
report_mat_64_2_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_64_0.5_3_results.mat')['classification_report']


val_loss_64_3_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_64_0.75_1_results.mat')['val_loss']
conf_mat_64_3_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_64_0.75_1_results.mat')['confusion_matrix']
report_mat_64_3_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_64_0.75_1_results.mat')['classification_report']

val_loss_64_3_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_64_0.75_2_results.mat')['val_loss']
conf_mat_64_3_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_64_0.75_2_results.mat')['confusion_matrix']
report_mat_64_3_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_64_0.75_2_results.mat')['classification_report']

val_loss_64_3_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_64_0.75_3_results.mat')['val_loss']
conf_mat_64_3_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_64_0.75_3_results.mat')['confusion_matrix']
report_mat_64_3_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_64_0.75_3_results.mat')['classification_report']


val_loss_64_4_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_64_1.0_1_results.mat')['val_loss']
conf_mat_64_4_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_64_1.0_1_results.mat')['confusion_matrix']
report_mat_64_4_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_64_1.0_1_results.mat')['classification_report']

val_loss_64_4_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_64_1.0_2_results.mat')['val_loss']
conf_mat_64_4_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_64_1.0_2_results.mat')['confusion_matrix']
report_mat_64_4_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_64_1.0_2_results.mat')['classification_report']

val_loss_64_4_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_64_1.0_3_results.mat')['val_loss']
conf_mat_64_4_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_64_1.0_3_results.mat')['confusion_matrix']
report_mat_64_4_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_64_1.0_3_results.mat')['classification_report']

fig, ax = plt.subplots(1, 1, figsize = (1 * 8, 1 * 8))

ax.set_title('Validation Loss >>> gru_units = 64, neg = 10, steps = 10 ms', fontsize = 10) 
ax.set_xlabel('Epoch')
ax.set_ylabel('Val Loss')
            
ax.plot(range(1, 11), val_loss_64_1_1.squeeze(0), "-b", label = "fraction_data = 0.25, from scratch")
ax.plot(range(1, 11), val_loss_64_1_2.squeeze(0), "--b", label = "fraction_data = 0.25, CPC + training")
ax.plot(range(1, 11), val_loss_64_1_3.squeeze(0), "+b", label = "fraction_data = 0.25, CPC + freezing")

ax.plot(range(1, 11), val_loss_64_2_1.squeeze(0), "-g", label = "fraction_data = 0.5, from scratch")
ax.plot(range(1, 11), val_loss_64_2_2.squeeze(0), "--g", label = "fraction_data = 0.5, CPC + training")
ax.plot(range(1, 11), val_loss_64_2_3.squeeze(0), "+g", label = "fraction_data = 0.5, CPC + freezing")

ax.plot(range(1, 11), val_loss_64_3_1.squeeze(0), "-r", label = "fraction_data = 0.75, from scratch")
ax.plot(range(1, 11), val_loss_64_3_2.squeeze(0), "--r", label = "fraction_data = 0.75, CPC + training")
ax.plot(range(1, 11), val_loss_64_3_3.squeeze(0), "+r", label = "fraction_data = 0.75, CPC + freezing") 

ax.plot(range(1, 11), val_loss_64_4_1.squeeze(0), "-c", label = "fraction_data = 1.0, from scratch")
ax.plot(range(1, 11), val_loss_64_4_2.squeeze(0), "--c", label = "fraction_data = 1.0, CPC + training")
ax.plot(range(1, 11), val_loss_64_4_3.squeeze(0), "+c", label = "fraction_data = 1.0, CPC + freezing")

ax.legend(loc = 'upper right', fontsize = 'medium')

fig.savefig('Validation Loss_Flies_Single Channel_64.png')

########## gru_units = 128 ##########

val_loss_128_1_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_128_0.25_1_results.mat')['val_loss']
conf_mat_128_1_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_128_0.25_1_results.mat')['confusion_matrix']
report_mat_128_1_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_128_0.25_1_results.mat')['classification_report']

val_loss_128_1_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_128_0.25_2_results.mat')['val_loss']
conf_mat_128_1_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_128_0.25_2_results.mat')['confusion_matrix']
report_mat_128_1_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_128_0.25_2_results.mat')['classification_report']

val_loss_128_1_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_128_0.25_3_results.mat')['val_loss']
conf_mat_128_1_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_128_0.25_3_results.mat')['confusion_matrix']
report_mat_128_1_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_128_0.25_3_results.mat')['classification_report']


val_loss_128_2_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_128_0.5_1_results.mat')['val_loss']
conf_mat_128_2_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_128_0.5_1_results.mat')['confusion_matrix']
report_mat_128_2_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_128_0.5_1_results.mat')['classification_report']

val_loss_128_2_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_128_0.5_2_results.mat')['val_loss']
conf_mat_128_2_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_128_0.5_2_results.mat')['confusion_matrix']
report_mat_128_2_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_128_0.5_2_results.mat')['classification_report']

val_loss_128_2_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_128_0.5_3_results.mat')['val_loss']
conf_mat_128_2_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_128_0.5_3_results.mat')['confusion_matrix']
report_mat_128_2_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_128_0.5_3_results.mat')['classification_report']


val_loss_128_3_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_128_0.75_1_results.mat')['val_loss']
conf_mat_128_3_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_128_0.75_1_results.mat')['confusion_matrix']
report_mat_128_3_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_128_0.75_1_results.mat')['classification_report']

val_loss_128_3_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_128_0.75_2_results.mat')['val_loss']
conf_mat_128_3_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_128_0.75_2_results.mat')['confusion_matrix']
report_mat_128_3_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_128_0.75_2_results.mat')['classification_report']

val_loss_128_3_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_128_0.75_3_results.mat')['val_loss']
conf_mat_128_3_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_128_0.75_3_results.mat')['confusion_matrix']
report_mat_128_3_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_128_0.75_3_results.mat')['classification_report']


val_loss_128_4_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_128_1.0_1_results.mat')['val_loss']
conf_mat_128_4_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_128_1.0_1_results.mat')['confusion_matrix']
report_mat_128_4_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_128_1.0_1_results.mat')['classification_report']

val_loss_128_4_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_128_1.0_2_results.mat')['val_loss']
conf_mat_128_4_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_128_1.0_2_results.mat')['confusion_matrix']
report_mat_128_4_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_128_1.0_2_results.mat')['classification_report']

val_loss_128_4_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_128_1.0_3_results.mat')['val_loss']
conf_mat_128_4_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_128_1.0_3_results.mat')['confusion_matrix']
report_mat_128_4_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_128_1.0_3_results.mat')['classification_report']

fig, ax = plt.subplots(1, 1, figsize = (1 * 8, 1 * 8))

ax.set_title('Validation Loss >>> gru_units = 128, neg = 10, steps = 10 ms', fontsize = 10) 
ax.set_xlabel('Epoch')
ax.set_ylabel('Val Loss')
            
ax.plot(range(1, 11), val_loss_128_1_1.squeeze(0), "-b", label = "fraction_data = 0.25, from scratch")
ax.plot(range(1, 11), val_loss_128_1_2.squeeze(0), "--b", label = "fraction_data = 0.25, CPC + training")
ax.plot(range(1, 11), val_loss_128_1_3.squeeze(0), "+b", label = "fraction_data = 0.25, CPC + freezing")

ax.plot(range(1, 11), val_loss_128_2_1.squeeze(0), "-g", label = "fraction_data = 0.5, from scratch")
ax.plot(range(1, 11), val_loss_128_2_2.squeeze(0), "--g", label = "fraction_data = 0.5, CPC + training")
ax.plot(range(1, 11), val_loss_128_2_3.squeeze(0), "+g", label = "fraction_data = 0.5, CPC + freezing")

ax.plot(range(1, 11), val_loss_128_3_1.squeeze(0), "-r", label = "fraction_data = 0.75, from scratch")
ax.plot(range(1, 11), val_loss_128_3_2.squeeze(0), "--r", label = "fraction_data = 0.75, CPC + training")
ax.plot(range(1, 11), val_loss_128_3_3.squeeze(0), "+r", label = "fraction_data = 0.75, CPC + freezing") 

ax.plot(range(1, 11), val_loss_128_4_1.squeeze(0), "-c", label = "fraction_data = 1.0, from scratch")
ax.plot(range(1, 11), val_loss_128_4_2.squeeze(0), "--c", label = "fraction_data = 1.0, CPC + training")
ax.plot(range(1, 11), val_loss_128_4_3.squeeze(0), "+c", label = "fraction_data = 1.0, CPC + freezing") 

ax.legend(loc = 'upper right', fontsize = 'medium')

fig.savefig('Validation Loss_Flies_Single Channel_128.png')

# ########## gru_units = 256 ##########

# val_loss_256_1_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_256_0.25_1_results.mat')['val_loss']
# conf_mat_256_1_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_256_0.25_1_results.mat')['confusion_matrix']
# report_mat_256_1_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_256_0.25_1_results.mat')['classification_report']

# val_loss_256_1_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_256_0.25_2_results.mat')['val_loss']
# conf_mat_256_1_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_256_0.25_2_results.mat')['confusion_matrix']
# report_mat_256_1_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_256_0.25_2_results.mat')['classification_report']

# val_loss_256_1_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_256_0.25_3_results.mat')['val_loss']
# conf_mat_256_1_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_256_0.25_3_results.mat')['confusion_matrix']
# report_mat_256_1_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_256_0.25_3_results.mat')['classification_report']


# val_loss_256_2_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_256_0.5_1_results.mat')['val_loss']
# conf_mat_256_2_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_256_0.5_1_results.mat')['confusion_matrix']
# report_mat_256_2_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_256_0.5_1_results.mat')['classification_report']

# val_loss_256_2_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_256_0.5_2_results.mat')['val_loss']
# conf_mat_256_2_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_256_0.5_2_results.mat')['confusion_matrix']
# report_mat_256_2_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_256_0.5_2_results.mat')['classification_report']

# val_loss_256_2_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_256_0.5_3_results.mat')['val_loss']
# conf_mat_256_2_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_256_0.5_3_results.mat')['confusion_matrix']
# report_mat_256_2_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_256_0.5_3_results.mat')['classification_report']


# val_loss_256_3_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_256_0.75_1_results.mat')['val_loss']
# conf_mat_256_3_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_256_0.75_1_results.mat')['confusion_matrix']
# report_mat_256_3_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_256_0.75_1_results.mat')['classification_report']

# val_loss_256_3_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_256_0.75_2_results.mat')['val_loss']
# conf_mat_256_3_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_256_0.75_2_results.mat')['confusion_matrix']
# report_mat_256_3_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_256_0.75_2_results.mat')['classification_report']

# val_loss_256_3_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_256_0.75_3_results.mat')['val_loss']
# conf_mat_256_3_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_256_0.75_3_results.mat')['confusion_matrix']
# report_mat_256_3_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_256_0.75_3_results.mat')['classification_report']


# val_loss_256_4_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_256_1.0_1_results.mat')['val_loss']
# conf_mat_256_4_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_256_1.0_1_results.mat')['confusion_matrix']
# report_mat_256_4_1 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_256_1.0_1_results.mat')['classification_report']

# val_loss_256_4_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_256_1.0_2_results.mat')['val_loss']
# conf_mat_256_4_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_256_1.0_2_results.mat')['confusion_matrix']
# report_mat_256_4_2 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_256_1.0_2_results.mat')['classification_report']

# val_loss_256_4_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_256_1.0_3_results.mat')['val_loss']
# conf_mat_256_4_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_256_1.0_3_results.mat')['confusion_matrix']
# report_mat_256_4_3 = scipy.io.loadmat('parameter_search_flies_singlechannel_steps=1/flies_256_1.0_3_results.mat')['classification_report']

# fig, ax = plt.subplots(1, 1, figsize = (1 * 8, 1 * 8))

# ax.set_title('Validation Loss >>> gru_units = 256, neg = 10, steps = 10 ms', fontsize = 10) 
# ax.set_xlabel('Epoch')
# ax.set_ylabel('Val Loss')
            
# ax.plot(range(1, 11), val_loss_256_1_1.squeeze(0), "-b", label = "fraction_data = 0.25, from scratch")
# ax.plot(range(1, 11), val_loss_256_1_2.squeeze(0), "--b", label = "fraction_data = 0.25, CPC + training")
# ax.plot(range(1, 11), val_loss_256_1_3.squeeze(0), "+b", label = "fraction_data = 0.25, CPC + freezing")

# ax.plot(range(1, 11), val_loss_256_2_1.squeeze(0), "-g", label = "fraction_data = 0.5, from scratch")
# ax.plot(range(1, 11), val_loss_256_2_2.squeeze(0), "--g", label = "fraction_data = 0.5, CPC + training")
# ax.plot(range(1, 11), val_loss_256_2_3.squeeze(0), "+g", label = "fraction_data = 0.5, CPC + freezing")

# ax.plot(range(1, 11), val_loss_256_3_1.squeeze(0), "-r", label = "fraction_data = 0.75, from scratch")
# ax.plot(range(1, 11), val_loss_256_3_2.squeeze(0), "--r", label = "fraction_data = 0.75, CPC + training")
# ax.plot(range(1, 11), val_loss_256_3_3.squeeze(0), "+r", label = "fraction_data = 0.75, CPC + freezing") 

# ax.plot(range(1, 11), val_loss_256_4_1.squeeze(0), "-c", label = "fraction_data = 1.0, from scratch")
# ax.plot(range(1, 11), val_loss_256_4_2.squeeze(0), "--c", label = "fraction_data = 1.0, CPC + training")
# ax.plot(range(1, 11), val_loss_256_4_3.squeeze(0), "+c", label = "fraction_data = 1.0, CPC + freezing")

# ax.legend(loc = 'upper right', fontsize = 'medium')

# fig.savefig('Validation Loss_Flies_Single Channel_256.png')