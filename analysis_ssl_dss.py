import librosa.display
import librosa.feature
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn

import evaluate, event_utils, predict, pulse_utils, utils, utils_plot

# import pdb

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable = True)

name_first_part = ['32_10_20210612', '32_10_20210612', '32_10_20210613', '32_10_20210612', '32_10_20210612']
name_second_part = ['102112', '114108', '130350', '191123', '010902', '152219', '102112', '114108', '102112', '114108']

for data_idx in [1, 2, 3]:
    for train_flag in [1, 2]:
        
        if data_idx == 1:
            if train_flag == 1:
                save_name = 'parameter_search/' + name_first_part[0] + '_' + name_second_part[0] + '_' + str(data_idx) + '_' + str(train_flag)
            elif train_flag == 2:
                save_name = 'parameter_search/' + name_first_part[0] + '_' + name_second_part[1] + '_' + str(data_idx) + '_' + str(train_flag)
        elif data_idx == 2:
            if train_flag == 1:
                save_name = 'parameter_search/' + name_first_part[1] + '_' + name_second_part[2] + '_' + str(data_idx) + '_' + str(train_flag)
            elif train_flag == 2:
                save_name = 'parameter_search/' + name_first_part[1] + '_' + name_second_part[3] + '_' + str(data_idx) + '_' + str(train_flag)
        elif data_idx == 3:
            if train_flag == 1:
                save_name = 'parameter_search/' + name_first_part[2] + '_' + name_second_part[4] + '_' + str(data_idx) + '_' + str(train_flag)
            elif train_flag == 2:
                save_name = 'parameter_search/' + name_first_part[2] + '_' + name_second_part[5] + '_' + str(data_idx) + '_' + str(train_flag)
        elif data_idx == 4:
            if train_flag == 1:
                save_name = 'parameter_search/' + name_first_part[3] + '_' + name_second_part[6] + '_' + str(data_idx) + '_' + str(train_flag)
            elif train_flag == 2:
                save_name = 'parameter_search/' + name_first_part[3] + '_' + name_second_part[7] + '_' + str(data_idx) + '_' + str(train_flag)
        elif data_idx == 5:
            if train_flag == 1:
                save_name = 'parameter_search/' + name_first_part[4] + '_' + name_second_part[8] + '_' + str(data_idx) + '_' + str(train_flag)
            elif train_flag == 2:
                save_name = 'parameter_search/' + name_first_part[4] + '_' + name_second_part[9] + '_' + str(data_idx) + '_' + str(train_flag)
                
        print('**********')
        print(save_name)
        print('**********')
            
        params = utils.load_params(save_name)
        print(params)
        
        dwn_smp = 1
        
        if 'cnn' in params['model_name']:
            y_offset = params['y_offset']
        
        else:
            data_padding = 0
    
            if params['with_y_hist'] and params['ignore_boundaries']:
                data_padding = int(np.ceil(params['kernel_size'] * params['nb_conv']))

            y_offset = data_padding
        
        fs = params['samplerate_y_Hz']
               
        try: 
            segment_pred_index = params['class_types'][1:].index('segment') + 1
            print(f'model predicts segments at index {segment_pred_index}')
        except ValueError:
            print('model does not predict segments.')
            segment_pred_index = None
        
        try: 
            pulse_pred_index = params['class_types'].index('event')    
            print(f'model predicts pulses at index {pulse_pred_index}')
        
        except ValueError:
            print('model does not predict pulse.')
            pulse_pred_index = None
        
        print(fs, y_offset)
        
        datasets = utils.load_from(save_name + '_results.h5', ['x_test', 'y_test', 'y_pred'])
        x_test, y_test, y_pred = [datasets[key] for key in ['x_test', 'y_test', 'y_pred']]  # Unpack dict items to vars
        
        labels_test = predict.labels_from_probabilities(y_test)
        labels_pred = predict.labels_from_probabilities(y_pred)
        
        def prc_pulse(pred_pulse, pulsetimes_true, fs, tol, min_dist, dwn_smp = 1, y_offset = 0, thresholds = None):
            if thresholds is None:
                thresholds = np.arange(0, 1.01, 0.01)
            
            precision = []
            recall = []
            f1_score = []
            threshold = []
        
            for thres in thresholds:
                pulsetimes_pred, pulsetimes_pred_confidence = event_utils.detect_events(pred_pulse, thres = thres, min_dist = min_dist)
                pulsetimes_pred = ((pulsetimes_pred * dwn_smp) + y_offset) / fs
                d, nn_pred_pulse, nn_true_pulse, nn_dist = event_utils.evaluate_eventtimes(pulsetimes_true, pulsetimes_pred, fs, tol)
                precision.append(d['precision'])
                recall.append(d['recall'])
                f1_score.append(d['f1_score'])
                threshold.append(thres)
            
            return threshold, precision, recall, f1_score
        
        # Evaluate events based on timing, allowing for some tolerance
        tol = 0.01  # seconds = 10 ms
        min_dist = 0.01
            
        if pulse_pred_index is not None:
            pulsetimes_true, _ = event_utils.detect_events(y_test[:, 1], thres = 0.5, min_dist = 100)
            pulsetimes_true = pulsetimes_true / fs
            
            # Calculate f1 score for different thresholds and choose opt threshold
            threshold, precision, recall, f1_score = prc_pulse(y_pred[:, pulse_pred_index], pulsetimes_true, fs, tol, min_dist * fs)
            threshold_opt = threshold[np.argmax(f1_score)]
            
            # Predict pulses using optimal threshold
            pulsetimes_pred, pulsetimes_pred_confidence = event_utils.detect_events(y_pred[:, pulse_pred_index], thres = threshold_opt, min_dist = min_dist * fs)
            pulsetimes_pred = pulsetimes_pred / fs
            d, nn_pred_pulse, nn_true_pulse, nn_dist = event_utils.evaluate_eventtimes(pulsetimes_true, pulsetimes_pred, fs, tol)
            
            print(f"FP {d['FP']}, TP {d['TP']}, FN {d['FN']}")
            print(f"precision {d['precision']:1.2f}, recall {d['recall']:1.2f}, f1-score {d['f1_score']:1.2f}")
            
            plt.gcf().set_size_inches(15, 6)
            
            plt.subplot(231)
            plt.plot(precision, recall, c = 'k')
            plt.scatter(precision, recall, c = threshold)
            plt.xlabel('Precision')
            plt.ylabel('Recall')
            plt.axis('square')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            
            plt.subplot(234)
            plt.plot(threshold, f1_score, c = 'k')
            plt.scatter(threshold, f1_score, c = threshold)
            plt.axvline(threshold_opt, c = 'k')
            plt.xlabel('Threshold')
            plt.ylabel('F1 score')
            plt.axis('square')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            print(f'best threshold (max f1 score): {threshold_opt:1.2f}')
            
            plt.subplot(132)
            plt.plot(pulsetimes_true / 60, '.', alpha = 0.5)
            plt.plot(pulsetimes_pred / 60, '.', alpha = 0.5)
            plt.legend(['True pulses', 'Predicted pulses'])
            plt.xlabel('Pulse number')
            plt.ylabel('Time [minutes]')
            plt.ylim(0, x_test.shape[0] / fs / 60 * 1.01 * dwn_smp)
            utils_plot.remove_axes()
        
            plt.subplot(233)
            plt.plot(nn_dist, '.-', markersize = 10)
            plt.xlim(0, len(nn_dist))
            plt.axhline(tol, color = 'k')
            plt.yscale('log')
            plt.ylabel('Distance to pulse [s]')
            utils_plot.remove_axes()
        
            plt.subplot(236)
            plt.hist(nn_dist, bins = np.arange(0, 4 * tol, .001), density = True)
            plt.axvline(tol, color = 'k')
            plt.xlabel('Distance to pulse [s]')
            plt.ylabel('Probability');
            utils_plot.remove_axes()
            
            plt.savefig('Evaluation of events based on timing allowing for some tolerance' + '_' + str(data_idx) + '_' + str(train_flag) + '.png')
            
        # Plot the raw predictions
        for t0 in [160_000, int(1_270_000), int(1_430_000)]:
            t1 = int(t0 + 100_000)
            t0d = int(t0 / dwn_smp)
            t1d = int(t1 / dwn_smp)
            
            plt.gcf().set_size_inches(40, 5)
            
            plt.subplot(311)
            plt.plot(x_test[y_offset + t0:y_offset + t1], c = 'k', linewidth = 0.5)
            plt.xlim(0, (t1 - t0))
            plt.ylim(-0.2, 0.2)
            plt.xticks([])
            utils_plot.remove_axes()
        
            plt.subplot(613)
            plt.imshow(y_test[t0d:t1d].T, cmap = 'Blues')
            plt.yticks((0, 1), labels = params['class_names'])
            plt.ylabel('FSS')
            plt.xticks([])
            utils_plot.remove_axes()
        
            plt.subplot(614)
            plt.imshow(y_pred[t0d:t1d].T, cmap = 'Oranges')
            plt.yticks((0, 1), labels = params['class_names'])
            plt.ylabel('TCN')
            plt.xticks([])
            utils_plot.remove_axes()
            
            plt.subplot(615)
            plt.plot(labels_test[t0d:t1d])
            plt.plot(labels_pred[t0d:t1d], '.', alpha = 0.1)
            plt.yticks((0, 1), labels = params['class_names'])
            
            plt.legend(('true', 'TCN'))
            utils_plot.remove_axes()
            
        plt.savefig('The raw predictions' + '_' + str(data_idx) + '_' + str(train_flag) + '.png')
            
        # Plot the shapes of false/true positive/negative pulses - if you use the network trained on the toy data set, most false positives arise from wrong annotations
        def plot_pulses(pulseshapes, col = 1, title = ''):
            win_hw = pulseshapes.shape[0] / 2
            
            plt.subplot(2, 3, col)
            plt.axvline(win_hw, color = 'k')
            plt.axhline(0, color = 'k')
            plt.plot(pulseshapes, linewidth = 0.75, alpha = 0.2)
            plt.ylim(-0.5, 0.5)
            plt.title(title)
            utils_plot.scalebar(2, units = 'ms', dx = 0.1)
            utils_plot.remove_axes(all = True)
            
            plt.subplot(2, 3, col + 3)
            plt.imshow(pulseshapes.T, cmap = 'RdBu_r')
            plt.clim(-0.5, 0.5)
            plt.axvline(win_hw, color = 'k')
            utils_plot.scalebar(2, units = 'ms', dx = 0.1)
            utils_plot.remove_axes(all = True)
        
        win_hw = 100
        
        if pulse_pred_index is not None:
            pulseshapes_pred = pulse_utils.get_pulseshapes(pulsetimes_pred * fs + win_hw, x_test, win_hw)
            pulsenorm_pred = np.linalg.norm(np.abs(pulseshapes_pred[50:-50, :]), axis = 0)
            pulsefreq_pred = np.array([pulse_utils.pulse_freq(p)[0] for p in pulseshapes_pred[50:-50, :].T])
            pulseshapes_pred = np.apply_along_axis(pulse_utils.normalize_pulse, axis = -1, arr = pulseshapes_pred.T).T
            tp_pulses = pulseshapes_pred[:, ~ nn_pred_pulse.mask]
            fp_pulses = pulseshapes_pred[:, nn_pred_pulse.mask]
        
            pulseshapes_true = pulse_utils.get_pulseshapes(pulsetimes_true * fs + win_hw, x_test, win_hw)
            pulsenorm_true = np.linalg.norm(np.abs(pulseshapes_true[50:-50,:]), axis = 0)
            pulsefreq_true = np.array([pulse_utils.pulse_freq(p)[0] for p in pulseshapes_true[50:-50, :].T])
            pulseshapes_true = np.apply_along_axis(pulse_utils.normalize_pulse, axis = -1, arr = pulseshapes_true.T).T
        
            fn_pulses = pulseshapes_true[:, nn_true_pulse.mask]
        
            plt.gcf().set_size_inches(24, 12)
            
            plot_pulses(tp_pulses, 1, f'True positives (N={tp_pulses.shape[1]})')
            plot_pulses(fp_pulses, 2, f'False positives (N={fp_pulses.shape[1]})')
            plot_pulses(fn_pulses, 3, f'False negatives (N={fn_pulses.shape[1]})')
            
            plt.savefig('The shapes of false/true positive/negative pulses' + '_' + str(data_idx) + '_' + str(train_flag) + '.png')
            
        # Check the amplitude and frequency of pulses as well as the pulse confidence scores
        if pulse_pred_index is not None:
            plt.gcf().set_size_inches(18, 4)
            
            plt.subplot(131)
            sns.kdeplot(np.log2(pulsenorm_pred[~ nn_pred_pulse.mask]), shade = True, clip = (-6, 4));
            sns.kdeplot(np.log2(pulsenorm_pred[nn_pred_pulse.mask]), shade = True, clip = (-6, 4));
            sns.kdeplot(np.log2(pulsenorm_true[nn_true_pulse.mask]), shade = True, clip = (-6, 4));
            plt.legend(['True positives', 'False positives', 'False negatives'])
            plt.xlim(-6, 4)
            plt.xlabel('Pulse amplitude (log units)')
            plt.ylabel('Probability')
            plt.locator_params(axis = 'y', nbins = 4)
            plt.locator_params(axis = 'x', nbins = 6)
            utils_plot.remove_axes()
            
            plt.subplot(132)
            sns.kdeplot(pulsefreq_pred[~ nn_pred_pulse.mask], shade = True, clip = (0, 600));
            sns.kdeplot(pulsefreq_pred[nn_pred_pulse.mask], shade = True, clip = (0, 600));
            sns.kdeplot(pulsefreq_true[nn_true_pulse.mask], shade = True, clip = (0, 600));
            plt.legend(['True positives', 'False positives', 'False negatives'])
            plt.xlim(0, 600)
            plt.xlabel('Pulse frequency [Hz]')
            plt.ylabel('Probability')
            plt.locator_params(axis = 'y', nbins = 4)
            plt.locator_params(axis = 'x', nbins = 6)
            utils_plot.remove_axes()
        
            plt.subplot(133)
            sns.kdeplot(pulsetimes_pred_confidence[~ nn_pred_pulse.mask], shade = True, clip = (0, 1))
            sns.kdeplot(pulsetimes_pred_confidence[nn_pred_pulse.mask], shade = True, clip = (0, 1))
            plt.axis('tight')
            plt.xlim(0.5, 1)
            plt.legend(['True positives', 'False positives'], loc = 'upper left')
            plt.xlabel('Pulse confidence score')
            plt.ylabel('Probability (?)')
            plt.locator_params(axis = 'y', nbins = 4)
            plt.locator_params(axis = 'x', nbins = 8)
            utils_plot.remove_axes()
            
            plt.savefig('The amplitude and frequency of pulses as well as the pulse confidence scores' + '_' + str(data_idx) + '_' + str(train_flag) + '.png')
            
        # Class-wise precision-recall curves
        plt.gcf().set_size_inches(20, 15)
        cls_label = params['class_names']
        
        for cls in range(1, len(cls_label)):
            precision, recall, threshold = sklearn.metrics.precision_recall_curve(y_test[::10, cls] > 0.5, y_pred[::10, cls])    
            f1score = 2 * (precision * recall) / (precision + recall)
            
            threshold_opt = threshold[np.argmax(f1score)]
            
            plt.subplot(2, 2, cls)
            plt.scatter(precision[:-1:10], recall[:-1:10], c = threshold[::10])
            plt.xlabel('Precision')
            plt.ylabel('Recall')
            plt.title(cls_label[cls].capitalize())
            plt.axis('square')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            
            plt.subplot(2, 2, 2 + cls)
            print(threshold.shape, f1score.shape)
            plt.scatter(threshold[:], f1score[:-1], c = threshold[:])
            plt.axvline(threshold_opt, c = 'k')
            plt.xlabel('Threshold')
            plt.ylabel('F1 score')
            plt.axis('square')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            print(f'Best f1 score of {np.max(f1score):1.2f} at a threshold at {threshold_opt:1.2f}.')
            
        plt.savefig('Class-wise precision-recall curves' + '_' + str(data_idx) + '_' + str(train_flag) + '.png')
            
        labels_pred = predict.labels_from_probabilities(y_pred[:, 1], threshold = threshold_opt)
        conf_mat, report = evaluate.evaluate_segments(labels_test, labels_pred, params['class_names'])
        print('Performance with treshold that maximizes the f1 score (geometric mean of precision and recall):')
        print(report)
        
        # To bias towards precision (reduce false positives), you can chose a slightly higher threshold
        labels_pred = predict.labels_from_probabilities(y_pred[:, 1], threshold = 0.5)  
        conf_mat, report = evaluate.evaluate_segments(labels_test, labels_pred, params['class_names'])
        print('Performance with slightly higher treshold:')
        print(report)
        
        # Fine-tune segment prediction - min_len and gap_fill
        # Maybe alternatively evaluate timing of onsets and offsets?
        def fill_gaps(sine_pred, gap_dur = 100):
            onsets = np.where(np.diff(sine_pred.astype(np.int)) == 1)[0]
            offsets = np.where(np.diff(sine_pred.astype(np.int)) == -1)[0]
            
            if len(onsets) and len(offsets):
                onsets = onsets[onsets < offsets[-1]]
                offsets = offsets[offsets > onsets[0]]
                durations = offsets - onsets
                
                for idx, (onset, offset, duration) in enumerate(zip(onsets, offsets, durations)):
                    if idx > 0 and offsets[idx - 1] > onsets[idx] - gap_dur:
                        sine_pred[offsets[idx - 1]:onsets[idx] + 1] = 1
            
            return sine_pred
        
        def remove_short(sine_pred, min_len = 100):
            # Remove too short sine songs
            onsets = np.where(np.diff(sine_pred.astype(np.int)) == 1)[0]
            offsets = np.where(np.diff(sine_pred.astype(np.int)) == -1)[0]
            
            if len(onsets) and len(offsets):
                onsets = onsets[onsets < offsets[-1]]
                offsets = offsets[offsets > onsets[0]]
                durations = offsets - onsets
                
                for cnt, (onset, offset, duration) in enumerate(zip(onsets, offsets, durations)):
                    if duration < min_len:
                        sine_pred[onset:offset + 1] = 0
            
            return sine_pred

        labels_pred1 = labels_pred.copy()
        
        gap_dur = 40 / 1000 * fs  # Fill all gaps between segments smaller than 40 ms
        labels_pred2 = fill_gaps(labels_pred1.copy(), gap_dur)
        
        min_len = 25 / 1000 * fs  # Delete all remaining segments shorter than 25 ms
        labels_pred3 = remove_short(labels_pred2.copy(), min_len)
        
        conf_mat, report = evaluate.evaluate_segments(labels_test, labels_pred2, params['class_names'])
        print('Performance after filling gaps:')
        print(report)
        
        conf_mat, report = evaluate.evaluate_segments(labels_test, labels_pred3, params['class_names'])
        print('Performance after filling gaps and deleting short segments:')
        print(report)
        
        def fixlen(onsets, offsets):
            if len(onsets) > len(offsets):
                onsets = onsets[:-1]
            elif len(offsets) > len(onsets):
                offsets = offsets[1:]
            
            return onsets, offsets
        
        tol = 0.05  # seconds = 50 ms
        print(fs)
        
        if segment_pred_index is not None:
            segment_onset_times_true, segment_offset_times_true = fixlen(*evaluate.segment_timing(labels_test, fs))
            segment_onset_times_pred, segment_offset_times_pred = fixlen(*evaluate.segment_timing(labels_pred3, fs))
            
            durations_true = segment_offset_times_true - segment_onset_times_true
            durations_pred = segment_offset_times_pred - segment_onset_times_pred
        
            segment_onsets_report, segment_offsets_report, nearest_predicted_onsets, nearest_predicted_offsets = evaluate.evaluate_segment_timing(labels_test, labels_pred3, fs, tol)
                    
            print(segment_onsets_report)
            print(segment_offsets_report)
            
            print(f'Temporal errors of all predicted sine onsets: {np.median(nearest_predicted_onsets) * 1000:1.2f} ms')
            print(f'Temporal errors of all predicted sine offsets: {np.median(nearest_predicted_offsets) * 1000:1.2f} ms')
            
            plt.gcf().set_size_inches(15, 5)
            
            plt.subplot(131)
            plt.hist(nearest_predicted_onsets, bins = np.arange(0, 10 * tol, 0.01), density = True)
            plt.axvline(tol, color = 'k')
            plt.xlabel('Distance to segment onset [s]')
            plt.ylabel('Probability');
            utils_plot.remove_axes()
        
            plt.subplot(132)
            plt.hist(nearest_predicted_offsets, bins = np.arange(0, 10 * tol, 0.01), density = True)
            plt.axvline(tol, color = 'k')
            plt.xlabel('Distance to segment offset [s]')
            plt.ylabel('Probability');
            utils_plot.remove_axes()
        
            plt.subplot(133)
            plt.hist(durations_true, bins = np.arange(0, 2, 0.05), histtype = 'bar', label = 'true', alpha = 0.33)
            plt.hist(durations_pred, bins = np.arange(0, 2, 0.05), histtype = 'bar', label = 'pred', alpha = 0.33)
            plt.xlabel('Segment duration [seconds]')
            plt.ylabel('Count')
            plt.legend()
            utils_plot.remove_axes()
            
            print(f'Temporal errors of matched onsets: {np.mean(nearest_predicted_onsets[nearest_predicted_onsets < tol]) * 1000:1.2f} ms')
            print(f'Temporal errors of matched offsets: {np.mean(nearest_predicted_offsets[nearest_predicted_offsets < tol]) * 1000:1.2f} ms')
        
            plt.savefig('Fine-tuning of segment prediction (a)' + '_' + str(data_idx) + '_' + str(train_flag) + '.png')
                       
        t0 = 0
        t1 = t0 + 100_000
        
        nb_channels = x_test.shape[1]
        x_snippet = x_test[y_offset + t0 * dwn_smp:y_offset + t1 * dwn_smp, :]
        
        plt.gcf().set_size_inches(40, 10)
        
        plt.subplot((nb_channels + 5) / 2, 1, 1)
        plt.plot(x_snippet + np.arange(nb_channels) / 10, c = 'k', linewidth = 0.5)
        plt.xticks([])
                
        plt.subplot(nb_channels + 5, 1, 3)
        plt.imshow(y_test[t0:t1].T, cmap = 'Blues')
        plt.yticks((0, 1), labels = params['class_names'])
        plt.ylabel('FSS')
        plt.xticks([])
        utils_plot.remove_axes()
                
        plt.subplot(nb_channels + 5, 1, 4)
        plt.imshow(y_pred[t0:t1].T, cmap = 'Oranges')
        plt.plot(labels_pred3[t0:t1], linewidth = 2)
        plt.yticks((0, 1), labels = params['class_names'])
        plt.ylabel('TCN')
        plt.xlim(0, t1 - t0)
        utils_plot.remove_axes()
        
        for cnt, x in enumerate(x_snippet.T):
            specgram = librosa.feature.melspectrogram(x, sr = 10000, n_fft = 512, hop_length = 1, power = 1)
            plt.subplot(nb_channels + 5, 1, 5 + cnt)
            librosa.display.specshow(np.log2(1 + specgram), sr = 10_000, hop_length = 1, y_axis = 'mel', x_axis = 'ms')
            plt.clim(0, 0.2)
            plt.ylim(0, 500)
            utils_plot.remove_axes()
            
            if cnt < 8:
                plt.xticks([])
                plt.xlabel([])
            
            plt.yticks(np.arange(0, 500, 100))
            plt.ylabel(f'Freq for chan{cnt}')
            
        plt.savefig('Fine-tuning of segment prediction (b)' + '_' + str(data_idx) + '_' + str(train_flag) + '.png')