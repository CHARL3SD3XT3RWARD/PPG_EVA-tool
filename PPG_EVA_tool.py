# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 14:37:23 2025

@author: alko18
"""
#import pyedflib as plib

# import ast
# import warnings

import numpy as np

import matplotlib.pyplot as plt
from collections import defaultdict
import os
import pickle
import matplotlib as mpl
import pandas as pd
import eva_toolkit as kit
import eva_classes as evaclass

import configparser

config = configparser.ConfigParser() 

config_path = r'' # path to config.ini

mpl.rcParams.update({
    "figure.figsize": (6, 4),
    "axes.titlesize": 20,
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 15,
    "lines.linewidth": 2,
    "axes.grid": True,
    "grid.alpha": 0.7,
    "font.family": "serif",  # Oder "sans-serif"
    "savefig.dpi": 300
})

#%% classes


def preprocessing(signal_directory, TN, plot=True):
    '''
    The preprocessing pipeline. Starting with the signalimport. It continues with the filtering and slicing into chunks. Returns a object containing all SQIs.

    Parameters
    ----------
    signal_directory : string
        The baspath to signaldirectory. In that directory, only signals in .txt-format are allowed.
    TN : string
        The TN-ID to find the signals within the signal_directory. The IDs are provided by the os.listdir()-funktion. Therefore The ID is synonymous for the filename.
    plot : bool, optional
        If True, the continous signal will be plotted. The chunks will be seperated by dashed lines. The default is True.

    '''
    # Datei einlesen
    config.read(config_path) #path to config file

    fs_A = int(config['Settings']['fs_a'])
    chunk_length = float(config['Settings']['chunk_length'])
    lowcut = float(config['Settings']['lowcut'])
    highcut = float(config['Settings']['highcut'])
    order = int(config['Settings']['order'])


    #clac signallength. signal_length is in config
    chunk_size=int(chunk_length*fs_A)     

    #importing corsanesignal
    file_path=os.path.join(signal_directory, TN)# rf"{training_signal}\{TN_list[i]}"
        
    #importing somnosignal
    somno_signals, somno_timestamps=kit.read_signal(file_path, signal_key=2, time_key=0, sep=';', header=None)
  
    #processing somno e.g. filter, sequencing and calc SQIs
    global somno_processing #debugg
    somno_processing=evaclass.Processing(signals=somno_signals, timestamps=somno_timestamps)  
        
    somno_processing.pleth_filter(fs=fs_A, lowcut=lowcut, highcut=highcut, order=order,)   
        
    slicing  
    somno_processing.slice_list(chunk_size=chunk_size)                  
    calc SQI     
    somno_SQI=evaclass.SQI(somno_processing.signal_chunks)
            
    somno_SQI.skewness()
    somno_SQI.kurt()
    somno_SQI.calc_SNR()
    somno_SQI.ZCR()
    somno_SQI.shanon_entropy()
        
            
    if plot:
        fig, ax1 = plt.subplots(figsize=(18,4))
        
        ax1.plot(somno_signals)
        chunk=0
        chunk_names = []
        chunk_name_pos = []
        for i in range(len(somno_processing.signal_chunks)):
            ax1.vlines(chunk, np.min(somno_signals), np.max(somno_signals), color='k', linestyle='--')
            chunk_names.append(f'Chunk {i}')
            # chunk_name_pos.append(chunk+((1/2)*chunk_size))
            # chunk += chunk_size
            
        ax1.set_xticks(chunk_name_pos, chunk_names)
        plt.title(TN)
        plt.show()
           
    return somno_SQI
 
#%% pipeline


def process(stop_event, train=False, testrun = False, ):
    '''
    The main funktion. Here it is decided wether PPG-Eva is in trainingmode or not.

    Parameters
    ----------
    stop_event: objekt
        Provides a stop event so PPG-Eva can be aborted while running. For convenience purpose.
    train : bool, optional
        If False, PPG-Eva is in Evaluationmode.
        It starts with importing all filenames from the given directory and the classifier. 
        Then it iterates over all elements in TN_list. The preprocessing is provided by PPG_EVA_tool.preprocessing().
        After that, it creates a binary scoring and wraps the data into a dictionairy.\n
        final_result : dict
            The Evaluation of each signal. It gives the number of good chunks, bad chunks, the number of all chunks, and the proportion of the number of bad chunks to all chunks.
            Example::
                final_result[name] = {
                                      'good': good_chunks, 
                                      'bad': bad_chunks, 
                                      'sum': sum_, 
                                      'bad/sum': bad_chunks/sum_
                                      }
    
        Otherwise it is in trainingmode. It needs a datafile and a annotationfile. The filepath to these two are set while installation.
        For more info, got to the PPG_EVA_GUI.set_values() docs. While processing, the ROC-curves of the 5 subsets will be plottet.
        Further the performance of the classifier on the five subsets and on the testset will be shown in ROC-space. 
        There will be a prompt in the kernel asking wether to save [s] the classifier, start over [n] or quit [q].
        The default is False.
    testrun : bool, optional
        For evaluationmode only. It takes the first signal from the directory and process it to check for errors. The default is False.

    '''



    def training():
        '''
        The training-pipeline. It needs a datafile and a annotationfile. The filepath to these two are set while installation.
        For more info, got to the PPG_EVA_GUI.set_values() docs. While processing, the ROC-curves of the 5 subsets will be plottet.
        Further the performance of the classifier on the five subsets and on the testset will be shown. 
        There will be a prompt in the kernel asking wether to save [s] the classifier, start over [n] or quit [q].

        '''
        config.read(config_path)#path to configfile

        training_values_path = config['Paths']['training_values_path']
        classifier_path = config['Paths']['classifier_path']
       
        global training_sets, validation_sets, test_set #debugg
        training_sets, validation_sets, test_set = kit.import_training_data(training_values_path)
  
        #building hist
        master_hists = defaultdict(dict)
        
        for subset in training_sets:
            #building hist for every subset
            good = training_sets[subset]['good']
            bad = training_sets[subset]['bad']
            
            init_values=np.concatenate((good, bad), axis=0)
            skew_values = init_values[:,0]
            kurt_values = init_values[:,1]
            anno_values = init_values[:,5]#entropy is skipped
           
            obj_training = evaclass.Training(skew_values, kurt_values, anno_values)
            obj_training.separate_values()
            obj_training.building_hists(plot=False)
            
            master_hists[subset] = (obj_training.hist, obj_training.xedges, obj_training.yedges)

        #calc thresholds for every subset
        master_thresholds = kit.train_hist(validation_sets, master_hists, plot=True)
        #mean hists            
        mean_hist, xedges, yedges, best_thresh = kit.mean_hists(master_hists, master_thresholds)
        #performance test
        analysis_tupel = kit.test_subsets(validation_sets, mean_hist, xedges, yedges, best_thresh, plot=True)
        
        test_fpr, test_tpr = kit.test_testset(test_set, mean_hist, xedges, yedges, best_thresh, analysis_tupel, plot=True)
        #wrapping data for export        
        hist = {'counts': mean_hist, 'xedges': xedges, 'yedges': yedges, 'threshold': best_thresh}
        plt.pause(1)
         
        # userinput: y: save classifier; n: start over; q: quit
        save = input('Save? (y/n/q):')
        
        if save == 'y':
            with open(classifier_path, 'wb') as f:
                pickle.dump(hist, f)                
            return 0
        
        elif save =='q':
            return 0
        
        elif save == 'n':
            process(train=True)
             
    config.read(config_path)#path to config file

    working_folder = config['Paths']['working_folder']
    export_path = config['Paths']['export_path']   
    classifier_path = config['Paths']['classifier_path']
         
    def no_train():
        '''
        The Evaluationmode of PPG-Eva. It starts with importing all filenames from the given directory and the classifier. 
        Then it iterates over all elements in TN_list. The preprocessing is provided by PPG_EVA_tool.preprocessing().
        After that, it creates a binary scoring and wraps the data into a dictionairy.

        Returns
        -------
        final_result : dict
            The Evaluation of each signal. It gives the number of good chunks, bad chunks, the number of all chunks, and the proportion of the number of bad chunks to all chunks.
            Example::
                final_result[name] = {
                                      'good': good_chunks, 
                                      'bad': bad_chunks, 
                                      'sum': sum_, 
                                      'bad/sum': bad_chunks/sum_
                                      }
        '''
        
        TN_list = os.listdir(working_folder)


        import classifier
        with open(classifier_path, 'rb') as f:
            hist = pickle.load(f)
           

        scoring_hist = hist['counts'] 
        xedges = hist['xedges']
        yedges = hist['yedges']
        thresh = hist['threshold']
       
        final_result = defaultdict(dict)
        
        the actual processing
        for name in TN_list:
            if stop_event.is_set():
                print('Abgebrochen')
                return final_result
            print('Processing', name)

            try:
                SQI_objekt = preprocessing(working_folder, name, plot=False, training=False)
    
                #scoring
                scores = kit.classify_data(scoring_hist, SQI_objekt.skewness, SQI_objekt.kurt, xedges, yedges)
                #predicting            
                y_pred = (scores >= thresh).astype(int)
                #wrapping
                good_chunks, bad_chunks, sum_ = kit.wrapping_results(y_pred)
                
                final_result[name]={'good': good_chunks, 'bad': bad_chunks, 'sum': sum_, 'bad/sum': bad_chunks/sum_}
                print(final_result[name])
                if testrun:
                    break
            
            except Exception as e:
                print(f'{name} could not be processed: {e}')
                continue  
            
        return final_result
    

    
    if train:
        temp = training()
        return temp
    else:
        global final_result
        final_result = no_train()
        df_final_result = pd.DataFrame.from_dict(final_result, orient='index')
        # df_final_result.to_excel(export_path + 'results.xlsx')
        return final_result
    








