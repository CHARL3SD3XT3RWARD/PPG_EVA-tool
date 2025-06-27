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


def preprocessing(signal_directory, TN, training = True, plot=True):
    '''
    A preprocessing returns the SQI as objects for each device.

    Parameters
    ----------
    training_signal_path : string
        The baspath to signaldirectory.
    TN : string
        The TN-ID to find the signals within the training_signal_path.
    chunk_length : float, optional
        The chunklength in seconds.
    training : bool, optional
        Wether PPG-Eva is in training mode or not. If False, corsano will only used for synchronicing. Was used for the aqusition of the training_data. The default is True.
    plot : bool, optional
        If True, the whole signal will be plotted. The default is True.

    '''
    # Datei einlesen
    config.read(r'A:\project\directory\config.ini') #path to config file

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
        
    #slicing  
    somno_processing.slice_list(chunk_size=chunk_size)                  
    #calc SQI     
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
            chunk_name_pos.append(chunk+((1/2)*chunk_size))
            chunk += chunk_size
            
        ax1.set_xticks(chunk_name_pos, chunk_names)
        plt.title(TN)
        plt.show()
           
    return somno_SQI
 
#%% pipeline


def process(stop_event, train=False,  plot=False, testrun = False, ):
    '''
    

    Parameters
    ----------
    TN_list : list
        A list of the names.
    train : bool, optional
        If True, PPG-Eva is in training mode. Otherwise it is in Evaluation mode. The default is False.
    plot : bool, optional
        Deprecated. The default is False.

    Returns
    -------
    in Train-mode:
        Nothing. The results of the classifier are shown. The operator is prompted to enter:
         y : classifier will be saved and PPG-Eva terminates.
         n : training will be performed again with a new randomization of the sets.
         q : Quit. PPG.Eva terminates.

    in Evaluation-mode:
        final_results_somno: dict
            A dictionairy where the names are the keys. Behind every key is a dictionairy containing the results from 
            wrapping_results()
            'good': number og good sequences
            'bad' : number of bad sequences
            'sum' : nuber of sequences

    '''



    def training():
        '''
        The trining Pipeline. It uses the extracted values from the Reference dataset to build a 
        2D-histogram based calssifier.

        
        '''
        
        config.read(r'A:\project\directory\config.ini')#path to configfile

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
             
    config.read(r'A:\project\directory\config.ini')#path to config file

    working_folder = config['Paths']['working_folder']
    export_path = config['Paths']['export_path']   
    classifier_path = config['Paths']['classifier_path']
         
    def no_train():

        
        TN_list = os.listdir(working_folder)
        
        '''
        Evaluation-mode
        Evaluates Signals based on the classifier. Both Signals are processed somewhat parallel.

        Returns
        -------
        final_result_somno : dict
            A dictionairy where the names are the keys. Behind every key is a dictionairy containing the results from 
            wrapping_results()
            'good': number og good sequences
            'bad' : number of bad sequences
            'sum' : nuber of sequences

        final_result_corsano : dict
            A dictionairy where the names are the keys. Behind every key is a dictionairy containing the results from 
            wrapping_results()
            'good': number og good sequences
            'bad' : number of bad sequences
            'sum' : nuber of sequences


        '''
        # import classifier
        with open(classifier_path, 'rb') as f:
            hist = pickle.load(f)
           

        scoring_hist = hist['counts'] 
        xedges = hist['xedges']
        yedges = hist['yedges']
        thresh = hist['threshold']
       
        final_result = defaultdict(dict)
        
        #the actual processing
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
        df_final_result.to_excel(export_path + 'results.xlsx')
        return final_result
    








