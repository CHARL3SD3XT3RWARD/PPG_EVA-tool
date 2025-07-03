import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sc
from collections import defaultdict

import learn
# from sklearn.metrics import precision_recall_curve, confusion_matrix
# from sklearn.metrics import roc_curve, auc

def lin_reg(signal_chunks, cross_pos):  
    """
    A function wicht performs a lin. regression for every given signalchunk.
    
    Parameters
    ----------
    signal_chunks : 2D-array
        The sequenced signal.
    cross_pos : 2D-list
        The Position of each zero crossing for every signal_chunks
        
    Returns
    -------
    slope : list
        The slope of every lin. regression.
    intersect : list
        The y-intercept for every lin. regression.
        
    """
    slope=[]
    intersect=[]
    for idx in range(len(signal_chunks)):
        if len(cross_pos[idx])!=0: # so no errors occur if the chunk has no zero crossings
            x=np.arange(0, len(cross_pos[idx]), 1)       
            fit=sc.stats.linregress(x, cross_pos[idx])
        
            slope.append(fit[0])
            intersect.append(fit[1])
        
        else:
            slope.append(np.nan)
            intersect.append(np.nan)

    return slope, intersect

def variance(slopes, intersect, cross_pos):
    '''
    A function wich calculates the variance of the zero crossings relative to the lin. regression from the eva_toolkit.lin_reg() function.
    
    Parameters
    ----------
    slopes : list
        The slopes for every lin. regression.
    intersect : list
        The y-intersect for ever lin. regression.
    cross_pos : list
        The position of every zero crossing as an array for every signalchunk.
    
    Returns
    -------
    var : list
        The variance of the data relative to the lin. regression
        
    '''
    var=[]
    for i in range(len(slopes)):
        if not np.isnan(slopes[i]): #if no zerocrossing occured, the value is NaN
        
            x=np.arange(0, len(cross_pos[i]), 1)
            mu_i=[slopes[i] * x + intersect[i]][0]
            
            var.append(np.mean((cross_pos[i]-mu_i)**2))
        
        else:
            var.append(np.nan)
        
    return var

def import_training_data(training_values_path):# check in pipeline    
    '''
    Imports a excel-sheet with all data. This sheet must be created in beforehand.
    Futhermore, a rondomization and subdivison into test- and trainingsets is performed.   
    
    Parameters
    ----------
    training_values_path : string
        The specific path of the excel-sheet. The path is provided in the config.ini. For more information go to the PPG_EVA_GUI.set_values() function.
       
    Returns
    -------
    validation_sets: dict
        Five subsets with a raugh equal amount of good and bad data.       
    training_sets: dict
        Five subsets wich contain four validation_sets. In every training_set one validation_set is missing.
    test_set: array
        One set wich contains 20% of the whole dataset with respect to the subdivision in good and bad data.        
    
    '''
    
    
    #importin data
    df_data = pd.read_excel(training_values_path)
    df_data.set_index('Unnamed: 0', drop=True, inplace=True)
    
    #separating data
    df_good_data = df_data[df_data['Annotation'] == 1].reset_index(drop=True)
    df_bad_data = df_data[df_data['Annotation'] == 0].reset_index(drop=True)
    
    good_data = df_good_data.to_numpy()
    bad_data = df_bad_data.to_numpy()
    
    #random stuff
    good_idx = np.arange(0,len(good_data), 1)
    bad_idx = np.arange(0,len(bad_data), 1)
    
    np.random.shuffle(good_idx)
    np.random.shuffle(bad_idx)
    
    #taking 20%  for the testset e.g. unseen data for performancetesting
    len_test_idx_g = np.ceil(len(good_data)*0.2).astype(int)
    len_test_idx_b = np.ceil(len(bad_data)*0.2).astype(int)
    
    test_idx_g = good_idx[:len_test_idx_g]
    test_idx_b = bad_idx[:len_test_idx_b]
    
    #initialising testset
    test_set = np.concatenate((good_data[test_idx_g], bad_data[test_idx_b]), axis=0)
    
    #splitting the ramaining data into five subsets
    split_remain_g = np.array_split(good_idx[len_test_idx_g:], 5)
    split_remain_b = np.array_split(bad_idx[len_test_idx_b:], 5)
    
    splits = defaultdict(dict)
    
    for i, good_split in enumerate(split_remain_g):
        bad_split = split_remain_b[i]
        splits[f'subset {i}'] = {'good': good_split, 'bad': bad_split} 
    
    #filling sets
    training_sets = defaultdict(dict)
    validation_sets = defaultdict(dict)
    
    
    for subset in splits:
        val_idx_g = splits[subset]['good']
        val_idx_b = splits[subset]['bad']
        validation_sets[subset] = np.concatenate((good_data[val_idx_g], bad_data[val_idx_b]), axis=0)
        tr_sets_idx_g = np.array([])
        tr_sets_idx_b = np.array([])
        outer_keys = [k for k in splits if k != subset]
        for key in outer_keys:
            tr_sets_idx_g = np.append(tr_sets_idx_g, splits[key]['good']).astype(int)
            tr_sets_idx_b = np.append(tr_sets_idx_b, splits[key]['bad']).astype(int)
        
        training_sets[subset] = {'good': good_data[tr_sets_idx_g], 'bad': bad_data[tr_sets_idx_b]}
   
    return training_sets, validation_sets, test_set

def read_signal(mod_path, signal_key, time_key, sep=',', skiprows=0, date_format= None, header='infer'): 
    '''
    Reads the signal stored in the given path.
    
    Parameters
    ----------
    mod_path : string
        The modified filepath to the signalfile wit its name as last part.\n
        Syntax: path + 'filename'
    signal_key : string
        The keyword/number for Pandas.Dataframe signal-column.\n
        Somno:   signal_key= 2\n
        Corsano: signal_key='value'
    time_key : string
        The keyword/number for Pandas.Dataframe timestamp-column.\n
        Somno:   time_key=0\n
        Corsano: time_key='date'
    sep : string, optional
        The seperator used to seperate the columns. Only necessary for Somno. Corsanofiles use the default. The default is ','.\n
        sep_somno = ';'
    skiprows : integer, optional
        !!!Deprecated!!!
        The number of rows that should be skipped. The default is 0.
    date_format : string, optional
        Date-format for the timestamps. Only necessary for Somno ("%d.%m.%Y %H:%M:%S,%f"). The default is None.
        
    Returns
    -------
    signals : array
        The signal as a timeseries.
    timestamps : Series
        the timestamps as pandas.Series.

    '''
    df=pd.read_csv(mod_path, sep=sep, skiprows=skiprows, header=header)
    df=df.reset_index()
    
    signals =df[signal_key].to_numpy()
    timestamps= pd.to_datetime(df[time_key], format=date_format).dt.tz_localize(None)
    
    return signals, timestamps
      
def classify_data(hist, data1, data2, xedges, yedges, annotation=None, plot=False):
    '''
    Classifies the given data with the classifier. 

    Parameters
    ----------
    hist : array
        The PDF-values.
    data1 : array
        The skewness values of a signal.
    data2 : array
        The kurtosis values of a signal.
    xedges : array
        The xedges of the bins.
    yedges : array
        The yedges of the bins.
    annotation : array, optional
        The annotation. It is only used to plot the data into the PDF. The default is None.
    plot : bool, optional
        If True, the PDFs will be plotted with the annotated data. The default is False.

    Returns
    -------
    scores : array
        The likelihood of a value being good.

    '''
    
    scores = []
    #lokating the skew and kurtvalues in the histogram
    for x_val, y_val in zip(data1, data2):
        x_idx = np.searchsorted(xedges, x_val, side="right") - 1
        y_idx = np.searchsorted(yedges, y_val, side="right") - 1
        # z_idx = np.searchsorted(zedges, z_val, side="right") - 1
        
        # checking for the value being within the boundaries
        if (0 <= x_idx < hist.shape[0] and 0 <= y_idx < hist.shape[1]):
        
            score = hist[x_idx, y_idx]
        else:
            score= 0
            
        scores.append(score)
    
    
    if plot:
        good_idx = np.where(annotation==1)
        good1 = data1[good_idx]
        good2 = data2[good_idx]
        
        bad_idx = np.where(annotation==0)
        bad1 = data1[bad_idx]
        bad2 = data2[bad_idx]
        
        
        xcenters = 0.5 * (xedges[:-1] + xedges[1:])
        ycenters = 0.5 * (yedges[:-1] + yedges[1:])
        X, Y = np.meshgrid(xcenters, ycenters)
        
        # Jetzt korrekt plotten
        plt.figure(figsize=(8, 6))
        plt.contourf(X, Y, hist.T, levels=100, cmap="viridis")
        plt.scatter(good1, good2, color='b')
        plt.scatter(bad1, bad2, color='r')
        plt.colorbar(label="Glättete Häufigkeit")
        plt.xlabel("X (original scale)")
        plt.ylabel("Y (original scale)")
        plt.title("Geglättetes 2D-Histogramm mit korrekter Skalierung")
        plt.tight_layout()
        plt.show()
        
    
    return scores

def train_hist(validation_sets, hists, plot=False):
    '''
    This funktion iterates over the subsets. Here the thresholds for every subset are determined with the use of ROC.

    Parameters
    ----------
    validation_sets : dict
        A dictionaity containing the data of the subsets -> import_training_data().
    hists : dict
        A dictionairy containing the PDFs of every trainingset.
    plot : bool, optional
        If True, all five ROC-Curves are plotted. The default is False.

    Returns
    -------
    master_thresholds : dict
        A dictionairy containing the best threshold for every subset.

    '''

    master_thresholds = defaultdict(dict)
    master_thresholds_array = defaultdict(dict)
    master_fpr = defaultdict(dict)
    master_tpr = defaultdict(dict)
    master_idx = defaultdict(dict)
    master_auc = defaultdict(dict)
    
    for subset in validation_sets:
        
        data = validation_sets[subset]
        data_skew = data[:,0]
        data_kurt = data[:,1]
        global data_anno #debugg
        data_anno = data[:,5].astype(bool)
        scoring_hist = hists[subset][0]
        scoring_xedges = hists[subset][1]
        scoring_yedges = hists[subset][2]
        global scores #debugg
        #scoring
        scores = classify_data(scoring_hist, data_skew, data_kurt, scoring_xedges, scoring_yedges, data_anno, plot=False)
        
        #calc parameters
        fpr, tpr, thresholds_roc = learn.roc_curve(data_anno, scores)
        roc_auc = learn.auc(fpr, tpr)
        
        # youden_j = tpr - fpr
        # best_idx = youden_j.argmax()
        # best_threshold = thresholds[best_idx]
        
        precision, recall, thresholds_pr = learn.precision_recall_curve(data_anno, scores)
        # avg_prec = average_precision_score(data_anno, scores)

        #calc best threshold with f_beta score
        beta = 0.12 # z.B. für FP vermeiden
        f_beta_scores = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-10)
        best_idx = np.argmax(f_beta_scores)
        best_threshold = thresholds_pr[best_idx]
        
        master_thresholds[subset] = np.mean(best_threshold)

        #preparing plot
        plot_idx = np.where(best_threshold == thresholds_roc)[0]
                
        master_fpr[subset] = fpr
        master_tpr[subset] = tpr
        master_thresholds_array[subset] = thresholds_pr
        master_idx[subset]=plot_idx 
        master_auc[subset]=roc_auc
   
    if  plot:
        
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18,18))
        ax = axes.ravel() 

        for i, subset in enumerate(master_fpr):
            best_idx = master_idx[subset]
            ax[i].plot(master_fpr[subset], master_tpr[subset], color='tab:orange', lw=2, label=f'ROC curve (AUC = {master_auc[subset]:.2f})')
            ax[i].scatter(master_fpr[subset][best_idx], master_tpr[subset][best_idx], color='tab:blue')

            ax[i].plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')  # Diagonale als Referenz
            ax[i].set_xlim([0.0, 1.0])
            ax[i].set_ylim([0.0, 1.05])
            ax[i].set_xlabel('False Positive Rate')
            ax[i].set_ylabel('True Positive Rate')
            ax[i].set_title(f'Fold {i}')
            ax[i].legend(loc="lower right")
        ax[5].axis('off')
        plt.tight_layout()    
        plt.show()
            
            
    return master_thresholds
      
def mean_hists(hists, thresholds):
    '''
    Calculates the mean from all PDFs as final classifier.

    Parameters
    ----------
    hists : dict
        All five PDFs from the trainingsets.
    thresholds : dict
        best threshold for every subset.

    Returns
    -------
    mean_hist : array
        The mean PDF.        
    xedges : array
        The xedges of the bins.
    yedges : array
        The yedges of the bins.
    mean_thresh : float
        the mean threshold from the best thresholds.

    '''
   
    all_hists = []
    all_thresh = []
    for subset in hists:
        hist = hists[subset][0]
        thresh = thresholds[subset]
        all_thresh.append(thresh)
        all_hists.append(hist)
    
    mean_hist = np.mean(all_hists, axis=0) 
    mean_thresh = np.mean(all_thresh)
    xedges = hists['subset 0'][1] # since the bins and boundaries of all hist are the same
    yedges = hists['subset 0'][2] # since the bins and boundaries of all hist are the same
    # zedges = master_hists['subset 0'][3]
    
    del all_hists # saving memory
        
    return mean_hist, xedges, yedges, mean_thresh
    
def test_subsets(validation_sets, mean_hist, xedges, yedges, best_thresh, plot=False):
    '''
    This funktion applies the final classifier to the five subsets to check the consitency of the classifier. 
    It should be selfexplainatory that the performances should have similar values, otherwise something went wron and a redo is recommended.
    
    Parameters
    ----------
    validation_sets : dict
        A dictionaity containing the data of the subsets -> import_training_data().
    mean_hist : array
        The classifying PDF.
    xedges : array
        The xedges of the bins.
    yedges : array
        The yedges of the bins.
    best_thresh : float
        The threshold.
    plot : TYPE, optional
        If True, the performance of the calssifier on the subsets will be plotted in ROC-Space.
        The default is False.

    Returns
    -------
    tuple
        A tuple containing the mean fpr, mean tpr, standard deviation of fpr, standard deviation of tpr, on the respective indices.

    '''
    
    labels=[0,1]# initializing labels for the confusion matrix
    master_analytics = defaultdict(dict)

    all_fpr = []
    all_tpr = []

    for subset in validation_sets:
        
        data = validation_sets[subset]
        data_skew = data[:,0]
        data_kurt = data[:,1]
        
        data_anno = data[:,5]        

        #scoring
        scores = classify_data(mean_hist, data_skew, data_kurt,
                               xedges, yedges)
       
        #predicting
        y_pred = (scores >= best_thresh).astype(int)
            
        #confusion matrix
        cm = learn.confusion_matrix(data_anno, y_pred, labels=labels)
         
        #calc parameters            
        fpr = cm[0,1]/np.sum(cm[0]) if np.sum(cm[0]) > 0 else np.nan
        tpr = cm[1,1]/np.sum(cm[1]) if np.sum(cm[1]) > 0 else np.nan
            
        all_fpr.append(fpr)
        all_tpr.append(tpr)
        
        master_analytics[subset] = {'fpr': fpr, 'tpr': tpr}
         
    mean_fpr = np.nanmean(all_fpr)
    mean_tpr = np.nanmean(all_tpr)

    print('all tpr', all_tpr)
    print('all fpr', all_fpr)
    print('mean tpr', mean_tpr)
    print('mean fpr', mean_fpr)
 
    if plot:
        plt.figure()
        for subset in master_analytics:
            fpr = master_analytics[subset]['fpr']
            tpr = master_analytics[subset]['tpr']
            
            plt.scatter(fpr, tpr, label=subset, alpha=.5)
    
        plt.scatter(mean_fpr, mean_tpr, color ='r', label='mean')
        plt.plot([0,1], [0,1], color='k', linestyle='--')
    
        plt.legend(loc='lower right')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
       
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.tight_layout()
        plt.show()
        

    return (mean_fpr, mean_tpr, np.std(all_fpr), np.std(all_tpr))

def test_testset(test_set, mean_hist, xedges, yedges, best_thresh, val_tupel = (0,0,0,0), plot=False):
    '''
    This funktion applies the final classifier to the subsets e.g. unseen data.
    This resembles the final performancetest.

    Parameters
    ----------
    test_set : dict
        A dictionaity containing the data of the testset -> import_training_data().
    mean_hist : array
        The classifying PDF.
    xedges : array
        The xedges of the bins.
    yedges : array
        The yedges of the bins.
    best_thresh : float
        The threshold.
    val_tupel : tuple, optional
        A tuple containing the, mean fpr, mean tpr, standard deviation of fpr, standard deviation of tpr, on the respective indices.
        Only necessary if the Performance is plotted.
    plot : bool, optional
        If True, the performance ofe the classifier on the testset will be plottet in the ROC-space with the data of the subsets as errorbars. The default is False.


    Returns
    -------
    test_fpr : float
        The fpr of the classifier on the testset.
    test_tpr : float
        The tpr of the classifier on the testset.


    '''
    #label for confusion matrix
    labels = [0,1]
    
    data = test_set
    data_skew = data[:,0]
    data_kurt = data[:,1]
   
    data_anno = data[:,5]     
        
    reference = list(map(int, data_anno))
    
    #scoring    
    scores = classify_data(mean_hist, data_skew, data_kurt,
                           xedges, yedges, data_anno, plot=False)
    
    #prediction
    y_pred = (scores >= best_thresh).astype(int)
    #confusion matrix    
    cm = learn.confusion_matrix(reference, y_pred, labels=labels)
    #calc parameters
    test_fpr = cm[0,1]/np.sum(cm[0]) if np.sum(cm[0]) > 0 else np.nan
    test_tpr = cm[1,1]/np.sum(cm[1]) if np.sum(cm[1]) > 0 else np.nan

    if plot:
        plt.figure()
        plt.scatter(test_fpr, test_tpr, color ='tab:red', label=f'TPR = {test_tpr: .3f}'+'\n'+
                    f'FPR = {test_fpr: .3f}')
        plt.errorbar(val_tupel[0], val_tupel[1], xerr=val_tupel[2], yerr=val_tupel[3], color='tab:green', capsize=5, label=rf'Hold-Out TPR = {val_tupel[1]: .3f}$\pm${val_tupel[3]: .3f}' + '\n'+
                     rf'Hold-Out FPR = {val_tupel[0]: .3f}$\pm${val_tupel[2]: .3f}')
        plt.plot([0,1], [0,1], color='k', linestyle='--')
        
        plt.legend()
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.tight_layout()
        plt.show()
        
    return test_fpr, test_tpr

def wrapping_results(prediction):
    '''
    Counting the good and bad labeled sequences.

    Parameters
    ----------
    prediction : array
        The array with the prdicted labels. 

    Returns
    -------
    tuple
        A tuple containing the number of good labels, bad labels and overall number of labels.

    '''
    
    good = np.where(prediction == 1)[0]
    bad = np.where(prediction == 0)[0]
    
    return len(good), len(bad), len(good)+len(bad) 


















