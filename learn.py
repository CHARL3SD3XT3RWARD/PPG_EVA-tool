import numpy as np

def roc_curve(y_true, y_score):
    '''
    Computes the ROC-values for given scores. It takes two arrays.
    One with the ground truth in binary an one with the scores.
    Both arrays must be of the same length, otherwise an error occurs.
    
    Example::
        # generating dummy ground truth
        a1 = np.zeros(10, dtype=int)
        a2 = np.ones(10, dtype=int)
        dummy_ground_truth = np.concatenate((a1, a2))
        np.random.shuffle(dummy_ground_truth)
        
        dummy_scores = np.random.rand(20)
        
        fpr_dummy, tpr_dummy, thresh_dummy = roc_curve(dummy_ground_truth, dummy_scores)
        
        print('FPR:', fpr_dummy)
        print('TPR:', tpr_dummy)
        print('thresholds:' thresh_dummy)
    
    Parameters
    ----------
    y_true : array
        The array with the graund truth values e.g. the annotation.
    y_score : array
        The array with the scoring values.

    Returns
    -------
    fpr : array
        The false positive rates for every threshold.
    tpr : array
        The true positive rate for every threshold.
    thresholds : array
        The thresholds.
    
    '''
    
    # converting input to numpyarray
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    
    # sorting in decending order
    desc_score_indices = np.argsort(y_score)[::-1]

    y_true = y_true[desc_score_indices]
    y_score = y_score[desc_score_indices]
    # finding indices where the score changes to elimenate redundancys
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size -1]

    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    tpr = tps / tps[-1]
    fpr = fps / (fps[-1] if fps[-1] > 0 else 1)

    thresholds = y_score[threshold_idxs]

    # Füge Startpunkt (0,0) hinzu
    tpr = np.r_[0, tpr]
    fpr = np.r_[0, fpr]
    thresholds = np.r_[thresholds[0] + 1, thresholds]
    
    return fpr, tpr, thresholds

def auc(x, y):
    '''
    Computes the area under curve of given ROC-values by applying the trapezodial rule.

    Parameters
    ----------
    x : array
        x-values of the given funktion e.g. the FPR-values.
    y : array
        y-values of the given funktion e.g. the TPR-values.

    Returns
    -------
    float
        The area under curve.

    '''
    return np.trapz(y, x)

def precision_recall_curve(y_true, probas_pred):
    '''
    Computes the precision-recall-curve for given predicions. 
    Much like the roc-funktion it takes two arrays of the same length.
    One with the predicting scores and one with the ground truth.

    Parameters
    ----------
    y_true : TYPE
        The array with the ground truth in binary.
    probas_pred : TYPE
        The array with the predicting scores.

    Returns
    -------
    precisions : array
        An array with the precisions for each threshold.
    recalls : array
        An array with the recall or sensitivity for each threshold.
    thresholds : array
        An array with the tresholds.

    '''
    desc_score_indices = np.argsort(probas_pred)[::-1]
    y_true = np.array(y_true)[desc_score_indices]
    probas_pred = np.array(probas_pred)[desc_score_indices]

    distinct_value_indices = np.where(np.diff(probas_pred))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    precisions = tps / (tps + fps)
    recalls = tps / tps[-1]

    thresholds = probas_pred[threshold_idxs]

    precisions = np.r_[1, precisions]
    recalls = np.r_[0, recalls]
    thresholds = np.r_[thresholds[0]+1, thresholds]

    return precisions, recalls, thresholds


def confusion_matrix(y_true, y_pred, labels=None):
    '''
    Creates a confusion matrix from the binary classification and ground truth.
    
    Parameters
    ----------
    y_true : array
        The ground truth labels.
    y_pred : array
        The predicted labels.
    labels : tupel, optional
        The desired labels. The default is None.

    Returns
    -------
    matrix : ndarray
        A 2x2 matrix containing the True Positive, True Negatife, False Positive and False Negative.

    '''
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))

    label_to_ind = {l: i for i, l in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    
    for t, p in zip(y_true, y_pred):
        matrix[label_to_ind[t], label_to_ind[p]] += 1
    
    return matrix







