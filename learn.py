import numpy as np

def roc_curve(y_true, y_score):
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    desc_score_indices = np.argsort(y_score)[::-1]

    y_true = y_true[desc_score_indices]
    y_score = y_score[desc_score_indices]

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
    return np.trapz(y, x)

def precision_recall_curve(y_true, probas_pred):
    # Binarer Fall angenommen: y_true in {0,1}
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

    # Füge Startpunkt (1,0) hinzu
    precisions = np.r_[1, precisions]
    recalls = np.r_[0, recalls]
    thresholds = np.r_[thresholds[0]+1, thresholds]

    return precisions, recalls, thresholds


def confusion_matrix(y_true, y_pred, labels=None):
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    label_to_ind = {l: i for i, l in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[label_to_ind[t], label_to_ind[p]] += 1
    return matrix







