from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from sklearn.metrics.cluster import adjusted_rand_score as ari
import numpy as np

def f1(y_tru, y_prd):
    """
    caluclate F1 score where the K parameter not necessarily known

    :y_tru: the real labels (ground truth)
    :y_prd: the cluster labels
    """

    # denfie dims
    dim1 = np.unique(y_tru).shape[0]
    dim2 = np.unique(y_prd).shape[0]

    # create (empty) matrix [N1ij]
    mat = np.zeros((dim1, dim2))

    # count each cell nij
    for yi, pi in zip(y_tru, y_prd):
        mat[yi, pi] += 1

    # calculate sum of row & col (N2,N1)
    by_row = np.sum(mat, axis=1).reshape((dim1, 1))
    by_col = np.sum(mat, axis=0).reshape((1, dim2))

    # precision & recall
    precision = mat / by_col
    recall = mat / by_row

    # F matrix
    dem = np.where(precision + recall > 0, precision + recall, 1)
    fmat = (2 * precision * recall) / (dem)

    # get total score
    max_by_row = np.max(fmat, axis=1)
    row_weight = by_row.reshape(-1) / np.sum(by_row)

    # get F1
    return sum(max_by_row * row_weight)

def evl(y_tru, y_prd):
    """
    caluclate F1, ARI & NMI for clustering results

    :y_tru: the real labels (ground truth)
    :y_prd: the cluster labels
    """
    return {'F1':   f1(y_tru,y_prd),
            'ARI': ari(y_tru,y_prd),
            'NMI': nmi(y_tru,y_prd)}
