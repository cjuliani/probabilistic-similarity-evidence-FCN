import numpy as np
from src.utils import weighting


def pruning_by_PCA(feat_mat, var_coef):
    """Returns indices and values features kept after applying dimensionality
    reduction by PCA.

    Args:.
        feat_mat: feature matrix.
        var_coef (float): ratio of variability (information) to keep among
            features when reducing feature dimension.
    """
    # Calculate SVD (U columns = singular left vectors).
    U, Sigma, Vh = np.linalg.svd(
        a=np.nan_to_num(feat_mat),
        full_matrices=False,
        compute_uv=True)

    # Get first vector.
    vect = U[:, 0]
    inds = np.arange(U.shape[0])
    vect_indexed = np.vstack((inds, vect)).T  # (indices, values)

    # Sort the vector given its singular values.
    vect_sorted = vect_indexed[np.abs(vect_indexed[:, 1]).argsort()][::-1]

    # Normalize.
    vect_norm = weighting(vect_sorted[:, 1])

    # Keep indices of features explaining X% of variance in
    # feature maps.
    tokeep_indexes, tokeep_var = [], []
    variance_sum = 0.
    for i in range(len(vect)):
        if variance_sum >= var_coef:
            break
        variance_sum += vect_norm[i]
        tokeep_indexes.append(int(vect_sorted[i, 0]))
        tokeep_var.append(vect_sorted[i, 1])

    return tokeep_indexes, tokeep_var


def pruning_by_CV(feat_mat, condition, cv_thresh, file_name):
    """Returns indices and values features kept after applying dimensionality
    reduction by coefficient of variation (CV). CV corresponds to how much
     percent the mean is explained by standard deviation, this is to compare
     variability between features (> 1 means high variance, while < 1 means
     low variance).

     Args:
        feat_mat: class of segmentation (object).
        condition (str or None): 'above' threshold, or below if None.
        cv_thresh (float): coefficient of variation threshold above (or below)
            which pruning  is applied.
        file_name (str) name of file with weights saved.
    """
    # Get coefficient of variation.
    var = np.apply_along_axis(
        func1d=lambda x: np.std(x) / np.mean(x),
        axis=0,
        arr=feat_mat)

    # Collect indices where CV is above-equal or below-equal to 1.
    if condition == "above":
        cv_idxs = np.where(var >= cv_thresh)[0]
    else:
        cv_idxs = np.where(var <= cv_thresh)[0]

    txt = 'indices kept out of {}: {}'.format(feat_mat.shape[1], len(cv_idxs))
    print('\t\t' + txt)

    # Write results in a separate text file.
    with open(file_name, 'a') as file_a:
        file_a.write('- ' + txt + '\n')

    return cv_idxs, var[cv_idxs]
