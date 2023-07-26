import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from utils import weighting


def fuzzify(layers, feat_stats, feat_idxs, stat_types, method, obj_range, gfactor,
            targets_id, weights, norm, sorting=True):
    """Returns the probability values and indices of fuzzified feature
    values according to the membership function used.

    Args:
        layers (list): convolution layers considered in fuzzy analysis.
        feat_stats (dict): dictionary of features.
        feat_idxs (): dictionary of features indices.
        stat_types (list): type_ of statistics ('mean', 'std', 'cv')
        method (str): membership function used in fuzzy inference.
        obj_range (int): number of object to consider.
        gfactor (float): standard deviation factor for the gaussian membership
            function OR ratio for outer margins of trapeze.
        targets_id (list): indices of target images considered.
        weights: 'pca' weights calculate per layer, or normalized sum of signal
            difference between clusters ('speciation').
        norm (bool): if True, normalize features (min-max) before fuzzification.
        sorting (bool): if True, sort probability values in decreasing order.
    """
    msg = "Non-existing method. Choose between 'gaussian' or 'trapeze' or 'gaussian_centroid'."
    assert method == "trapeze" or method == "gaussian" or method == 'gaussian_centroid', msg

    probs = []
    for type_ in stat_types:
        # Extract target ranges.
        tmp_lst = []
        for lyr in layers:
            idxs_tmp = feat_idxs[lyr]
            fts_tmp = feat_stats[lyr][type_][:, idxs_tmp][:obj_range]
            tmp_lst.append(fts_tmp)

        fts_tmp = np.concatenate(tmp_lst, axis=1)

        # Get the standard deviation of the feature distribution.
        gmf_tgt_std = np.std(fts_tmp, axis=0)

        # Assert that current network feature matrix has same
        # feature dimension than its related weight vector.
        msg = (f"Dimension of statistical features ({fts_tmp.shape[-1]}) does not match " +
               f"with weights provided ({len(weights)}). You may have to use 'get_statistics' " +
               f"again with the right number of non-pruned features.")
        assert fts_tmp.shape[-1] == len(weights), msg

        if norm is True:
            # Standardize feature values in the range 0-1.
            fts_tmp = standardize_by_minmax(fts_tmp)

        # Get target feature values.
        target_fts_tmp = np.vstack([fts_tmp[i] for i in targets_id])

        if method == 'gaussian_centroid':
            # Average gaussian centroid of all targets.
            target_fts_mean = np.mean(target_fts_tmp, axis=0)
            target_fts_std = np.std(target_fts_tmp, axis=0) * gfactor

        fuzzy_tmp = []
        for i in range(fts_tmp.shape[1]):
            if method == 'trapeze':
                # Get membership function parameters.
                # Formula: https://pythonhosted.org/scikit-fuzzy/_modules/skfuzzy/membership/generatemf.html#gaussmf
                inner1, inner2 = np.min(target_fts_tmp[:, i]), np.max(target_fts_tmp[:, i])  # get trapeze limits
                rge = inner2 - inner1  # get range
                outer_rge = rge * gfactor  # take xx% of range
                outer1, outer2 = inner1 - outer_rge, inner2 + outer_rge

                # Apply function.
                mfx1 = fuzz.trapmf(fts_tmp[:, i], np.array([outer1, inner1, inner2, outer2]))

                # Collect fuzzy values.
                fuzzy_tmp.append(mfx1)

            elif method == 'gaussian':
                # Map to gaussian value range iteratively given target vectors.
                # Formula: np.exp(-((x - mean) ** 2.) / float(sigma) ** 2.)
                gmf_lst_tmp = []
                for j in range(target_fts_tmp.shape[0]):
                    gmf_tgt_tmp = fuzz.gaussmf(x=fts_tmp[:, i], mean=target_fts_tmp[j][i],
                                               sigma=gfactor * gmf_tgt_std[i])
                    gmf_lst_tmp.append(np.nan_to_num(gmf_tgt_tmp))

                # Takes the maximum fuzzy value for the feature vectors
                # considered.
                gmf_lst_tmp = np.max(np.array(gmf_lst_tmp), axis=0)

                # Collect fuzzy values.
                fuzzy_tmp.append(gmf_lst_tmp)

            elif method == 'gaussian_centroid':
                # Map to gaussian values given centroid of mean axis calculated
                # from target vectors.
                gmf_tgt_tmp = fuzz.gaussmf(x=fts_tmp[:, i], mean=target_fts_mean[i], sigma=target_fts_std[i])
                gmf_tgt_tmp = np.nan_to_num(gmf_tgt_tmp, 0.)

                # Collect fuzzy values.
                fuzzy_tmp.append(gmf_tgt_tmp)

        # Convert to array.
        fuzzy_mat = np.array(fuzzy_tmp).T

        # Get feature weights.
        weights_ = weighting(weights)
        probs.append(np.average(fuzzy_mat, weights=weights_, axis=-1))

    # Get probability array.
    probs = np.array(probs).T  # 400x3

    # Give indices to probabilities.
    ids = np.arange(probs.shape[0])
    sprobs = np.mean(probs, axis=-1)
    comb_tmp = np.vstack([ids, sprobs]).T

    if sorting is True:
        # Only if you consider checking sorted results (given fuzzy
        # probability values).
        comb_tmp = comb_tmp[comb_tmp[:, 1].argsort()]

    return probs, comb_tmp[:, 1], comb_tmp[:, 0].astype(int)


def format_text(text):
    """Returns formatted text without forward slash symbol."""
    return text.replace('/', '_')


def plot_images(x, prob_values, indices, N, name, path, title, show=True, save=False):
    """Displays images according to fuzzy results.

    Args:
        x (list): list of image arrays.
        prob_values: sorted fuzzy probability values.
        indices: sorted indices associated to fuzzy probabilities.
        N (int): number of similar objects to display, in regard to
            the target(s).
        name (str): image name to save.
        path (str): folder path to save the plots.
        title (str): tile of plot.
        show (bool): if True, show the plot.
        save (bool): if True, save the plot.
    """
    # Create figure and set up the indices and probability ranges.
    fig = plt.figure()
    selected_ids = indices[-N:][::-1]
    selected_probs = prob_values[-N:][::-1]

    # Define grid of images to display, which consists of 10
    # columns.
    intv = 10
    n_columns = len(selected_ids)
    n_rows = int(n_columns // intv) + 1

    cnt = 0
    for i in selected_ids:
        # Plot image at given grid location.
        _ = ax = fig.add_subplot(n_rows, intv, cnt + 1)
        _ = plt.gca().axes.get_yaxis().set_visible(False)
        _ = plt.gca().axes.get_xaxis().set_visible(False)
        _ = ax.set_frame_on(False)
        _ = ax.imshow(x[i])

        # Display subplot title.
        if title == 'probability':
            _ = ax.set_title('{0:.2f}'.format(selected_probs[cnt]))
        else:
            _ = ax.set_title('{}'.format(i))
        cnt += 1

    if save is True:
        name_ = '{}.png'.format(name)
        output_name = os.path.join(os.path.abspath(path), name_)
        plt.savefig(output_name, dpi=300)
        print('○ ' + name_ + ' saved.')

    if show is True:
        plt.show()

    # Reset rc parameters to avoid getting red frame on other
    # plots.
    _ = matplotlib.rcParams.update(matplotlib.rcParamsDefault)


def plot_features(feats, sprobs, layer, feat_id, path, name, show=True, save=False):
    """ Displays layer representations of sub-clusters.

    Args:
        feats (list): features arrays considered.
        sprobs: sorted fuzzy probability values.
        layer: layer considered to display features.
        feat_id (int): activated output ID of feature layer to display
            if show_mode  is 'features'. Note that 'layer_ft_id' is the
            index from 'pruning_feat_idxs'.
        path (str): folder path to save the plots.
        name (str): image name to save.
        show (bool): if True, show the plot.
        save (bool): if True, save the plot.
    """
    # Set up figure and title.
    fig = plt.figure()
    fig.suptitle('Layer {} feature {}'.format(layer, feat_id), fontsize=16)

    # Define grid of images to display, which consists of 10
    # columns.
    intv = 10
    n_columns = len(sprobs)
    n_rows = int(n_columns // intv) + 1

    cnt = 0
    for ftp in feats:
        # Plot image at given grid location.
        _ = ax = fig.add_subplot(n_rows, intv, cnt + 1)
        _ = plt.gca().axes.get_yaxis().set_visible(False)
        _ = plt.gca().axes.get_xaxis().set_visible(False)
        _ = ax.set_frame_on(False)
        _ = ax.imshow(ftp, cmap='jet')

        # Display subplot title.
        _ = ax.set_title('{0:.2f}'.format(sprobs[cnt]))
        cnt += 1

    if save is True:
        name_ = '{}_layer-{}_ft-{}.png'.format(name, format_text(layer), feat_id)
        output_name = os.path.join(os.path.abspath(path), name_)
        plt.savefig(output_name, dpi=300)
        print('○ ' + name_ + ' saved.')

    if show is True:
        plt.show()

    # Reset rc parameters to avoid getting red frame on other plots.
    _ = matplotlib.rcParams.update(matplotlib.rcParamsDefault)


def standardize_by_minmax(x, axis=0, eps=1e-10):
    """Returns the x standardized around the mean."""
    min_val = np.min(x, axis=axis)
    return (x - min_val) / (np.max(x, axis=axis) - min_val + eps)
