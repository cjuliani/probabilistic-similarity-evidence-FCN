import os
import numpy as np
import matplotlib.pyplot as plt


def standardize_by_minmax(x, axis=0):
    """Returns x standardized around its mean."""
    num = (x - np.min(x, axis=axis))
    denom = (np.max(x, axis=axis) - np.min(x, axis=axis))
    return np.nan_to_num(num / denom)


def plot_group_signals_histogram(grp_keys, width, color, alpha, N, signals, fig_size,
                                 path, y_lim, show, save, tick_int):
    """Displays the histogram of feature signal difference calculated
    between groups of clusters.

    Args:
        grp_keys (list): cluster (group) keys considered.
        color (str): color of histogram.
        width (int): width of histogram bars.
        alpha (float): graph transparency.
        N (int): number of features considered.
        signals: sum of signals differences.
        fig_size (tuple): size of figure.
        path (str): folder path to save the plots.
        y_lim (list): plot limits on Y.
        show (bool): if True, show the plot.
        save (bool): if True, save the plot.
        tick_int: x-axis ticks to be drawn.
    """
    # Create figure.
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)

    # Set y limits.
    ax.set_ylim(y_lim[0], y_lim[1])

    # Plot signal difference as histogram.
    _ = ax.bar(
        range(N),
        abs(signals)[:N],
        width=width,
        zorder=-1,
        color=color,
        alpha=alpha)

    # Add grid and set the x-axis ticks.
    ax.set_xticks(tick_int, minor=False)

    if save is True:
        tmp = '-'.join([str(i) for i in grp_keys])
        img_name = f"clusters_signals_histogram_N-{N}_clusters-{tmp}"
        output_name = os.path.join(os.path.abspath(path), img_name)
        plt.savefig(output_name, dpi=300)

    if show is True:
        plt.show()


def plot_group_signals(fts, cls_groups, cls_colors, grp_keys, N, indices, curves_lw,
                       size, alpha, mode, fig_size, path, y_lim, show, save):
    """Displays the scatter plot of feature signal difference calculated
    between groups of clusters (of images), or single x images.

    Args:
        fts: layer features.
        cls_groups: groups of cluster IDs.
        cls_colors (list): list of colors per group.
        grp_keys (list): cluster (group) keys considered.
        N (int): number of features considered.
        indices: feature indices, which can be sorted according to the
            importance of signal difference.
        curves_lw: line width of plotted curve.
        size (list): sizes of scatters according to groups.
        alpha (float): graph transparency.
        mode (str): plotting mode ('scatter' or 'mean_curve').
        fig_size (tuple): size of figure.
        path (str): folder path to save the plots.
        y_lim (list): plot limits on Y.
        show (bool): if True, show the plot.
        save (bool): if True, save the plot.
    """
    # Get groups of clusters given indices.
    groups = [cls_groups[i] for i in grp_keys]
    colors = [cls_colors[i] for i in range(len(grp_keys))]

    # Create figure.
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)

    # Set y limits.
    ax.set_ylim(y_lim[0], y_lim[1])

    grp_i = 0
    if mode == "scatter":
        # Plot scatter points.
        for grp in groups:
            cls_i = 0
            for cls in grp:
                for i in cls:
                    if type(indices) is list:
                        r = standardize_by_minmax(fts)[i, indices[:N]]
                    else:
                        r = standardize_by_minmax(fts)[i, :N]
                    ax.scatter(range(N), r, s=size[grp_i], c=colors[grp_i][cls_i], alpha=alpha)
                cls_i += 1
            grp_i += 1
    elif mode == "mean_curve":
        # Plot curves.
        for grp in groups:
            cls_i = 0
            for cls in grp:
                if type(indices) is list:
                    r = standardize_by_minmax(fts)[cls, :][:, indices[:N]]
                else:
                    r = standardize_by_minmax(fts)[cls, :N]
                r = np.median(r, axis=0)
                ax.plot(range(N), r, color=colors[grp_i][cls_i], linewidth=curves_lw, alpha=0.5)
                ax.scatter(range(N), r, s=size[grp_i], c=colors[grp_i][cls_i], alpha=alpha)
                cls_i += 1
            grp_i += 1

    if save is True:
        tmp = '-'.join([str(i) for i in grp_keys])
        img_name = f"clusters_signals_N-{N}_mode-{mode}_clusters-{tmp}"
        output_name = os.path.join(os.path.abspath(path), img_name)
        plt.savefig(output_name, dpi=300)

    if show is True:
        plt.show()


def get_sum_signal_difference(feats, clusters, sorted_diff, mode, norm=True):
    """ Returns the indices and values of the summed difference of
    signals between groups of clusters.

    Args:
        feats: feature matrix.
        clusters (list): list of clusters.
        sorted_diff (bool): if True, sort signal difference by order of
            importance.
        mode (str): 'per_cluster' for differentiating clusters; features are
            analyzed otherwise.
        norm (bool): if True, normalize features in min-max (0-1) scale.
    """
    if norm is True:
        # Normalize features.
        feats = standardize_by_minmax(feats)

    # Get groups of clusters given indices.
    diff_lst = []
    for i in range(len(clusters)):
        # Get N non-pruned features given reference cluster's IDs.
        ref = feats[clusters[i]]
        ref = np.median(ref, axis=0) if mode == 'per_cluster' else ref

        for j in range(i + 1, len(clusters)):
            # Get temporary feature vector other than the reference one.
            tmp = feats[clusters[j]]
            tmp = np.median(tmp, axis=0) if mode == 'per_cluster' else tmp

            # Combine signals.
            comb = np.array([ref, tmp]) if mode == 'per_cluster' else np.concatenate([ref, tmp], axis=0)

            # Get difference and rule out negative signs for cumulative
            # sum.
            tmp_diff = abs(np.diff(comb, axis=0)[0])
            diff_lst.append(tmp_diff)

    # Transform to array.
    diff_sum = np.sum(np.array(diff_lst), axis=0)

    if sorted_diff is True:
        # Sort signals difference and indices given different values.
        diff_sum_s = np.vstack([range(feats.shape[1]), diff_sum]).T
        diff_sum_s = diff_sum_s[diff_sum_s[:, 1].argsort()]
        return list(diff_sum_s[:, 0].astype(int)), diff_sum_s[:, 1]
    else:
        return list(range(feats.shape[1])), diff_sum


def get_signal_percentage(signals, interval):
    """Returns the percentage of data whose signals fall within
    provided interval"""
    tmp = signals[interval[0] <= signals]
    tmp = tmp[tmp <= interval[1]]
    pct = len(tmp) / len(signals)
    print('Percentage between {}-{}: {:.2f}%'.format(interval[0], interval[1], pct * 100))
    return pct


def plot_signal_percentage(signals, path, mode, save):
    """Displays the percentage of signal values within intervals.

    Args:
        signals: signal difference summed.
        path (str): saving path of plot.
        mode (str): type of plot, i.e. 'histogram' or 'cumsum'.
        save (bool): if True, save the plot.
    """
    msg = "Choose appropriate mode between 'histogram' and 'cumsum'."
    assert mode == "histogram" or mode == "cumsum", msg

    if mode == "histogram":
        # Get percentages per interval.
        intervals = [[0., 0.1], [0.1, 0.2], [0.2, 0.3],
                     [0.3, 0.4], [0.4, 0.5], [0.5, 0.6],
                     [0.6, 0.7], [0.7, 0.8], [0.8, 0.9],
                     [0.9, 1.]]
        ys = []
        for intv in intervals:
            tmp = get_signal_percentage(signals, intv)
            ys.append(tmp)

        # Plot histograms.
        for i in range(0, 10, 1):
            try:
                plt.plot([i / 10, (i + 1) / 10], [ys[i], ys[i]], color='black')
                plt.plot([(i + 1) / 10, (i + 1) / 10], [ys[i], ys[i + 1]], color='black')
            except Exception as _:
                pass

    elif mode == "cumsum":
        # Plot cumulative sum of signal values at given intervals.
        intervals = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
        vcumsum = np.array([((np.sum((signals < v).astype(int)) / len(signals)) * 100) for v in intervals])
        plt.plot(intervals, vcumsum)

    if save is True:
        img_name = "signal_change_percentage_mode-{}".format(mode)
        output_name = os.path.join(os.path.abspath(path), img_name)
        plt.savefig(output_name, dpi=300)

    plt.show()


def get_signal_per_layer(layers, signals, intervals):
    """Returns percentage of total signal change per layer.

    Args:
        layers (list): convolution layers considered.
        signals: signal difference summed.
        intervals (list): intervals of signal considered.
    """
    ratios = []
    for i in range(len(intervals)):
        try:
            tmp_sum = np.sum(signals[intervals[i]:intervals[i + 1]])
            tmp_ratio = tmp_sum / np.sum(signals)
            ratios.append(tmp_ratio)
        except Exception as _:
            pass

    tmp_layers = ['{}'.format(i) for i in layers]
    ratios = ['{:.2f}%'.format(i * 100) for i in ratios]
    print('\nLayers: {}'.format(', '.join(tmp_layers)))
    print('\t▹ Ratios per layer: {}\n'.format(', '.join(ratios)))
