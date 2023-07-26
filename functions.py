import config
import src.analyzer as ca


# Set up agent for analyzing convolution layers.
agent = ca.ConvAgent(conv_layers=config.CONV_LAYERS)

# Detect objects and get related layers features
agent.predict(
    N=5,
    cls=config.PREDICT_LAYER_CLASS,
    cmap='jet',
    output=1)

# Collect object images.
agent.collect_image_arrays()
agent.collect_features(
    cls='conv100',
    feat_type='object')

# Get feature statistics
agent.get_obj_feat_statistics(cls='conv100')
agent.save_objects_attributes(
    classes=['conv100'],
    replace_stats=True)

# Load pre-saved feature statistics.
agent.load_objects_attributes(classes=['conv100'])

# Plot histogram of a feature mean values.
agent.plot_feature_mean_signal_distribution(
    cls='conv100',
    lyr='conv92',
    feature=2,
    incr=.001,
    stat_type='mean',
    show=True,
    save=False)

# check network sparsity (channel level)
agent.check_channel_sparsity(cls='conv100')

# Apply pruning (per layer)
agent.apply_pruning(
    mode='pca',  # by 'pca' or 'cv' (coefficient of variation)
    cv_condition='above',
    cls='conv100',
    coef=.99)

# Scatter plot of weight values associated to network features (non-pruned)
agent.plot_weigths(
    cls='conv100',
    weight_type='pca')

# Apply fuzzy evaluation of feature vectors described, which are
# by descriptive statistics (mean, std, and/or cv), given a target
# object(s).
agent.fuzzify(
    targets=[0],  # indices of target objects from dataset
    stat_types=['mean'],  # mean, std, cv
    method='gaussian',  # gaussian, gaussian_centroid, trapeze
    gfactor=0.2,  # spread parameter used in gaussian or trapeze function
    weighting=None,  # None, pca, speciation
    norm=False,
    obj_range=1000000  # limit the number of objects from dataset to compare target with
)

# Plot N objects sharing a certain similarity with target object(s)
# defined in fuzzy analysis.
agent.plot_target_similarity(
    show_mode='images',  # images, features
    out_n=20,  # number of objects shown (the closest ones to target)
    layer='conv11',  # if show mode is 'features', features of this layer showed
    layer_feat_id=0,
    title_type='probability',  # object, probability
    show=True,
    save=False)

# Explore transformed feature values per layer
agent.get_cluster_signal_difference(
    stat_type='mean',
    calc_mode='per_cluster',
    norm=True,
    norm_diff=True,
    cls_groups=config.clusters)

agent.plot_signals_histogram(
    cls_groups=config.clusters,
    width=3,
    color='black',
    alpha=1.,
    stat_type='mean',
    grp_keys=[f'{i}' for i in range(1, 10)],
    y_lim=[-0.1, 1.1],
    show=True,
    save=False)

agent.plot_group_signals(
    cls_groups=config.clusters,
    cls_groups_colors=config.cluster_colors,
    stat_type='mean',
    sz=[20, 8, 5, 3, 3, 3, 3, 3, 3],
    lw=0.,
    alpha=0.6,
    mode='mean_curve',  # mean_curve, scatter
    grp_keys=[f'{i}' for i in range(1, 10)],
    y_lim=[-0.1, 0.8],
    show=True,
    save=False)

# Check percentage of feature whose activation values are within
# a given interval among layers of the network analyzed
agent.get_signal_percentage(sig_interval=[0.2, 0.4])

# Calculates percentage of total signal change per layer
agent.get_signal_per_layer()

# Displays distribution of feature values in percentage
agent.plot_signal_percentage(
    mode='histogram',  # histogram, cumsum
    save=False)
