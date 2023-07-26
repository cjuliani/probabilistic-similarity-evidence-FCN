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

# Apply pruning (per layer) and calculate activation statistics
agent.apply_pruning(
    mode='pca',
    cv_condition='above',
    cls='conv100',
    coef=.99)

agent.apply_pruning(
    mode='cv',
    cv_condition='above',
    cls='conv100',
    coef=.5)

agent.plot_weigths(
    cls='conv100',
    weight_type='pca')

# Similarity analysis
agent.fuzzify(
    targets=[0],
    stat_types=['mean'],  # mean, std, cv
    method='gaussian',  # gaussian, gaussian_centroid, trapeze
    gfactor=0.2,  # spread parameter used in gaussian or trapeze function
    weighting=None,  # None, pca, speciation
    norm=False,
    obj_range=2000)

agent.plot_target_fuzzy_clusters(
    show_mode='images',  # images, features
    out_n=20,  # number of objects shown (closest ones to target)
    layer='conv11',  # if show mode is 'features', features of this layer showed
    layer_feat_id=0,
    title_type='probability',  # object, probability
    show=True,
    save=False)

# -----------------
# Explore clusters signals + get their weights
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

# check percentage of feature whose act. values within given interval
agent.get_signal_percentage(sig_interval=[0.2, 0.4])

agent.get_signal_per_layer()

# calculate percentage of total signal change per layer
agent.plot_signal_percentage(
    mode='histogram',  # histogram, cumsum
    save=False)
