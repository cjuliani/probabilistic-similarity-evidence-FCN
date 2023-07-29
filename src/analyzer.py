import os
import config
import time
import random
import numpy as np
import tensorflow as tf
import PIL.ImageOps

from src import fuzzy as fuzz
from src import model
from src import utils
from src import analyzer_tools as ext
from src import signals as da

from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.axes._axes import _log as matplotlib_axes_logger
from src.utils import clear_txt_file
from src.pruning import pruning_by_CV, pruning_by_PCA

matplotlib_axes_logger.setLevel('ERROR')
tf.compat.v1.disable_v2_behavior()


class ConvAgent:
    """
    Module extracting network information using the mask of the class 
    considered. It stores image and feature information within dictionaries.

    Available methods allow to apply a learning model, collect and analyze
    network features, prune features, apply similarity analysis via fuzzy
    inference, and plot feature signals (statistical values).

    Attributes:
        all_skipped (list): indices of data skipped from dataset during processing
            due to processing issues.
        diff_sum: signal difference (feature mean vectors) calculated
            between clusters of images.
        diff_idx (list): indices of features considered in the calculation of
            signal difference.
        sorted_probs: fuzzy probabilities sorted in decreasing order.
        sorted_ids: indices of objects (images) after applying the fuzzy inference method.
        fuzzy_targets (list): indices of target objects (images) used as reference
            for similarity analysis (fuzzy inference calculation).
        fuzzy_gfactor (float): spread parameter of the fuzzy membership function.
        fuzzy_method (str): the fuzzy membership function, i.e. gaussian, gaussian_centroid,
            or trapeze. Note that if 2 or more targets are considered, the gaussian
            method will account only one centroid, i.e. an averaged feature vector
            between the targets, while the gaussian_centroid method will account
            one centroid per target.
        pruning_weights (dict): weights calculated while pruning the network features
            via 'pca' or coefficient of variation ('cv').
        current_coef (float): maximum ratio of network information to keep while pruning
            (value within range 0-1).
        pruning_feat_idxs (dict): indices of features kept after pruning.
        current_cls (str): class considered for segmentation mask. The mask is used to
            extract information of feature maps related to the object analyzed
            in similarity analysis.
        pruning_layers_limits: indices defining the groups of features considered
            in convolution layers analyzed.
        nonpruned_n (int): number of network features kept after pruning (not pruned).
        img_size (int): size of x to network.
        predict_path (str): folder path to save network predictions.
        obj_path (str): folder path to save object-related analytical results.
        obj_graph_path (str): folder path for any graphical analysis.
        obj_fuzzy_path (str): folder path for fuzzy inference results.
        out_stats (str): folder path to save feature statistics.
        conv_layers (list): convolutional layers considered in the analysis.
        stat_types (list): stat_types of statistics considered in the analysis.
        obj_imgs (list): list of x image arrays.
        feature_dict (dict): dictionary of network features collected per x images.
        obj_feat_statistics (dict): dictionary of feature statistics.
    """

    def __init__(self, conv_layers):
        self.all_skipped = None
        self.diff_sum = None
        self.diff_idx = None
        self.sorted_probs = None
        self.sorted_ids = None
        self.fuzzy_targets = None
        self.fuzzy_gfactor = None
        self.fuzzy_method = None
        self.pruning_weights = None
        self.current_coef = None
        self.pruning_feat_idxs = None
        self.current_cls = None
        self.pruning_layers_limits = None
        self.nonpruned_n = None
        self.obj_imgs = None
        self.feature_dict = None
        self.img_size = config.IMG_SIZE
        self.img_size = config.IMG_SIZE
        self.stat_types = config.STAT_TYPES

        # Define folder paths for results.
        self.predict_path = config.PREDICT_PATH
        self.obj_path = config.OBJ_PATH
        obj_attr_suffix = self.obj_path + config.OBJ_ATTR_SUFFIX
        self.obj_graph_path = self.obj_path + config.OBJ_GRAPH_SUFFIX
        self.obj_fuzzy_path = self.obj_path + config.OBJ_FUZZY_SUFFIX
        self.out_stats = os.path.join(os.path.abspath(obj_attr_suffix + config.OUT_STAT_SUFFIX), '')

        # Set convolutional layers whose features will be used
        # in similarity analysis.
        self.conv_layers = conv_layers

        # Get image data.
        self.img_names, self.data_dirs = utils.load_train(path=config.IMG_PATH)

        # Set up placeholders.
        self.x = tf.compat.v1.placeholder(
            tf.float32,
            shape=[1, self.img_size, self.img_size, 3],
            name='x')

        # Get network logits.
        logits = model.unet(
            x=self.x,
            n_class=config.CLASSES,
            batch_norm=config.BATCH_NORM)

        # Get session and restore model parameters.
        tf_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True
        tf_config.gpu_options.per_process_gpu_memory_fraction = config.GPU_MEM_FRACTION
        self.sess = tf.compat.v1.Session(config=tf_config)

        # Set up variables of network.
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.sess.run(tf.compat.v1.local_variables_initializer())

        saver = tf.compat.v1.train.Saver()
        saver.restore(self.sess, os.path.join(config.MODEL_PATH, config.MODEL))

        # Start session.
        tmp_sample = utils.get_array_from_image(
            img=self.img_names[0],
            img_size=self.img_size,
            normalize=True)

        _ = self.sess.run(
            logits[0],
            feed_dict={
                self.x: np.expand_dims(tmp_sample, 0)
            })
        print('Session started.')

        # Define dictionaries of features.
        self.obj_feat_statistics = {}
        for cls in config.CONV_SEGMENTS:
            self.obj_feat_statistics[cls] = {}

        print("\t▹ Data folders: ", self.data_dirs)
        print("\t▹ Total x: ", len(self.img_names))
        print('\t▹ Shape of x:', tmp_sample.shape)
        print('\t▹ Segments:', ', '.join(config.CONV_SEGMENTS))

    def collect_image_arrays(self):
        """Collects images to analyse."""
        self.obj_imgs = []
        for i, name in enumerate(self.img_names):
            tmp = utils.get_array_from_image(name, self.img_size, True)
            self.obj_imgs.append(tmp)
            print(f"processed (collecting arrays): {i + 1}/{len(self.img_names)}", end='\r')

        print(f"{len(self.img_names)} object images collected.")

    def reset_obj_feat_statistics(self, cls):
        """Resets features statistics of given class."""
        for lyr in self.conv_layers:
            self.obj_feat_statistics[cls][lyr] = {}

    def save_feat_statistics(self, classes, replace_stats=True):
        """Save properties and features of detected objects.

        Args:
            classes (list): classes of segmentations considered.
            replace_stats (bool): if True replace existing attributes
                previously saved.
        """
        print('\n∎ Saving objects attributes')

        for cls in classes:
            print('\t▹ class <{}>'.format(cls))

            tmp_dirpath = self.out_stats + cls + '/'
            if replace_stats is True:
                utils.check_directory_and_replace(tmp_dirpath, show=False)

            try:
                for layer in self.conv_layers:
                    for stat_type in self.stat_types:
                        # Create a folder for each statistical type.
                        tmp_dirpath_type = tmp_dirpath + stat_type + '/'
                        utils.check_directory(tmp_dirpath_type, show=False)

                        # Save objects features.
                        np.save(tmp_dirpath_type + layer, self.obj_feat_statistics[cls][layer][stat_type])

                print('\t\tStatistics saved.\n')

            except (AttributeError, KeyError, IndexError) as err:
                print('\t\tStatistics not saved, because: {}.\n'.format(err))

    def load_feat_statistics(self, classes):
        """Loads previously generated features statistics.

        Args:
            classes (list): classes of segmentations considered.
        """
        for cls in classes:
            print('\t▹ class <{}>'.format(cls))

            self.reset_obj_feat_statistics(cls)

            for layer in self.conv_layers:
                for stat_type in self.stat_types:
                    # Create a folder for each type of statistics.
                    tmp_dirpath_type = self.out_stats + cls + '/' + stat_type + '/'

                    # Save objects features.
                    self.obj_feat_statistics[cls][layer][stat_type] = np.load(
                        file=tmp_dirpath_type + layer + '.npy',
                        allow_pickle=True)

        print(f'Feature statistics of {len(self.conv_layers)} layers loaded.')

    def check_channel_sparsity(self, cls):
        """Returns the number of non-zeroes (active) features per layer.

        Args:
            cls (str): class of segmentations considered.
        """
        print('∎ Checking layers sparsity:')

        tot_nonzeros, tot_fts = 0, 0
        for layer in self.conv_layers:
            tmp = self.obj_feat_statistics[cls][layer]['mean']  # get layer statistics
            tmp = np.sum(tmp, axis=0)  # sum over object

            nonzeros = np.count_nonzero(tmp)  # number of active features
            fts = len(tmp)  # total number of features

            print("\t▹ {}: {} active features (out of {})".format(layer, nonzeros, fts))
            tot_nonzeros += nonzeros
            tot_fts += fts

        pct = ((tot_nonzeros / tot_fts) * 100)
        print(f'Total actives: {tot_nonzeros} (out of {tot_fts}, representing {pct}%)')

    def apply_pruning(self, mode, cls, cv_condition, coef):
        """Apply layer-per-layer pruning of convolutional outputs given
        the variance of activated features.

        Args:
            mode (str): pruning type, i.e. 'pca' or coefficient of variation 'cv'.
            cls (str): class of segmentations considered.
            cv_condition (str): condition for which features with <cv> values are
                kept 'above' or 'below' a threshold 'coef'.
            coef (float): the ratio of information kept from network features. If
                pruning is done with 'pca', it corresponds to the variance coefficient
                defining the maximum information (variability) to keep per layer.
                Otherwise, if 'cv' is chosen, it corresponds to the threshold value
                above (or below) which features are kept.
        """
        msg = "You must define <pruning_mode> as <pca> or <cv>."
        assert (mode == 'pca') or (mode == 'cv'), msg

        # Make sure object folder path exists.
        utils.check_directory(self.obj_path)

        # Set parameters used in next functions when processing
        # non-pruned features.
        self.current_cls, self.current_coef = cls, coef

        # Define dictionaries for pruning results.
        self.pruning_feat_idxs, self.pruning_weights = {cls: {}}, {cls: {}}
        self.nonpruned_n = {cls: {}}

        # Define weights given pruning mode.
        self.pruning_weights[cls][mode] = {}

        # -----
        print(f"∎ Pruning features of {len(self.conv_layers)} layers")

        # Clear existing text file where we record pruning results.
        file_name = clear_txt_file(
            path=self.obj_path,
            cls=cls,
            name=mode + '_pruning_results')

        with open(file_name, 'a') as file_a:
            file_a.write('Variance coefficient: {}'.format(coef) + '\n')

        # Apply pruning per layer.
        tot_feat_kept, tot_feat = 0, 0
        for lyr in self.conv_layers:

            with open(file_name, 'a') as file_a:
                file_a.write('\nLayer {}\n'.format(lyr))

            # Store pruning results.
            # Note that 'pruning_feat_idxs' corresponds to the remaining
            # feature indices kept after pruning, sorted by order of
            # importance (PCA weights), i.e. 0 being the most important feature.
            # 'pruning_weights' are either PCA or CV weights.
            if mode == "pca":
                # Feature pruning by PCA.
                feat_tmp = self.obj_feat_statistics[cls][lyr]['mean'].T

                (self.pruning_feat_idxs[cls][lyr],
                 self.pruning_weights[cls][mode][lyr]) = pruning_by_PCA(
                    feat_mat=feat_tmp,
                    var_coef=coef)
            else:
                # Feature pruning by coefficient of variation (CV).
                feat_tmp = self.obj_feat_statistics[cls][lyr]['mean']

                (self.pruning_feat_idxs[cls][lyr],
                 self.pruning_weights[cls][mode][lyr]) = pruning_by_CV(
                    feat_mat=feat_tmp,
                    cv_thresh=coef,
                    condition=cv_condition,
                    file_name=file_name)

            # Counting of features.
            tot_feat_kept += len(self.pruning_feat_idxs[cls][lyr])
            tot_feat += self.obj_feat_statistics[cls][lyr]['mean'].shape[1]

            # -----
            msg = f'indices kept out of {feat_tmp.shape[0]}: {len(self.pruning_feat_idxs[cls][lyr])}'
            print(f'\t▹ {lyr}: ' + msg)

            # Write pruning results in separate file.
            with open(file_name, 'a') as file_a:
                file_a.write('- ' + msg + '\n')

        # Define features kept (not pruned).
        self.nonpruned_n[cls] = tot_feat_kept

        # Get layers limits given kept (non pruned) feature indices.
        layers_idxs = [self.pruning_feat_idxs[cls][key] for key in self.pruning_feat_idxs[cls].keys()]
        layer_limits = [0] + [len(lst) for lst in layers_idxs]
        self.pruning_layers_limits = np.cumsum(layer_limits)

        # -----
        txt = 'Total features remaining, out of {}: {}'.format(tot_feat, tot_feat_kept)
        with open(file_name, 'a') as file_a:
            file_a.write('\n' + txt + '\n')
        print(txt)

    def get_obj_features(self, cls, feat_type, feat_layers, input_range, cnt_start, batch):
        """Returns a dictionary of features collected per layer.

        Args:
            cls (str): class of segmentation considered (layer name).
            feat_type (str): the way feature values are collected from feature maps.
                If 'box' chosen, it returns the entire image feature, while 'object'
                allows returning feature values of the segmented object only (in the
                image), and 'positives' allows returning only positive values from
                the image (valid for Relu or Sigmoid activations).
            feat_layers (list): feature layers considered from network
            input_range (list or tuple): the range of dataset considered for
                collecting features from.
            cnt_start (int): the dataset starting index to collect features
                from given the batch.
            batch (int): number of data instances considered in the feature
                collection.
        """
        # Create dictionary of feature arrays to be stored.
        feat_dict = {}

        # Iterate feature extraction per image sample.
        k, cnt, avg_speed = 0, cnt_start, 0.
        skipped = []
        est_remaining_time = (batch * 1.) / 60
        for spl_i in range(*input_range):
            start = time.time()

            try:
                # Get image array.
                x_batch = self.obj_imgs[spl_i]
            except IndexError:
                continue

            # Make sure image array is correct.
            if (x_batch.ndim < 3) or (x_batch.shape[-1] != 3):
                msg = (f'Image {self.img_names[spl_i]} skipped from feature ' +
                       f'collection due to incorrect shape dimension.')
                print(msg)
                skipped.append(self.img_names[spl_i])
                continue

            # Collect tensors
            tensors = []
            for l in feat_layers:
                ly = tf.compat.v1.get_default_graph().get_tensor_by_name(l + config.OPERATIONS[0])
                tensors.append(ly)
            tensors += [tf.compat.v1.get_default_graph().get_tensor_by_name(cls + config.OBJ_OPERATION)]

            feat_maps = ext.get_representations_from_tensors(
                sample=x_batch,
                tensors=tensors,
                sess=self.sess,
                feed_x=self.x,
                softmax=False)

            x_segmentation = ext.softmax(feat_maps[-1])
            feat_maps = feat_maps[:-1]

            # Get binary map with main object of given class.
            binary = ext.get_binary_map(
                x=x_segmentation[:, :, 1],
                blur_coef=(3, 3),
                binary_thresh=124)

            for feat, layer in zip(feat_maps, self.conv_layers):
                # Get dimension ratio between network output segmentation
                # and current layer.
                dim_ratio = self.img_size / feat.shape[1]

                feat_dict[cnt] = feat_dict.get(cnt, {})
                feat_dict[cnt][layer] = feat_dict.get(layer, [])

                # Get object's mask and related mask indices.
                _, mask_indices = ext.get_object_mask(
                    dim_ratio=dim_ratio,
                    binary=binary)

                # Get object features of current layer given number of layer
                # outputs.
                # Extract feature values where mask is True.
                if feat_type == 'box':
                    # Consider the entire image containing the object.
                    feat_items = feat[mask_indices[0], mask_indices[1], :].T
                else:
                    # Consider the object (masked) only.
                    feat_items = feat.reshape([-1, feat.shape[-1]]).T

                # Convert features into array.
                feat_items = np.array(feat_items)
                if feat_items.shape[1] == 0:
                    feat_items = np.zeros(shape=(feat_items.shape[0], 1))
                feat_dict[cnt][layer] = feat_items

            # ----- define loop speed
            speed_val = time.time() - start
            speed = "({:.2f} sec)".format(time.time() - start)
            msg = (f'processed (building dictionary): images,' +
                   f' {k + 1}/{batch} ' +
                   speed + " (est. time left: {:.2f} mins)".format(est_remaining_time))
            print(msg, end="\r")

            # ----- define timing and increment
            est_remaining_time = (speed_val * (batch - k)) / 60  # in minutes
            cnt += 1
            k += 1

        return feat_dict, skipped

    def collect_features(self, cls, feat_type='object', batch=1000):
        """Returns a dictionary of network features.

        Args:
            cls (str): class of segmentation considered (layer name).
            feat_type (str): type of feature, i.e. 'box' to collect the entire
                image's feature values or 'object' to collect feature values
                related to the segmented object only.
            batch (int): number of data instances considered in the feature
                collection.
        """
        self.reset_obj_feat_statistics(cls)

        tmp = len(self.img_names) % batch
        steps = len(self.img_names) // batch

        if tmp > 0:
            steps += 1

        self.all_skipped = []
        for i in range(steps):
            print(f"step: {i + 1}/{steps}")

            self.feature_dict, skipped = self.get_obj_features(
                cls=cls,
                feat_layers=self.conv_layers,
                feat_type=feat_type,
                input_range=[i * batch, (i * batch) + batch],
                cnt_start=i * batch,
                batch=batch)
            self.all_skipped += skipped

            # Get feature statistics
            self.get_obj_feat_statistics(
                cls='conv100',
                input_range=[i * batch, (i * batch) + batch])

        # Convert list of statistics stored into arrays.
        for i, layer in enumerate(self.conv_layers):
            for j, stat_type in enumerate(self.stat_types):
                lst = self.obj_feat_statistics[cls][layer][stat_type]
                self.obj_feat_statistics[cls][layer][stat_type] = np.array(lst)

                # -----
                msg = (f'processed (array conversion): images {j + 1}/{len(self.stat_types)},' +
                       f' {i + 1}/{len(self.conv_layers)}')
                print(msg, end="\r")

        print(f"Statistics of {len(self.conv_layers)} layers features for {len(self.img_names)} inputs collected.")
        print('\t▹ stat_types considered: {}.'.format(', '.join(self.stat_types)))

        # Reformat images processed.
        self.img_names = [path for path in self.img_names if path not in self.all_skipped]

    def get_obj_feat_statistics(self, cls, input_range):
        """Collects statistics of non-pruned feature.

        Args:
            cls (str): class of segmentation considered (layer name).
            input_range (list or tuple): range of dataset considered.
        """
        # Calculate feature statistics per input and convolution
        batch_size = input_range[1] - input_range[0]

        # layer.
        j = 0
        for obj_i in range(*input_range):
            for i, layer in enumerate(self.conv_layers):
                # Get layer feature array.
                feat_arr = self.feature_dict[obj_i][layer]

                mean_, std_, cv_ = ext.get_desc_statistics(obj_feat=feat_arr)
                tmp_arr = np.vstack([mean_, std_, cv_]).T

                # Feed database per type of statistics.
                cnt = 0
                for stat_type in self.stat_types:
                    self.obj_feat_statistics[cls][layer][stat_type] = \
                        self.obj_feat_statistics[cls][layer].get(stat_type, []) + [tmp_arr[:, cnt]]  # (stats,)
                    cnt += 1

                # -----
                msg = (f'processed (statistics): images {j + 1}/{batch_size},' +
                       f' {i + 1}/{len(self.conv_layers)}')
                print(msg, end="\r")
            j += 1

    def fuzzify(self, targets, obj_range, stat_types, method, gfactor, weighting, norm, show_info=True, sorting=True):
        """Fuzzify features with respect to targets.

        Args:
            targets (list): target objetc(s) based on which features are fuzzified.
            obj_range (int): number of object to consider.
            stat_types (list): type of statistical description considered.
            method (str): type of membership function.
            gfactor (float): standard deviation factor for the gaussian membership
                function OR ratio for outer margins of trapeze.
            weighting (str or None): type of weighting for probabilistic similarity
                calculation.
            norm (bool): if True, normalize features (min-max) before fuzzification.
            show_info (bool): if True, show information relative to newly created
                attributes.
            sorting (bool): if True, sort the probabilistic similarity results.
        """
        msg = f"Type of statistics not understood. Choose between among {', '.join(self.stat_types)}."
        assert all(stat_type in self.stat_types for stat_type in stat_types) is True, msg

        # Set parameters used in next functions when processing
        # non-pruned features.
        self.fuzzy_targets = targets
        self.fuzzy_method = method
        self.fuzzy_gfactor = gfactor

        # Define weighting scheme.
        if weighting == 'pca':
            # Get 1st principal vector of feature for each layer, converted
            # into a vector-shape factor (optional) to be used as weights
            # when applying fuzzy inference.
            weights = [self.pruning_weights[self.current_cls]['pca'][lyr] for lyr in self.conv_layers]
            weights = np.array([abs(e) for sub in weights for e in sub])
        elif weighting == 'speciation':
            weights = self.diff_sum  # normalized sum of signal difference between clusters (manually defined)
        else:
            weights = np.array([1] * self.nonpruned_n[self.current_cls])  # no weights

        # Calculate pobabilistic similarities.
        _, self.sorted_probs, self.sorted_ids = fuzz.fuzzify(
            layers=self.conv_layers,
            feat_stats=self.obj_feat_statistics[self.current_cls],
            feat_idxs=self.pruning_feat_idxs[self.current_cls],
            obj_range=obj_range,
            stat_types=stat_types,
            method=method,
            gfactor=gfactor,
            targets_id=targets,
            weights=weights,
            norm=norm,
            sorting=sorting)

        if show_info is True:
            print(f"Features of {len(self.conv_layers)} layers fuzzified given target(s): {targets}.")
            if weighting != 'pca' and weighting != 'speciation':
                print("\t▹ No weighting scheme used. Choose 'pca' if pruning by pca method was applied,"
                      " or 'speciation' after calculating neurons speciation via 'get_cluster_signal_difference'.")
            print("\n(!) Probabilistic similarity is calculated per 'stat_types', then averaged and sorted.")

    @staticmethod
    def convert(value):
        """Converts integers of type np.int64 to normal int for saving in JSON.
        It seems like there may be a issue to dump into json string in Python 3."""
        if isinstance(value, np.int64) or isinstance(value, np.int32):
            return int(value)
        elif isinstance(value, np.float64) or isinstance(value, np.float32):
            return float(value)
        raise TypeError

    def get_cluster_signal_difference(self, stat_type, calc_mode, norm, norm_diff, cls_groups):
        """Calculates the difference of signal magnitude between clusters or items of clusters.
        Notes: groups represent multiple clusters, while a cluster is made of items.
        Groups allow categorizing (manually) similar clusters whose objects have visual
        equivalence. This is relevant for vizualization of magnitude differences. If no
        manual selection of groups is made, then results from automated analysis will be used,
        i.e. there will be 1 cluster per group only.

        Args:
            stat_type (str): statistics previously calculated, between 'mean', 'std' and 'cv'.
            calc_mode (str or None): 'per_cluster' for differentiating mean clusters' signals.
                Features are analyzed otherwise
            norm (bool): if True, normalize of statistics feature-wise.
            norm_diff (bool): if True, normalize the signal difference calculated per cluster.
            cls_groups (dict): groups of clusters of images considered.
        """
        msg = f"The stat type should be among: {', '.join(self.stat_types)}."
        assert (stat_type in self.stat_types) is True, msg

        # Make sure pruning has been applied.
        try:
            _ = self.pruning_feat_idxs[self.current_cls][self.conv_layers[0]]
        except (AttributeError, KeyError):
            Exception(f"You must prune features via PCA or CV before evaluating signal differences.")

        print('\n(!) Results for class <{}> and pruning coef. of {}.'.format(self.current_cls, self.current_coef))
        print('(!) If <mode> is <per_cluster>, the median signal of every cluster is compared to get the difference.')
        print('(!) Choose <manual> as <cluster_mode> for manual definition of clusters.')

        # Get groups of clusters.
        groups = [cls_groups[key] for key in cls_groups.keys()]
        clusters = [e for sub in groups for e in sub]  # flatten the groups of lists

        # Get signal differences based on 'stat_type' values obtained.
        fts_tmp = [self.get_nonpruned_features(lyr, stat_type, self.current_cls) for lyr in self.conv_layers]
        fts_tmp = np.hstack(fts_tmp)

        # Calculate signal differences. Note that the signal difference is
        # calculated between clusters (not groups) or items of clusters.
        self.diff_idx, self.diff_sum = da.get_sum_signal_difference(
            feats=fts_tmp,
            clusters=clusters,
            sorted_diff=False,
            mode=calc_mode,
            norm=norm)

        if norm_diff is True:
            # Standardize signal difference between 0-1 for comparison
            # between features.
            self.diff_sum = da.standardize_by_minmax(self.diff_sum)

    def get_signal_percentage(self, sig_interval):
        """Checks the percentage of feature whose activation values
        are within given interval

        Args:
            sig_interval (list): signal magnitude interval to check its
                percentage taken in network.
        """
        da.get_signal_percentage(
            signals=self.diff_sum,
            interval=sig_interval)

    def get_signal_per_layer(self):
        """Calculates percentage of total signal change per layer."""
        print(f"(!) Results for class '{self.current_cls}' and pruning coef. of {self.current_coef}.")
        da.get_signal_per_layer(
            layers=self.conv_layers,
            signals=self.diff_sum,
            intervals=[0] + self.pruning_layers_limits)

    def get_nonpruned_features(self, layer, stat_type, cls):
        """Returns features of a given layer kept after pruning.

        Args:
            layer (str): feature layer considered.
            stat_type (str): statistics previously calculated, between 'mean', 'std' and 'cv'.
            cls (str): class of segmentation considered (layer name).
        """
        msg = "Feature pruning must be applied first prior to using this method."
        assert hasattr(self, "pruning_feat_idxs") is True, msg

        kept_idxs = self.pruning_feat_idxs[cls][layer]
        return self.obj_feat_statistics[cls][layer][stat_type][:, kept_idxs]

    def plot_feature_mean_signal_distribution(self, cls, feature, lyr, incr, stat_type, show, save):
        """Displays distribution of a feature mean activation in the form of histogram
        (given incremented interval 'incr').

        Args:
            cls (str): classe of segmentation (object)
            feature (int): feature name.
            lyr (str): convolutional layer considered.
            incr (float): histogram bar increment/interval.
            stat_type (str): statistics previously calculated, between 'mean', 'std' and 'cv'.
            show (bool): if True, show the plot.
            save (bool): if True, save the plot.
        """
        # Get feature array.
        arr = self.obj_feat_statistics[cls][lyr][stat_type]
        x = list(arr[:, feature])

        # Plot histogram.
        plt.hist(x, bins=np.arange(np.min(x), np.max(x) + incr, incr))
        plt.grid(linestyle=':')

        if save is True:
            # Make sure folder path for graphs exists.
            utils.check_directory(self.obj_graph_path)

            img_name = 'hist_cls-{}_layer-{}_feature-{}'.format(cls, lyr, feature)
            output_name = os.path.join(os.path.abspath(self.obj_graph_path), img_name)
            plt.savefig(output_name)

            print('\nobj_cls-{}_layer-{}_ft-{} saved.'.format(cls, lyr, feature))

        if show is True:
            plt.show()

    def plot_group_signals(self, cls_groups, cls_groups_colors, grp_keys, stat_type,
                           sz, lw, alpha, mode, y_lim, show, save):
        """Displays single-valued activations signals per layer feature for
        each cluster group.

        Args:
            cls_groups (dict): dictionary of cluster groups.
            cls_groups_colors (list): colors considered per cluster group.
            grp_keys (list): cluster (group) keys considered.
            stat_type (str): single-valued statistics of network features ('mean',
                'std', 'cv').
            sz (list): scatter point size per cluster group.
            lw (float): line width of 'mean_curve'.
            alpha (float): alpha channel of scatter points.
            mode (str): 'mean_curve' of cluster groups, or 'scatter' to display
                statistical values per objects (segments).
            y_lim (list): plot limits on Y.
            show (bool): if True, show the plot.
            save (bool): if True, save the plot.
        """
        assert mode == "scatter" or mode == "mean_curve", "Mode should be <scatter> or <mean_curve>."
        msg = f"The stat type should be among: {', '.join(self.stat_types)}."
        assert any(stat_type in word for word in self.stat_types) is True, msg

        # Make sure folder path for graphs exists.
        utils.check_directory(self.obj_graph_path)

        # Make sure group keys are strings.
        grp_keys = [str(i) for i in grp_keys]

        # Assert that provided indices/keys are within the correct range.
        tmp_digits = [int(i) for i in list(cls_groups.keys())]
        available = set(tmp_digits)
        given = set([int(i) for i in grp_keys])
        not_available = list(given.difference(available))

        msg = f"You must choose <grp_keys> within the range [{min(tmp_digits)},{max(tmp_digits)}]."
        assert len(not_available) == 0, msg

        # Get not pruned features for given class and type of statistics.
        fts_tmp = [self.get_nonpruned_features(lyr, stat_type, self.current_cls) for lyr in self.conv_layers]
        fts_tmp = np.hstack(fts_tmp)

        # Plot scatter points.
        da.plot_group_signals(
            fts=fts_tmp,
            grp_keys=grp_keys,
            cls_groups=cls_groups,
            cls_colors=cls_groups_colors,
            indices=self.diff_idx,
            curves_lw=lw,
            N=self.nonpruned_n[self.current_cls],
            size=sz,
            alpha=alpha,
            mode=mode,
            fig_size=(30, 3),
            y_lim=y_lim,
            path=self.obj_graph_path,
            show=show,
            save=save)

    def plot_signals_histogram(self, cls_groups, width, color, alpha, grp_keys, stat_type, y_lim, show, save):
        """Displays histogram of the summed difference of activations signals
        calculated between cluster groups per layer.

        Args
            cls_groups (dict): dictionary of cluster groups.
            width (int): width of histogram bars.
            color (str): color of the histogram bars.
            alpha: alpha channel of scatter points.
            grp_keys (list): cluster (group) keys considered.
            stat_type (str): single-valued statistics of network features
                ('mean', 'std', 'cv').
            y_lim (list): plot limits on Y.
            show (bool): if True, show the plot.
            save (bool): if True, save the plot.
        """
        msg = f"The stat type should be among: {', '.join(self.stat_types)}."
        assert any(stat_type in word for word in self.stat_types) is True, msg

        # Make sure folder path for graphs exists.
        utils.check_directory(self.obj_graph_path)

        # Make sure group keys are strings.
        grp_keys = [str(i) for i in grp_keys]

        # Assert that provided indices/keys are within the correct range.
        tmp_digits = [int(i) for i in list(cls_groups.keys())]
        available = set(tmp_digits)
        given = set([int(i) for i in grp_keys])
        not_available = list(given.difference(available))

        msg = f"You must choose <grp_keys> within the range [{min(tmp_digits)}, {max(tmp_digits)}]."
        assert len(not_available) == 0, msg

        # Plot histograms.
        da.plot_group_signals_histogram(
            grp_keys=grp_keys,
            width=width,
            color=color,
            alpha=alpha,
            signals=self.diff_sum,
            N=self.nonpruned_n[self.current_cls],
            fig_size=(30, 3),
            y_lim=y_lim,
            path=self.obj_graph_path,
            show=show,
            save=save,
            tick_int=self.pruning_layers_limits)

    def plot_signal_percentage(self, mode, save):
        """Displays distribution of signal amplitudes in percentage.

        Args:
            mode (str): graph mode, i.e. 'histogram' or cumulative sum
                'cumsum'.
            save (bool): if True, save the plot
        """
        msg = "Choose appropriate mode between 'histogram' and 'cumsum'."
        assert mode == "histogram" or mode == "cumsum", msg

        # Make sure folder path for graphs exists.
        utils.check_directory(self.obj_graph_path)

        _ = da.plot_signal_percentage(
            signals=self.diff_sum,
            path=self.obj_graph_path,
            mode=mode,
            save=save)

    def plot_similarity(self, show_mode, out_n, intv, title_type, show, save):
        """Display probabilistic similarity results for given target(s).

        Args:
            show_mode (str): display 'features', or 'images' (by default).
            out_n (int): number of similar objects to display, in regard to
                the target(s).
            title_type (str): display 'probability' value, or the 'object'
                ID (by default).
            intv (int): number of items to plot in a row.
            show (bool): if True, show the plot.
            save (bool): if True, save the plot.
        """
        msg = "The show mode should be either 'features' or 'images'."
        assert show_mode == 'features' or show_mode == 'images', msg

        msg = ("The title type should be either 'probability' or 'object'" +
               " or an existing object property name.")
        assert (title_type in ['probability', 'object']) is True, msg

        # Make sure folder path for graphs exists.
        utils.check_directory(self.obj_fuzzy_path)

        # Set name of plot.
        tgt_txt = '-'.join(str(i) for i in self.fuzzy_targets)
        txt = (f"targets-{tgt_txt}_outn{out_n}_gfactor{self.fuzzy_gfactor}_layersn" +
               f"-{len(self.conv_layers)}_method-{self.fuzzy_method}")

        # Plot target image object and corresponding similar image
        # objects estimated via fuzzy inference.
        object_ids = fuzz.plot_images(
            prob_values=self.sorted_probs,
            indices=self.sorted_ids,
            N=out_n,
            x=self.obj_imgs,
            name=txt,
            path=self.obj_fuzzy_path,
            title=title_type,
            intv=intv,
            show=show,
            save=save)

        return object_ids

    def plot_weigths(self, cls, weight_type):
        """Displays the weight results from pruning.

        Args:
            cls (str): classe of segmentation (object).
            weight_type (str): weights calculated either from 'pca' or coefficient
                of variation 'cv'.
        """
        # Display vertical bars at limits.
        for limit in self.pruning_layers_limits:
            plt.axvline(
                x=limit,
                linewidth=1,
                ls='--',
                c='black')

        # Display coefficient of variations kept per layer.
        cnt = 0
        for key in self.pruning_weights[cls][weight_type].keys():
            tmp_start = self.pruning_layers_limits[cnt]
            nb = np.array(range(len(self.pruning_weights[cls][weight_type][key])))
            nb += tmp_start

            # Plot scatter points.
            plt.scatter(nb, self.pruning_weights[cls][weight_type][key], c='blue', s=1)
            cnt += 1

        plt.show()

    def predict(self, N, cls, cmap='jet', output=1):
        """
        Predicts instances from trained models.

        Args:
            N (int): number of randomly sampled x for predictions.
            cls (str): classe of segmentation (object)
            cmap (str): color gradient.
            output (int): softmax output index.
        """
        # Make sure the folder path to save predictions exists.
        utils.check_directory(self.predict_path, show=False)

        print('\n∎ Predicting objects')

        # Get colormap.
        cm_gradient = plt.get_cmap(cmap)

        # Randomize indices of input images to predict from.
        idx = list(range(len(self.img_names)))
        selected_indices = random.sample(idx, N)

        for spl_i in selected_indices:
            # Get parent folder name.
            sample_name = self.img_names[spl_i].split('\\')[-2]

            # Get segmentation from input image.
            x_batch = utils.get_array_from_image(self.img_names[spl_i], self.img_size, True)
            prob_map = ext.get_representation(
                sample=x_batch,
                layer=cls,
                op_name=config.OBJ_OPERATION,
                sess=self.sess,
                feed_x=self.x,
                softmax=True)

            # Convert probability map from feature layer into PIL image
            # object with color mapping.
            result = np.uint32(prob_map[:, :, output] * 255)
            result = cm_gradient(result)  # give it colors
            result = Image.fromarray(np.uint8(result * 255))  # make it 8 bits image

            # Save probability maps.
            img_name = sample_name + '_probability_class-{}_sample-{}.png'.format(cls, spl_i)
            output_name = os.path.join(self.predict_path, img_name)
            result.save(output_name)

            # Get segmentation.
            seg_map = np.argmax(prob_map, axis=-1)
            result = np.uint32(seg_map)
            result = Image.fromarray(np.uint8(result * 255))
            result = PIL.ImageOps.invert(result)

            # Save segmented (binary) maps..
            img_name = sample_name + '_segmentation_class-{}_sample-{}.png'.format(cls, spl_i)
            output_name = os.path.join(self.predict_path, img_name)
            result.save(output_name)

            print('\t▹ predictions of sample {} from {} saved.'.format(spl_i, sample_name))
