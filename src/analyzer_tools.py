import cv2
import utils
import numpy as np
import tensorflow as tf

from scipy import ndimage
from PIL import Image
from matplotlib import cm


def get_binary_map(x, blur_coef, binary_thresh=124):
    """Returns the binary image of segmentation x.

    Args:
        x: segmentation output from network.
        blur_coef (tuple): gaussian blur coefficient used prior to
            thresholding image.
        binary_thresh (int): threshold value for image processing of predicted
            object within the 0-255 range.
    """
    # Convert normalized image with gradient mapping.
    im = np.uint8(cm.viridis(x) * 255)

    # Get image (with 3 bands) from array.
    im = Image.fromarray(im)
    rgb_map = utils.remove_transparency(im, (255, 255, 255))
    rgb_map = np.array(rgb_map)

    # Transform RGB image in gray level.
    grayM = cv2.cvtColor(rgb_map, cv2.COLOR_BGR2GRAY)

    # Apply thresholding.
    blur = cv2.GaussianBlur(grayM, blur_coef, 0)
    ret, binary_map = cv2.threshold(blur, binary_thresh, 255, 0)

    return binary_map


def get_activations(x, layer, sess, feed_x, softmax=False):
    """Returns the activated <layer>, given the <x>.

    Args:
        x: image x to network.
        layer: convolution layer considered.
        sess: tensorflow session.
        feed_x: placeholder for x samples.
        softmax (bool): if True, apply softmax (boolean)
    """
    if softmax is True:
        layer = tf.nn.softmax(layer)
    return sess.run(layer, feed_dict={feed_x: x})


def get_representation(sample, layer, op_name, sess, feed_x, softmax=False):
    """Returns the layer activations of given layer (representation).

    Args:
        sample: image x to network.
        layer: convolution layer considered.
        op_name (str): layer operation of activation type.
        sess: tensorflow session.
        feed_x: placeholder for x samples.
        softmax (bool): if True, apply softmax (boolean)
    """
    ly = tf.compat.v1.get_default_graph().get_tensor_by_name(layer + op_name)
    reprs = get_activations(
        layer=ly,
        x=np.expand_dims(sample, axis=0),
        sess=sess,
        feed_x=feed_x,
        softmax=softmax)
    return reprs[0, :, :, :]


def get_layer_features(layer, layer_ops, sess, feed_x, sample):
    """Returns feature maps of layer.

    Args:
        layer: convolution layer considered.
        layer_ops (list): list of operation names from convolution
            layer.
        sess: tensorflow session.
        feed_x: placeholder for x samples.
        sample: image x to network.
    """
    # reference layer to extract feature activations at object level
    feat_maps = None
    error = None
    for op in layer_ops:
        try:
            # try the function given the operation name
            # some layers do not use same operations (e.g. Relu, mul...)
            # the list of activations should be defined in self.layer_ops
            feat_maps = get_representation(
                sample=sample,
                layer=layer,
                op_name=op,
                sess=sess,
                feed_x=feed_x,
                softmax=False)
            break
        except Exception as err:
            error = err
            continue

    if feat_maps is None:
        raise Exception(f"No feature maps due to error '{error}'.")

    return feat_maps


def get_object_mask(dim_ratio, binary):
    """Get the binary mask of cropped object given the dimension of feature
    layers called.

    Args:
        dim_ratio (float): dimension ratio between the segmentation or
            sample size and the network feature size. About 'dim_ratio',
            check the dimension difference of segmentation map and analyzed
            feature map. Note: backward feature maps can be of size e.g.
            32x32, 64x64, 128x128 given the conv. layer so we need to
            re-adjust the backward cropping window to extract a feature at
            correct position.
        binary: binarized segmentation.
    """
    # Resize mask given the dimension ratio.
    if dim_ratio != 1:
        shape = int(binary.shape[0] / dim_ratio)  # corrected dimension
        binary = utils.resize_array(binary, shape)

    # Remove any annotation added around masked object while
    # extending the margins i.e., eventual objects nearby that
    # are segmented. There must be only 1 main object.
    objects, number_of_objects = ndimage.label(binary)  # clusters of separated segments (objects)
    if number_of_objects > 1:
        # If more than 2 objects, remove extra ones nearby.
        obj_ids = np.unique(objects)[1:]  # get unique values of matrix (ruling out 0, which is the background)
        obj_sizes = [np.sum(objects[objects == i]) for i in obj_ids]  # get total pixels covered by objects
        obj_main = np.argmax(obj_sizes) + 1  # get index of value mainly represented

        for i in obj_ids:
            # Convert less represented value into 0s, and the most represented
            # value into 1s.
            objects[objects == i] = 0 if (i != obj_main) else 255

        # Current cluster become the converted objects.
        binary = objects

    # Collect indices of pixels classified True (i.e. related to the main
    # object, not background).
    mask_indices = np.where(binary != 0)

    return binary, mask_indices


def get_object_feature(layer_arr, layer_feat_id, mask, mask_indices, feat_type):
    """Returns feature matrix and values of cropped object.

    Args:
        layer_arr (): x layer feature,
        layer_feat_id (int): dimension considered from the
            feature layer.
        mask: binarized segmentation map.
        mask_indices: indices of segmentation mao part of
            object mask.
        feat_type (str): the way feature values are collected from feature maps.
            If 'box' chosen, it returns the entire image feature, while 'object'
            allows returning feature values of the segmented object only (in the
            image), and 'positives' allows returning only positive values from
            the image (valid for Relu or Sigmoid activations).
    """
    obj_feat = layer_arr[:, :, layer_feat_id]

    # Extract feature values where mask is True.
    obj_feat_values = obj_feat[mask_indices[0], mask_indices[1]]

    # Rule out feature values not related to segmented object.
    obj_feat[mask < 1] = 0

    if feat_type == 'box':
        # Consider the entire image containing the object.
        return obj_feat
    else:
        # Consider the object (masked) only.
        return obj_feat_values


def get_desc_statistics(obj_feat, eps=1e-10):
    """Returns statistics of feature matrices extracted from network.

    Args:
        obj_feat: feature matrix.
        eps (float): epsilon value to avoid division by 0.
    """
    mean, stddev = np.nan_to_num(np.mean(obj_feat)), np.nan_to_num(np.std(obj_feat))
    coef_var = stddev / (mean + eps)
    return mean, stddev, coef_var  # mean, standard deviation, coefficient of variation
