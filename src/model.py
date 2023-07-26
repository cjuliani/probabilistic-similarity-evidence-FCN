import tensorflow as tf

tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()


def conv2d(x, filters, kernel_size, stride, padding, name, dilation=(1, 1),
           act=tf.nn.relu, batch_norm=False):
    """Returns 2D convolutional operation."""
    reg_fact = 0.
    layer = tf.keras.layers.Conv2D(
        filters, kernel_size,
        strides=stride,
        padding=padding,
        activation=act,
        kernel_initializer=tf.keras.initializers.VarianceScaling(),
        bias_initializer=tf.constant_initializer(value=0.1),
        kernel_regularizer=tf.keras.regularizers.l1(l=reg_fact),
        bias_regularizer=tf.keras.regularizers.l1(l=reg_fact),
        activity_regularizer=tf.keras.regularizers.l1(l=reg_fact),
        dilation_rate=dilation,
        name='conv' + name)(x)

    if batch_norm:
        # Apply batch normalization.
        batch_normed = tf.keras.layers.BatchNormalization(
            axis=-1,
            momentum=0.99,
            epsilon=1e-3,
            center=True,
            scale=True,
            beta_initializer='zeros',
            gamma_initializer='ones',
            moving_mean_initializer='zeros',
            moving_variance_initializer='ones',
            trainable=True,
            name='bn' + name)
        layer = batch_normed(layer, training=True)

        # TensorFlow continually estimates the mean and variance of the
        # weights over the training dataset. These are then stored in
        # the tf.GraphKeys.UPDATE_OPS variable.
        tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, batch_normed.moving_mean)
        tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, batch_normed.moving_variance)

    return layer


def max_pool(x, kernel_size, stride, padding, name):
    """Returns max pooling operation."""
    return tf.keras.layers.MaxPooling2D(
        kernel_size,
        stride,
        padding=padding,
        name=name)(x)


def avg_pool(x, kernel_size, stride, padding, name):
    """Returns average pooling operation."""
    return tf.keras.layers.AveragePooling2D(
        pool_size=kernel_size,
        strides=stride,
        padding=padding,
        name=name)(x)


def deconv2d(x, size, name):
    """Returns deconvoluted 2D input."""
    return tf.keras.layers.UpSampling2D(
        size=(size, size),
        name=name)(x)


def unet(x, n_class, f=32, krl=3, pad="same", batch_norm=False):
    """Returns the UNet network logits.

    Args:
        x: x image arrays.
        n_class (int): number of classes considered.
        f (int): initial number of filters.
        krl (int): kernel size of convolution layers
        pad (str): 'same' or 'valid'.
        batch_norm (bool): if True, apply batch normalization.
    """
    # encoder
    conv11 = conv2d(x, f, krl, 1, pad, '11', batch_norm=batch_norm)
    conv12 = conv2d(conv11, f, krl, 1, pad, '12', batch_norm=batch_norm)

    pool1 = max_pool(conv12, (2, 2), (2, 2), pad, 'pool1')

    conv21 = conv2d(pool1, f * 2, krl, 1, pad, '21', batch_norm=batch_norm)
    conv22 = conv2d(conv21, f * 2, krl, 1, pad, '22', batch_norm=batch_norm)

    pool2 = max_pool(conv22, (2, 2), (2, 2), pad, 'pool2')

    conv31 = conv2d(pool2, f * 3, krl, 1, pad, '31', batch_norm=batch_norm)
    conv32 = conv2d(conv31, f * 3, krl, 1, pad, '32', batch_norm=batch_norm)

    pool3 = max_pool(conv32, (2, 2), (2, 2), pad, 'pool3')

    conv41 = conv2d(pool3, f * 4, krl, 1, pad, '41', batch_norm=batch_norm)
    conv42 = conv2d(conv41, f * 4, krl, 1, pad, '42', batch_norm=batch_norm)

    # bottleneck
    pool4 = max_pool(conv42, (2, 2), (2, 2), pad, 'pool4')

    conv51 = conv2d(pool4, f * 5, krl, 1, pad, '51', batch_norm=batch_norm)
    conv52 = conv2d(conv51, f * 5, krl, 1, pad, '52', batch_norm=batch_norm)

    # decoder
    deconv1 = deconv2d(conv52, 2, 'deconv1')
    merge11 = tf.concat(values=[conv42, deconv1], axis=-1, name='merge11')

    conv61 = conv2d(merge11, f * 4, krl, 1, pad, '61', batch_norm=batch_norm)
    conv62 = conv2d(conv61, f * 4, krl, 1, pad, '62', batch_norm=batch_norm)

    deconv2 = deconv2d(conv62, 2, 'deconv2')
    merge12 = tf.concat(values=[conv32, deconv2], axis=-1, name='merge12')

    conv71 = conv2d(merge12, f * 3, krl, 1, pad, '71', batch_norm=batch_norm)
    conv72 = conv2d(conv71, f * 3, krl, 1, pad, '72', batch_norm=batch_norm)

    deconv3 = deconv2d(conv72, 2, 'deconv3')
    merge13 = tf.concat(values=[conv22, deconv3], axis=-1, name='merge13')

    conv81 = conv2d(merge13, f * 2, krl, 1, pad, '81', batch_norm=batch_norm)
    conv82 = conv2d(conv81, f * 2, krl, 1, pad, '82', batch_norm=batch_norm)

    deconv4 = deconv2d(conv82, 2, 'deconv4')
    merge14 = tf.concat(values=[conv12, deconv4], axis=-1, name='merge14')

    conv91 = conv2d(merge14, f, krl, 1, pad, '91', batch_norm=batch_norm)
    conv92 = conv2d(conv91, f, krl, 1, pad, '92', batch_norm=batch_norm)

    # head
    layers_out = []
    for clss in range(n_class):
        name = '10{}'.format(clss)
        conv_ = conv2d(
            x=conv92,
            filters=2,
            kernel_size=1,
            stride=1,
            padding=pad,
            name=name,
            act=None,
            batch_norm=batch_norm)
        layers_out.append(conv_)

    return layers_out
