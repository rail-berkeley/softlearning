import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers

from softlearning.utils.keras import PicklableSequential
from softlearning.models.normalization import (
    LayerNormalization,
    GroupNormalization,
    InstanceNormalization)
from softlearning.utils.tensorflow import nest


tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions
tfb = tfp.bijectors


def convnet_model(
        conv_filters=(64, 64, 64),
        conv_kernel_sizes=(3, 3, 3),
        conv_strides=(2, 2, 2),
        padding="SAME",
        normalization_type=None,
        normalization_kwargs={},
        downsampling_type='conv',
        activation=layers.LeakyReLU,
        name="convnet",
        *args,
        **kwargs):
    normalization_layer = {
        'batch': layers.BatchNormalization,
        'layer': LayerNormalization,
        'group': GroupNormalization,
        'instance': InstanceNormalization,
        None: None,
    }[normalization_type]

    def conv_block(conv_filter, conv_kernel_size, conv_stride):
        block_parts = [
            layers.Conv2D(
                filters=conv_filter,
                kernel_size=conv_kernel_size,
                strides=(conv_stride if downsampling_type == 'conv' else 1),
                padding=padding,
                activation='linear',
                *args,
                **kwargs),
        ]

        if normalization_layer is not None:
            block_parts += [normalization_layer(**normalization_kwargs)]

        block_parts += [(layers.Activation(activation)
                         if isinstance(activation, str)
                         else activation())]

        if downsampling_type == 'pool' and conv_stride > 1:
            block_parts += [getattr(layers, 'AvgPool2D')(
                pool_size=conv_stride, strides=conv_stride)]

        block = tfk.Sequential(block_parts, name='conv_block')
        return block

    def preprocess(x):
        """Cast to float, normalize, and concatenate images along last axis."""
        x = nest.map_structure(
            lambda image: tf.image.convert_image_dtype(image, tf.float32), x)
        x = nest.flatten(x)
        x = tf.concat(x, axis=-1)
        x = (tf.image.convert_image_dtype(x, tf.float32) - 0.5) * 2.0
        return x

    model = PicklableSequential((
        tfkl.Lambda(preprocess),
        *[
            conv_block(conv_filter, conv_kernel_size, conv_stride)
            for (conv_filter, conv_kernel_size, conv_stride) in
            zip(conv_filters, conv_kernel_sizes, conv_strides)
        ],
        tfkl.Flatten(),

    ), name=name)

    return model
