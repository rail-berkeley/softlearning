import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.engine import training_utils

from softlearning.utils.keras import PicklableKerasModel
from softlearning.models.normalization import (
    LayerNormalization,
    GroupNormalization,
    InstanceNormalization)


def convnet_model(
        inputs,
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

    float_inputs = tf.keras.layers.Lambda(
        lambda inputs: tf.cast(inputs, tf.float32)
    )(inputs)

    concatenated = tf.keras.layers.Lambda(
        lambda inputs: tf.concat(inputs, axis=-1)
    )(float_inputs)

    x = concatenated
    for (conv_filter, conv_kernel_size, conv_stride) in zip(
            conv_filters, conv_kernel_sizes, conv_strides):
        x = layers.Conv2D(
            filters=conv_filter,
            kernel_size=conv_kernel_size,
            strides=(conv_stride if downsampling_type == 'conv' else 1),
            padding=padding,
            activation='linear',
            *args,
            **kwargs
        )(x)

        if normalization_type == 'batch':
            x = layers.BatchNormalization(**normalization_kwargs)(x)
        elif normalization_type == 'layer':
            x = LayerNormalization(**normalization_kwargs)(x)
        elif normalization_type == 'group':
            x = GroupNormalization(**normalization_kwargs)(x)
        elif normalization_type == 'instance':
            x = InstanceNormalization(**normalization_kwargs)(x)
        elif normalization_type == 'weight':
            raise NotImplementedError(normalization_type)
        else:
            assert normalization_type is None, normalization_type

        x = (layers.Activation(activation)(x)
             if isinstance(activation, str)
             else activation()(x))

        if downsampling_type == 'pool' and conv_stride > 1:
            x = getattr(layers, 'AvgPool2D')(
                pool_size=conv_stride, strides=conv_stride
            )(x)

    flattened = layers.Flatten()(x)

    model = PicklableKerasModel(inputs, flattened, name=name)

    return model
