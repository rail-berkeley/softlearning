import tensorflow as tf
import tensorflow_probability as tfp

from softlearning.utils.tensorflow import cast_and_concat


tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions
tfb = tfp.bijectors


def feedforward_model(hidden_layer_sizes,
                      output_shape,
                      activation='relu',
                      output_activation='linear',
                      preprocessors=None,
                      name='feedforward_model',
                      *args,
                      **kwargs):
    output_size = tf.reduce_prod(output_shape)
    if 1 < len(output_shape):
        raise NotImplementedError("TODO(hartikainen)")
    model = tf.keras.Sequential((
        tfkl.Lambda(cast_and_concat),
        *[
            tf.keras.layers.Dense(
                hidden_layer_size, *args, activation=activation, **kwargs)
            for hidden_layer_size in hidden_layer_sizes
        ],
        tf.keras.layers.Dense(
            output_size, *args, activation=output_activation, **kwargs),
        # tf.keras.layers.Reshape(output_shape),
    ), name=name)

    return model
