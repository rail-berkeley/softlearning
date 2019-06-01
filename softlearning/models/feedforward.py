import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.keras.engine import training_utils

from softlearning.utils.keras import PicklableSequential
from softlearning.utils.tensorflow import nest


tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions
tfb = tfp.bijectors


def feedforward_model(hidden_layer_sizes,
                      output_size,
                      activation='relu',
                      output_activation='linear',
                      preprocessors=None,
                      name='feedforward_model',
                      *args,
                      **kwargs):
    def cast_and_concat(x):
        x = nest.map_structure(training_utils.cast_if_floating_dtype, x)
        x = nest.flatten(x)
        x = tf.concat(x, axis=-1)
        return x

    model = PicklableSequential((
        tfkl.Lambda(cast_and_concat),
        *[
            tf.keras.layers.Dense(
                hidden_layer_size, *args, activation=activation, **kwargs)
            for hidden_layer_size in hidden_layer_sizes
        ],
        tf.keras.layers.Dense(
            output_size, *args, activation=output_activation, **kwargs)
    ), name=name)

    return model
