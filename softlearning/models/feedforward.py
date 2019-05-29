import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.keras.engine import training_utils

from softlearning.utils.keras import PicklableSequential

from distutils.version import LooseVersion
if LooseVersion(tf.__version__) > LooseVersion("2.00"):
    from tensorflow import nest
else:
    from tensorflow.contrib.framework import nest


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
    model = PicklableSequential((
        tfkl.Lambda(lambda x: tf.concat(
            nest.flatten(nest.map_structure(
                training_utils.cast_if_floating_dtype, x)),
            axis=-1)),
        *[
            tf.keras.layers.Dense(
                hidden_layer_size, *args, activation=activation, **kwargs)
            for hidden_layer_size in hidden_layer_sizes
        ],
        tf.keras.layers.Dense(
            output_size, *args, activation=output_activation, **kwargs)
    ), name=name)

    return model
