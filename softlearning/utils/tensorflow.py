from distutils.version import LooseVersion

import tensorflow as tf


if LooseVersion(tf.__version__) > LooseVersion("2.00"):
    # We import tf.python.util.nest because the public tf.nest
    # doesn't expose `map_structure_with_paths` method.
    # TODO(hartikainen): Figure out a way to use regular tf.nest.
    from tensorflow.python.util import nest
else:
    from tensorflow.contrib.framework import nest


def set_gpu_memory_growth(growth):
    physical_gpus = tf.config.experimental.list_physical_devices('GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(f"{len(physical_gpus)} Physical GPUs, "
          f"{len(logical_gpus)} Logical GPUs.")

    for physical_gpu in physical_gpus:
        print(f"GPU: {physical_gpu}")
        try:
            tf.config.experimental.set_memory_growth(gpu, growth)
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def apply_preprocessors(preprocessors, inputs):
    nest.assert_same_structure(inputs, preprocessors)
    preprocessed_inputs = nest.map_structure(
        lambda preprocessor, input_: (
            preprocessor(input_) if preprocessor is not None else input_),
        preprocessors,
        inputs,
    )

    return preprocessed_inputs


def cast_and_concat(x):
    x = nest.map_structure(
        lambda element: tf.cast(element, tf.float32), x)
    x = nest.flatten(x)
    x = tf.concat(x, axis=-1)
    return x
