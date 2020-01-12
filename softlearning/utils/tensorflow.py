from distutils.version import LooseVersion

import tensorflow as tf


if LooseVersion(tf.__version__) > LooseVersion("2.00"):
    # We import tf.python.util.nest because the public tf.nest
    # doesn't expose `map_structure_with_paths` method.
    # TODO(hartikainen): Figure out a way to use regular tf.nest.
    from tensorflow.python.util import nest
else:
    from tensorflow.contrib.framework import nest


