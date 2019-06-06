from distutils.version import LooseVersion

import tensorflow as tf


if LooseVersion(tf.__version__) > LooseVersion("2.00"):
    from tensorflow import nest
else:
    from tensorflow.contrib.framework import nest
