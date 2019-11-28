from distutils.version import LooseVersion

import tensorflow as tf


if LooseVersion(tf.__version__) > LooseVersion("2.00"):
    from tensorflow import nest
else:
    from tensorflow.contrib.framework import nest


def initialize_tf_variables(session, only_uninitialized=True):
    variables = tf.compat.v1.global_variables() + tf.compat.v1.local_variables()

    def is_initialized(variable):
        try:
            session.run(variable)
            return True
        except tf.errors.FailedPreconditionError:
            return False

        return False

    if only_uninitialized:
        variables = [
            variable for variable in variables
            if not is_initialized(variable)
        ]

    session.run(tf.compat.v1.variables_initializer(variables))
