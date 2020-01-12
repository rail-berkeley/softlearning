import tensorflow as tf
import tree


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


def apply_preprocessors(preprocessors, inputs):
    tree.assert_same_structure(inputs, preprocessors)
    preprocessed_inputs = tree.map_structure(
        lambda preprocessor, input_: (
            preprocessor(input_) if preprocessor is not None else input_),
        preprocessors,
        inputs,
    )

    return preprocessed_inputs


def cast_and_concat(x):
    x = tree.map_structure(
        lambda element: tf.cast(element, tf.float32), x)
    x = tree.flatten(x)
    x = tf.concat(x, axis=-1)
    return x
