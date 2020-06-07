import tensorflow as tf
import tree

from softlearning.models.feedforward import feedforward_model
from softlearning.models.utils import create_inputs
from softlearning.utils.tensorflow import apply_preprocessors
from softlearning import preprocessors as preprocessors_lib
from softlearning.utils.tensorflow import cast_and_concat

from .base_value_function import StateActionValueFunction


def create_ensemble_value_function(N, value_fn, *args, **kwargs):
    # TODO(hartikainen): The ensemble Q-function should support the same
    # interface as the regular ones. Implement the double min-thing
    # as a Keras layer.
    value_fns = tuple(value_fn(*args, **kwargs) for i in range(N))
    return value_fns


def double_feedforward_Q_function(*args, **kwargs):
    return create_ensemble_value_function(
        2, feedforward_Q_function, *args, **kwargs)


def ensemble_feedforward_Q_function(N, *args, **kwargs):
    return create_ensemble_value_function(
        N, feedforward_Q_function, *args, **kwargs)


def feedforward_Q_function(input_shapes,
                           *args,
                           preprocessors=None,
                           observation_keys=None,
                           name='feedforward_Q',
                           **kwargs):
    inputs = create_inputs(input_shapes)

    if preprocessors is None:
        preprocessors = tree.map_structure(lambda _: None, inputs)

    preprocessors = tree.map_structure_up_to(
        inputs, preprocessors_lib.deserialize, preprocessors)

    preprocessed_inputs = apply_preprocessors(preprocessors, inputs)

    # NOTE(hartikainen): `feedforward_model` would do the `cast_and_concat`
    # step for us, but tf2.2 broke the sequential multi-input handling: See:
    # https://github.com/tensorflow/tensorflow/issues/37061.
    out = tf.keras.layers.Lambda(cast_and_concat)(preprocessed_inputs)
    Q_model_body = feedforward_model(
        *args,
        output_shape=[1],
        name=name,
        **kwargs
    )

    Q_model = tf.keras.Model(inputs, Q_model_body(out), name=name)

    Q_function = StateActionValueFunction(
        model=Q_model, observation_keys=observation_keys, name=name)

    return Q_function
