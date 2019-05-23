import tensorflow as tf
from flatten_dict import flatten


def get_inputs_for_nested_shapes(input_shapes, name=None):
    if isinstance(input_shapes, dict):
        return type(input_shapes)([
            (name, get_inputs_for_nested_shapes(value, name))
            for name, value in input_shapes.items()
        ])
    elif isinstance(input_shapes, (tuple, list)):
        if all(isinstance(x, int) for x in input_shapes):
            return tf.keras.layers.Input(shape=input_shapes, name=name)
        else:
            return type(input_shapes)((
                get_inputs_for_nested_shapes(input_shape, name=None)
                for input_shape in input_shapes
            ))
    elif isinstance(input_shapes, tf.TensorShape):
        return tf.keras.layers.Input(shape=input_shapes, name=name)

    raise NotImplementedError(input_shapes)


def flatten_input_structure(inputs):
    if isinstance(inputs, dict):
        inputs_flat_dict = flatten(inputs)
        inputs_flat = [
            inputs_flat_dict[key]
            for key in sorted(inputs_flat_dict.keys())
        ]
    elif isinstance(inputs, list):
        inputs_flat = list(inputs)
    elif isinstance(inputs, tuple):
        if all (isinstance(x, int) for x in inputs):
            inputs_flat = [inputs]
        else:
            inputs_flat = list(inputs)

    return inputs_flat


def create_inputs(input_shapes):
    """Creates `tf.keras.layers.Input`s based on input shapes.

    Args:
        input_shapes: (possibly nested) list/array/dict structure of
        inputs shapes.

    Returns:
        inputs_flat: a tuple of `tf.keras.layers.Input`s.

    TODO(hartikainen): Need to figure out a better way for handling the dtypes.
    """
    if isinstance(input_shapes, dict):
        inputs_flat_dict = flatten(input_shapes)
        inputs_flat = [
            tf.keras.layers.Input(
                shape=inputs_flat_dict[key],
                name=key[-1],
                dtype=(tf.uint8 # Image observation
                       if len(inputs_flat_dict[key]) == 3
                       else tf.float32) # Non-image
            )
            for key in sorted(inputs_flat_dict.keys())
        ]
    elif isinstance(input_shapes, list):
        inputs_flat = [
            tf.keras.layers.Input(shape=shape)
            for shape in input_shapes
        ]
    elif isinstance(input_shapes, tuple):
        if all (isinstance(x, int) for x in input_shapes):
            inputs_flat = [tf.keras.layers.Input(shape=input_shapes)]
        else:
            inputs_flat = [
                tf.keras.layers.Input(shape=shape)
                for shape in input_shapes
            ]

    return inputs_flat
