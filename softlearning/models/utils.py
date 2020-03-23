import tensorflow as tf
import tree


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
    inputs_flat = tree.flatten(inputs)
    return inputs_flat


def create_input(path, shape, dtype=None):
    name = "/".join(str(x) for x in path)

    if dtype is None:
        # TODO(hartikainen): This is not a very robust way to handle the
        # dtypes. Need to figure out something better.
        # Try to infer dtype manually
        dtype = (tf.uint8  # Image observation
                 if len(shape) == 3 and shape[-1] in (1, 3)
                 else tf.float32)  # Non-image

    input_ = tf.keras.layers.Input(
        shape=shape,
        name=name,
        dtype=dtype
    )

    return input_


def create_inputs(shapes, dtypes=None):
    """Creates `tf.keras.layers.Input`s based on input shapes.

    Args:
        input_shapes: (possibly nested) list/array/dict structure of
        inputs shapes.

    Returns:
        inputs: nested structure, of same shape as input_shapes, containing
        `tf.keras.layers.Input`s.

    TODO(hartikainen): Need to figure out a better way for handling the dtypes.
    """
    if dtypes is None:
        dtypes = tree.map_structure(lambda _: None, shapes)
    inputs = tree.map_structure_with_path(create_input, shapes, dtypes)

    return inputs


def create_sequence_inputs(shapes, dtypes=None):
    """Creates `tf.keras.layers.Input`s usable for sequential models like RNN.

    Args:
        See `create_inputs`.

    Returns:
        inputs: nested structure, of same shape as input_shapes, containing
        `tf.keras.layers.Input`s, each with shape (None, ...).
    """
    shapes = tree.map_structure(lambda x: tf.TensorShape([None]) + x, shapes)
    sequence_inputs = create_inputs(shapes, dtypes)

    return sequence_inputs
