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


def create_input(path, input_shape):
    name = "/".join(str(x) for x in path)
    input_ = tf.keras.layers.Input(
        shape=input_shape,
        name=name,
        dtype=(tf.uint8  # Image observation
               if len(input_shape) == 3 and input_shape[-1] in (1, 3)
               else tf.float32),  # Non-image
    )

    return input_


def create_inputs(input_shapes):
    """Creates `tf.keras.layers.Input`s based on input shapes.

    Args:
        input_shapes: (possibly nested) list/array/dict structure of
        inputs shapes.

    Returns:
        inputs: nested structure, of same shape as input_shapes, containing
        `tf.keras.layers.Input`s.

    TODO(hartikainen): Need to figure out a better way for handling the dtypes.
    """
    inputs = tree.map_structure_with_path(create_input, input_shapes)

    return inputs
