import tensorflow as tf


def feedforward_preprocessor_model(
        input_shape,
        hidden_layer_sizes,
        output_size,
        ignore_input=0,
        activation=tf.nn.relu,
        output_activation='linear',
        name="feedforward_preprocessor",
        *args,
        **kwargs):

    inputs = tf.keras.layers.Input(shape=input_shape)

    def split_passthrough_layer(x):
        if ignore_input > 0:
            inputs_to_preprocess = x[..., :-ignore_input]
            passthrough = x[..., -ignore_input:]
        else:
            inputs_to_preprocess = x
            passthrough = x[..., 0:0]

        return inputs_to_preprocess, passthrough

    inputs_to_preprocess, passthrough = tf.keras.layers.Lambda(
        split_passthrough_layer)(inputs)

    out = inputs_to_preprocess
    for units in hidden_layer_sizes:
        out = tf.keras.layers.Dense(
            units, *args, activation=activation, **kwargs)(out)

    preprocessed = tf.keras.layers.Dense(
        output_size-ignore_input,
        *args,
        activation=output_activation,
        **kwargs)(out)

    concatenated = tf.keras.layers.Concatenate(
        axis=-1)([preprocessed, passthrough])

    model = tf.keras.Model(inputs, concatenated, name=name)

    return model
