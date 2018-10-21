import tensorflow as tf


def feedforward_model(inputs,
                      hidden_layer_sizes,
                      output_size,
                      activation='relu',
                      output_activation='linear',
                      name=None,
                      *args,
                      **kwargs):
    if isinstance(inputs, (list, tuple)):
        concatenated = (
            tf.keras.layers.Concatenate(axis=-1)(inputs)
            if len(inputs) > 1
            else inputs[0])
    else:
        concatenated = inputs

    out = concatenated
    for units in hidden_layer_sizes:
        out = tf.keras.layers.Dense(
            units, *args, activation=activation, **kwargs)(out)

    out = tf.keras.layers.Dense(
        output_size, *args, activation=output_activation, **kwargs)(out)

    model = tf.keras.Model(inputs, out, name=name)

    return model


def feedforward_net_v2(inputs,
                       hidden_layer_sizes,
                       output_size,
                       activation=tf.nn.relu,
                       output_activation='linear',
                       *args,
                       **kwargs):
    if isinstance(inputs, (list, tuple)):
        inputs = tf.concat(inputs, axis=-1)

    out = inputs
    for units in hidden_layer_sizes:
        out = tf.layers.dense(
            inputs=out,
            units=units,
            activation=activation,
            *args,
            **kwargs)

    out = tf.layers.dense(
        inputs=out,
        units=output_size,
        activation=output_activation,
        *args,
        **kwargs)

    return out
