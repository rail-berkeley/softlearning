import tensorflow as tf


def feedforward_model(input_shapes,
                      hidden_layer_sizes,
                      output_size,
                      activation='relu',
                      output_activation='linear',
                      name=None,
                      *args,
                      **kwargs):
    model = tf.keras.Sequential(name=name)

    model.add(tf.keras.layers.Lambda(lambda x: tf.concat(x, axis=-1)))

    for units in hidden_layer_sizes:
        model.add(tf.keras.layers.Dense(
            units, *args, activation=activation, **kwargs))

    model.add(tf.keras.layers.Dense(
        output_size, *args, activation=output_activation, **kwargs))

    return model


def feedforward_net_v2(inputs,
                       hidden_layer_sizes,
                       output_size,
                       activation='relu',
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
