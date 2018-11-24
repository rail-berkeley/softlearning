from softlearning.models.feedforward import feedforward_model


def create_feedforward_Q_function(observation_shape,
                                  action_shape,
                                  *args,
                                  name='feedforward_Q',
                                  **kwargs):
    input_shapes = (observation_shape, action_shape)
    return feedforward_model(
        input_shapes, *args, output_size=1, name=name, **kwargs)


def create_feedforward_V_function(observation_shape,
                                  *args,
                                  name='feedforward_V',
                                  **kwargs):
    input_shapes = (observation_shape, )
    return feedforward_model(input_shapes, *args, output_size=1, **kwargs)
