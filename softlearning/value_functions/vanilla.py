from softlearning.models.feedforward import feedforward_model


def create_feedforward_Q_function(input_shapes,
                                  *args,
                                  preprocessors=None,
                                  observation_keys=None,
                                  name='feedforward_Q',
                                  **kwargs):
    Q_function = feedforward_model(
        input_shapes,
        *args,
        output_size=1,
        preprocessors=preprocessors,
        name=name,
        **kwargs)

    Q_function.observation_keys = observation_keys

    return Q_function
