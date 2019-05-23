from softlearning.models.feedforward import feedforward_model


def create_feedforward_Q_function(*args,
                                  observation_keys=None,
                                  name='feedforward_Q',
                                  **kwargs):
    Q_function = feedforward_model(
        *args,
        output_size=1,
        name=name,
        **kwargs)

    Q_function.observation_keys = observation_keys

    return Q_function
