import abc
from collections import OrderedDict

import tensorflow as tf

from softlearning.utils.tensorflow import nest


class BaseValueFunction:
    def __init__(self, model, observation_keys):
        self._observation_keys = observation_keys
        self.model = model

    @property
    def observation_keys(self):
        return self._observation_keys

    def reset(self):
        """Reset and clean the value function."""

    @property
    def trainable_variables(self):
        return self.model.trainable_variables

    def get_weights(self, *args, **kwargs):
        return self.model.get_weights(*args, **kwargs)

    def set_weights(self, *args, **kwargs):
        return self.model.set_weights(*args, **kwargs)

    def save_weights(self, *args, **kwargs):
        self.model.save_weights(*args, **kwargs)

    def load_weights(self, *args, **kwargs):
        self.model.load_weights(*args, **kwargs)

    @abc.abstractmethod
    def values(self, inputs):
        """Compute values for given inputs, (e.g. observations)."""
        raise NotImplementedError

    def value(self, input_):
        """Compute a value for a single input, (e.g. observation)."""
        inputs = nest.map_structure(lambda x: x[None, ...], input_)
        values = self.values(inputs)
        value = nest.map_structure(lambda x: x[0], values)
        return value

    def _filter_observations(self, observations):
        if (isinstance(observations, dict)
            and self._observation_keys is not None):
            observations = type((observations))((
                (key, observations[key])
                for key in self.observation_keys
            ))
        return observations

    def get_diagnostics(self, *inputs):
        """Return loggable diagnostic information of the value function."""
        diagnostics = OrderedDict()
        return diagnostics

    def __getstate__(self):
        state = self.__dict__.copy()
        model = state.pop('model')
        state.update({
            'model_config': model.get_config(),
            'model_weights': model.get_weights(),
        })
        return state

    def __setstate__(self, state):
        model_config = state.pop('model_config')
        model_weights = state.pop('model_weights')
        model = tf.keras.Model.from_config(model_config)
        model.set_weights(model_weights)
        state['model'] = model
        self.__dict__ = state


class StateValueFunction(BaseValueFunction):
    def values(self, observations):
        """Compute values given observations."""
        observations = self._filter_observations(observations)
        values = self.model(observations)
        return values


class StateActionValueFunction(BaseValueFunction):
    def values(self, observations, actions):
        """Compute values given observations."""
        observations = self._filter_observations(observations)
        values = self.model((observations, actions))
        return values


# def feedforward_Q_function(*args, name='feedforward_Q', **kwargs):
#     model = feedforward_model(
#         *args,
#         output_size=1,
#         name=name,
#         **kwargs
#     )

#     return StateActionValueFunction(model=model)
