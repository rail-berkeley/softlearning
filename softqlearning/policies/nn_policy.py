import tensorflow as tf
from rllab.misc.overrides import overrides
from rllab.policies.base import Policy
from rllab.core.serializable import Serializable


class NNPolicy(Policy, Serializable):
    """ NeuralNetwork wrapper that implements the rllab Policy interface. """
    def __init__(self, env_spec, obs_pl, action):
        Serializable.quick_init(self, locals())

        self._obs_pl = obs_pl
        self._action = action
        self._scope_name = tf.get_variable_scope().name
        super().__init__(env_spec)

    @overrides
    def get_params_internal(self, **tags):
        if len(tags) > 0:
            raise NotImplementedError
        scope = self._scope_name
        # Add "/" to 'scope' unless it's empty (otherwise get_collection will
        # return all parameters that start with 'scope'.
        scope = scope if scope == '' else scope + '/'
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

    @overrides
    def get_action(self, observation):
        feeds = {self._obs_pl: observation[None]}
        action = tf.get_default_session().run(self._action, feeds)
        return action.squeeze(), None

    def plot_samples(self, ax_lst, obs_lst, output=None):
        output = self._action if output is None else output

        feed = {self._obs_pl: obs_lst}

        actions_lst = tf.get_default_session().run(output, feed)

        for ax, actions in zip(ax_lst, actions_lst):
            x, y = actions[:, 0], actions[:, 1]
            ax.plot(x, y, '*')

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        d.update({
            'params': self.get_param_values(),
        })
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        tf.get_default_session().run(
            tf.variables_initializer(self.get_params())
        )
        self.set_param_values(d['params'])
