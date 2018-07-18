import tensorflow as tf

from rllab.core.serializable import Serializable

from rllab.misc.overrides import overrides
from sandbox.rocky.tf.policies.base import Policy


class NNPolicy(Policy, Serializable):
    def __init__(self, name, env_spec, observation_ph, actions):
        Serializable.quick_init(self, locals())

        self.name = name
        self._observations_ph = observation_ph
        self._actions = actions

        super(NNPolicy, self).__init__(env_spec)

    @overrides
    def get_action(self, observation, with_log_pis=False, with_raw_actions=False):
        """Sample single action based on the observations."""
        outputs = self.get_actions(observation[None], with_log_pis, with_raw_actions)
        if with_log_pis or with_raw_actions:
            outputs = [output[0] for output in outputs]
            return outputs + [{}]

        return outputs[0][0], {}

    @overrides
    def get_actions(self, observations, with_log_pis=False, with_raw_actions=False):
        """Sample actions based on the observations."""
        feed_dict = {self._observations_ph: observations}
        ops = [self._actions]
        if with_log_pis:
            ops.append(self._log_pis)
        if with_raw_actions:
            ops.append(self._raw_actions)
        outputs = tf.get_default_session().run(ops, feed_dict)
        return outputs

    @overrides
    def log_diagnostics(self, paths):
        pass

    @overrides
    def get_params_internal(self, scope='', **tags):
        """TODO: rewrite this using tensorflow collections."""
        if tags:
            raise NotImplementedError

        scope = scope or tf.get_variable_scope().name
        scope += '/' + self.name if len(scope) else self.name

        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
