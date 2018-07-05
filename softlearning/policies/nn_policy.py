import tensorflow as tf

from rllab.core.serializable import Serializable

from rllab.misc.overrides import overrides
from sandbox.rocky.tf.policies.base import Policy


class NNPolicy(Policy, Serializable):
    def __init__(self, env_spec, observation_ph, actions,
                 scope_name=None):
        Serializable.quick_init(self, locals())

        self._observations_ph = observation_ph
        self._actions = actions
        self._scope_name = (
            tf.get_variable_scope().name if not scope_name else scope_name
        )
        super(NNPolicy, self).__init__(env_spec)

    @overrides
    def get_action(self, observation, with_log_pis=False, with_raw_actions=False):
        """Sample single action based on the observations."""
        outputs = self.get_actions(observation[None], with_log_pis, with_raw_actions)
        if with_log_pis or with_raw_actions:
            outputs = [output[0] for output in outputs]
            return *outputs, {}

        return outputs[0], {}

    @overrides
    def get_actions(self, observations, with_log_pis=False, with_raw_actions=False):
        """Sample actions based on the observations."""
        feed_dict = {self._observations_ph: observations}
        ops = [self._actions]
        if with_log_pis:
            ops.append(self._log_pis)
        if with_raw_actions:
            ops.append(self._raw_actions)
        return tf.get_default_session().run(ops, feed_dict)

    @overrides
    def log_diagnostics(self, paths):
        pass

    @overrides
    def get_params_internal(self, **tags):
        """TODO: rewrite this using tensorflow collections."""
        if tags:
            raise NotImplementedError
        scope = self._scope_name
        # Add "/" to 'scope' unless it's empty (otherwise get_collection will
        # return all parameters that start with 'scope'.
        scope = scope if scope == '' else scope + '/'
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
