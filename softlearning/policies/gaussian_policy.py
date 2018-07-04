""" Gaussian mixture policy. """

from contextlib import contextmanager
import numpy as np
import tensorflow as tf

from rllab.misc.overrides import overrides
from rllab.misc import logger
from rllab.core.serializable import Serializable

from softlearning.distributions import Normal
from softlearning.policies import NNPolicy
from softlearning.misc import tf_utils

EPS = 1e-6

class GaussianPolicy(NNPolicy, Serializable):
    def __init__(self, env_spec, hidden_layer_sizes=(100, 100), reg=1e-3,
                 squash=True, reparameterize=True, name='gaussian_policy'):
        """
        Args:
            env_spec (`rllab.EnvSpec`): Specification of the environment
                to create the policy for.
            hidden_layer_sizes (`list` of `int`): Sizes for the Multilayer
                perceptron hidden layers.
            reg (`float`): Regularization coeffiecient for the Gaussian parameters.
            squash (`bool`): If True, squash the Gaussian the gmm action samples
               between -1 and 1 with tanh.
            reparameterize ('bool'): If True, gradients will flow directly through
                the action samples.
        """
        Serializable.quick_init(self, locals())

        self._hidden_layers = hidden_layer_sizes
        self._Da = env_spec.action_space.flat_dim
        self._Ds = env_spec.observation_space.flat_dim
        self._is_deterministic = False
        self._squash = squash
        self._reparameterize = reparameterize
        self._reg = reg

        self.name = name
        self.build()

        self._scope_name = (
            tf.get_variable_scope().name + "/" + name
        ).lstrip("/")

        super(NNPolicy, self).__init__(env_spec)

    def actions_for(self, observations, latents=None,
                    name=None, reuse=tf.AUTO_REUSE,
                    with_log_pis=False, regularize=False):
        name = name or self.name

        with tf.variable_scope(name, reuse=reuse):
            distribution = Normal(
                hidden_layers_sizes=self._hidden_layers,
                Dx=self._Da,
                reparameterize=self._reparameterize,
                cond_t_lst=(observations,),
                reg=self._reg
            )
        raw_actions = distribution.x_t
        actions = tf.tanh(raw_actions) if self._squash else raw_actions

        # TODO: should always return same shape out
        # Figure out how to make the interface for `log_pis` cleaner
        if with_log_pis:
            # TODO.code_consolidation: should come from log_pis_for
            log_pis = self._log_pis_for_raw(observations, raw_actions,
                            name, reuse=True, regularize=regularize)
            return actions, log_pis

        return actions

    def _log_pis_for_raw(self, observations, actions, name=None, 
                        reuse=tf.AUTO_REUSE, regularize=False, stable=True):
        name = name or self.name

        with tf.variable_scope(name, reuse=reuse):
            distribution = Normal(
                hidden_layers_sizes=self._hidden_layers,
                Dx=self._Da,
                reparameterize=self._reparameterize,
                cond_t_lst=(observations,),
                reg=self._reg
            )
        log_pis = distribution.log_prob(actions) 
        if self._squash:
            log_pis -= self._squash_correction(actions, stable)
        return log_pis

    def log_pis_for(self, observations, actions, name=None,
                    reuse=tf.AUTO_REUSE, regularize=False, stable=True):
        if self._squash:
            actions = tf.atanh(actions)
        return self._log_pis_for_raw(observations, actions, name, 
                                    reuse, regularize, stable)
        

    def build(self):
        self._observations_ph = tf.placeholder(
            dtype=tf.float32,
            shape=(None, self._Ds),
            name='observations',
        )
        self._actions_ph = tf.placeholder(
            dtype=tf.float32,
            shape=(None, self._Da),
            name='actions',
        )

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.distribution = Normal(
                hidden_layers_sizes=self._hidden_layers,
                Dx=self._Da,
                reparameterize=self._reparameterize,
                cond_t_lst=(self._observations_ph,),
                reg=self._reg,
            )

        raw_actions = self.distribution.x_t
        self._actions = tf.tanh(raw_actions) if self._squash else raw_actions
        self._raw_actions = raw_actions

    @overrides
    def get_actions(self, observations):
        """Sample actions based on the observations.

        If `self._is_deterministic` is True, returns the mean action for the
        observations. If False, return stochastically sampled action.

        TODO.code_consolidation: This should be somewhat similar with
        `LatentSpacePolicy.get_actions`.
        """
        if self._is_deterministic: # Handle the deterministic case separately.

            feed_dict = {self._observations_ph: observations}

            # TODO.code_consolidation: these shapes should be double checked
            # for case where `observations.shape[0] > 1`
            mu = tf.get_default_session().run(
                self.distribution.mu_t, feed_dict)  # 1 x Da
            if self._squash:
                mu = np.tanh(mu)

            return mu

        return super(GaussianPolicy, self).get_actions(observations)

    def _squash_correction(self, actions, stable=True):
        if not self._squash: return 0
        # more stable squash correction
        if stable:
            return tf.reduce_sum(2. * (tf.log(2.) - actions - tf.nn.softplus(-2. * actions)), axis=1)
        else:
            return tf.reduce_sum(tf.log(1 - tf.tanh(actions) ** 2 + EPS), axis=1)

    @contextmanager
    def deterministic(self, set_deterministic=True):
        """Context manager for changing the determinism of the policy.

        See `self.get_action` for further information about the effect of
        self._is_deterministic.

        Args:
            set_deterministic (`bool`): Value to set the self._is_deterministic
                to during the context. The value will be reset back to the
                previous value when the context exits.
        """
        was_deterministic = self._is_deterministic

        self._is_deterministic = set_deterministic

        yield

        self._is_deterministic = was_deterministic

    def log_diagnostics(self, iteration, batch):
        """Record diagnostic information to the logger.

        Records the mean, min, max, and standard deviation of the 
        means and covariances.
        """

        feeds = {self._observations_ph: batch['observations']}
        sess = tf_utils.get_default_session()
        actions, raw_actions, mu, log_sig, log_pi_stable, log_pi_eps = sess.run(
            (
                self._actions,
                self._raw_actions,
                self.distribution.mu_t,
                self.distribution.log_sig_t,
                # TODO: Move log_pi and correction under self.log_pi_for()
                (self.distribution.log_p_t
                 - self._squash_correction(self.distribution.x_t, stable=True)),
                (self.distribution.log_p_t
                 - self._squash_correction(self.distribution.x_t, stable=False)),
            ),
            feeds
        )

        # does the atanh
        log_pis_for_eps_t = self.log_pis_for(self._observations_ph, self._actions_ph, stable=False)
        log_pis_for_stable_t = self.log_pis_for(self._observations_ph, self._actions_ph, stable=True)

        feeds[self._actions_ph] = actions
        log_pis_for_eps, log_pis_for_stable = sess.run(
            (log_pis_for_eps_t, log_pis_for_stable_t), feeds
        )
        no_atanh_diff = np.absolute(log_pi_stable - log_pi_eps)
        atanh_diff = np.absolute(log_pis_for_stable - log_pis_for_eps)

        no_eps_diff = np.absolute(log_pi_stable - log_pis_for_stable)
        with_eps_diff = np.absolute(log_pi_eps - log_pis_for_eps)


        logger.record_tabular('no-atanh-diff-mean', np.mean(no_atanh_diff))
        logger.record_tabular('no-atanh-diff-max', np.max(no_atanh_diff))
        logger.record_tabular('no-atanh-diff-std', np.std(no_atanh_diff))

        logger.record_tabular('atanh-diff-mean', np.mean(atanh_diff))
        logger.record_tabular('atanh-diff-max', np.max(atanh_diff))
        logger.record_tabular('atanh-diff-std', np.std(atanh_diff))

        logger.record_tabular('no-eps-diff-mean', np.mean(no_eps_diff))
        logger.record_tabular('no-eps-diff-max', np.max(no_eps_diff))
        logger.record_tabular('no-eps-diff-std', np.std(no_eps_diff))

        logger.record_tabular('with-eps-diff-mean', np.mean(with_eps_diff))
        logger.record_tabular('with-eps-diff-max', np.max(with_eps_diff))
        logger.record_tabular('with-eps-diff-std', np.std(with_eps_diff))

        logger.record_tabular('policy-mus-mean', np.mean(mu))
        logger.record_tabular('policy-mus-min', np.min(mu))
        logger.record_tabular('policy-mus-max', np.max(mu))
        logger.record_tabular('policy-mus-std', np.std(mu))

        logger.record_tabular('log-sigs-mean', np.mean(log_sig))
        logger.record_tabular('log-sigs-min', np.min(log_sig))
        logger.record_tabular('log-sigs-max', np.max(log_sig))
        logger.record_tabular('log-sigs-std', np.std(log_sig))

        logger.record_tabular('-log-pi-mean', np.mean(-log_pi_eps))
        logger.record_tabular('-log-pi-max', np.max(-log_pi_eps))
        logger.record_tabular('-log-pi-min', np.min(-log_pi_eps))
        logger.record_tabular('-log-pi-std', np.std(-log_pi_eps))
