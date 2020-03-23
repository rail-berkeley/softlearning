from collections import OrderedDict

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tree

from softlearning import policies
from softlearning.policies.uniform_policy import ContinuousUniformPolicy
from softlearning.environments.utils import get_environment
from softlearning.samplers import utils as sampler_utils


class ContinuousUniformPolicyTest(tf.test.TestCase):
    def setUp(self):
        self.env = get_environment('gym', 'Swimmer', 'v3', {})
        self.policy = ContinuousUniformPolicy(
            action_range=(
                self.env.action_space.low,
                self.env.action_space.high,
            ),
            input_shapes=self.env.observation_shape,
            output_shape=self.env.action_shape,
            observation_keys=self.env.observation_keys)

    def test_actions_and_log_probs(self):
        observation1_np = self.env.reset()
        observation2_np = self.env.step(self.env.action_space.sample())[0]

        observations_np = type(observation1_np)((
            (key, np.stack((
                observation1_np[key], observation2_np[key]
            ), axis=0).astype(np.float32))
            for key in observation1_np.keys()
        ))

        observations_tf = tree.map_structure(
            lambda x: tf.constant(x, dtype=x.dtype), observations_np)

        for observations in (observations_np, observations_tf):
            actions = self.policy.actions(observations)
            log_pis = self.policy.log_probs(observations, actions)

            self.assertAllEqual(
                log_pis,
                tfp.distributions.Independent(
                    tfp.distributions.Uniform(
                        low=self.env.action_space.low,
                        high=self.env.action_space.high,
                    ),
                    reinterpreted_batch_ndims=1,
                ).log_prob(actions)[..., None])

            self.assertEqual(actions.shape, (2, *self.env.action_shape))

    def test_env_step_with_actions(self):
        observation_np = self.env.reset()
        action = self.policy.action(observation_np).numpy()
        self.env.step(action)

    def test_trainable_variables(self):
        self.assertEqual(len(self.policy.trainable_variables), 0)

    def test_get_diagnostics(self):
        observation1_np = self.env.reset()
        observation2_np = self.env.step(self.env.action_space.sample())[0]
        observations_np = {}
        observations_np = type(observation1_np)((
            (key, np.stack((
                observation1_np[key], observation2_np[key]
            ), axis=0).astype(np.float32))
            for key in observation1_np.keys()
        ))

        diagnostics = self.policy.get_diagnostics(observations_np)
        self.assertTrue(isinstance(diagnostics, OrderedDict))
        self.assertFalse(diagnostics)

    def test_serialize_deserialize(self):
        policy_1 = ContinuousUniformPolicy(
            action_range=(
                self.env.action_space.low,
                self.env.action_space.high,
            ),
            input_shapes=self.env.observation_shape,
            output_shape=self.env.action_shape,
            observation_keys=self.env.observation_keys)

        self.assertFalse(policy_1.trainable_weights)

        config = policies.serialize(policy_1)
        policy_2 = policies.deserialize(config)

        self.assertEqual(policy_2._action_range, policy_1._action_range)
        self.assertEqual(policy_2._input_shapes, policy_1._input_shapes)
        self.assertEqual(policy_2._output_shape, policy_1._output_shape)
        self.assertEqual(
            policy_2._observation_keys, policy_1._observation_keys)

        path = sampler_utils.rollout(
            self.env,
            policy_2,
            path_length=10,
            break_on_terminal=False)
        observations = path['observations']
        np.testing.assert_equal(
            policy_1.actions(observations).numpy().shape,
            policy_2.actions(observations).numpy().shape)


if __name__ == '__main__':
    tf.test.main()
