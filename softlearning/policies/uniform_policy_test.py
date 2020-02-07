from collections import OrderedDict
import pickle

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tree

from softlearning.policies.uniform_policy import ContinuousUniformPolicy
from softlearning.environments.utils import get_environment


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
                tfp.distributions.Uniform(
                    low=self.env.action_space.low,
                    high=self.env.action_space.high
                ).log_prob(actions))

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
        observation1_np = self.env.reset()
        observation2_np = self.env.step(self.env.action_space.sample())[0]

        observations_np = type(observation1_np)((
            (key, np.stack((
                observation1_np[key], observation2_np[key]
            ), axis=0).astype(np.float32))
            for key in observation1_np.keys()
        ))

        deserialized = pickle.loads(pickle.dumps(self.policy))

        np.testing.assert_equal(
            self.policy.actions(observations_np).numpy().shape,
            deserialized.actions(observations_np).numpy().shape)


if __name__ == '__main__':
    tf.test.main()
