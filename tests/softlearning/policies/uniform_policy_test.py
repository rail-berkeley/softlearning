from collections import OrderedDict
import pickle

import numpy as np
import tensorflow as tf

from softlearning.models.utils import flatten_input_structure
from softlearning.policies.uniform_policy import UniformPolicy
from softlearning.environments.utils import get_environment


class UniformPolicyTest(tf.test.TestCase):
    def setUp(self):
        self.env = get_environment('gym', 'Swimmer', 'v3', {})
        self.policy = UniformPolicy(
            input_shapes=self.env.observation_shape,
            output_shape=self.env.action_space.shape,
            observation_keys=self.env.observation_keys)

    def test_actions_and_log_pis_symbolic(self):
        observation1_np = self.env.reset()
        observation2_np = self.env.step(self.env.action_space.sample())[0]

        observations_np = {}
        for key in observation1_np.keys():
            observations_np[key] = np.stack((
                observation1_np[key], observation2_np[key]
            )).astype(np.float32)

        observations_np = flatten_input_structure(observations_np)
        observations_tf = [tf.constant(x, dtype=tf.float32)
                           for x in observations_np]

        actions = self.policy.actions(observations_tf)
        with self.assertRaises(NotImplementedError):
            log_pis = self.policy.log_pis(observations_tf, actions)

        self.assertEqual(actions.shape, (2, *self.env.action_space.shape))

        self.evaluate(tf.compat.v1.global_variables_initializer())

        actions_np = self.evaluate(actions)

        self.assertEqual(actions_np.shape, (2, *self.env.action_space.shape))

    def test_actions_and_log_pis_numeric(self):
        observation1_np = self.env.reset()
        observation2_np = self.env.step(self.env.action_space.sample())[0]

        observations_np = {}
        for key in observation1_np.keys():
            observations_np[key] = np.stack((
                observation1_np[key], observation2_np[key]
            )).astype(np.float32)
        observations_np = flatten_input_structure(observations_np)

        actions_np = self.policy.actions_np(observations_np)
        with self.assertRaises(NotImplementedError):
            log_pis_np = self.policy.log_pis_np(observations_np, actions_np)

        self.assertEqual(actions_np.shape, (2, *self.env.action_space.shape))

    def test_env_step_with_actions(self):
        observation1_np = self.env.reset()
        observations_np = flatten_input_structure({
            key: value[None, :] for key, value in observation1_np.items()
        })
        action = self.policy.actions_np(observations_np)[0, ...]
        self.env.step(action)

    def test_trainable_variables(self):
        self.assertEqual(len(self.policy.trainable_variables), 0)

    def test_get_diagnostics(self):
        observation1_np = self.env.reset()
        observation2_np = self.env.step(self.env.action_space.sample())[0]
        observations_np = {}
        for key in observation1_np.keys():
            observations_np[key] = np.stack((
                observation1_np[key], observation2_np[key]
            )).astype(np.float32)
        observations_np = flatten_input_structure(observations_np)

        diagnostics = self.policy.get_diagnostics(observations_np)
        self.assertTrue(isinstance(diagnostics, OrderedDict))
        self.assertFalse(diagnostics)

    def test_serialize_deserialize(self):
        observation1_np = self.env.reset()
        observation2_np = self.env.step(self.env.action_space.sample())[0]

        observations_np = {}
        for key in observation1_np.keys():
            observations_np[key] = np.stack((
                observation1_np[key], observation2_np[key]
            )).astype(np.float32)
        observations_np = flatten_input_structure(observations_np)

        deserialized = pickle.loads(pickle.dumps(self.policy))

        np.testing.assert_equal(
            self.policy.actions_np(observations_np).shape,
            deserialized.actions_np(observations_np).shape)


if __name__ == '__main__':
    tf.test.main()
