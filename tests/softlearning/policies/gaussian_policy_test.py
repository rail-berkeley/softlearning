import pickle
from collections import OrderedDict

import numpy as np
import tensorflow as tf

from softlearning.models.utils import flatten_input_structure
from softlearning.policies.gaussian_policy import FeedforwardGaussianPolicy
from softlearning.environments.utils import get_environment


class GaussianPolicyTest(tf.test.TestCase):
    def setUp(self):
        self.env = get_environment('gym', 'Swimmer', 'v3', {})
        self.hidden_layer_sizes = (128, 128)

        self.policy = FeedforwardGaussianPolicy(
            input_shapes=self.env.observation_shape,
            output_shape=self.env.action_space.shape,
            hidden_layer_sizes=self.hidden_layer_sizes,
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
        log_pis = self.policy.log_pis(observations_tf, actions)

        self.assertEqual(actions.shape, (2, *self.env.action_space.shape))
        self.assertEqual(log_pis.shape, (2, 1))

        self.evaluate(tf.compat.v1.global_variables_initializer())

        actions_np = self.evaluate(actions)
        log_pis_np = self.evaluate(log_pis)

        self.assertEqual(actions_np.shape, (2, *self.env.action_space.shape))
        self.assertEqual(log_pis_np.shape, (2, 1))

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
        log_pis_np = self.policy.log_pis_np(observations_np, actions_np)

        self.assertEqual(actions_np.shape, (2, *self.env.action_space.shape))
        self.assertEqual(log_pis_np.shape, (2, 1))

    def test_env_step_with_actions(self):
        observation_np = self.env.reset()
        observations_np = flatten_input_structure({
            key: value[None, :] for key, value in observation_np.items()
        })
        action = self.policy.actions_np(observations_np)[0, ...]
        self.env.step(action)

    def test_trainable_variables(self):
        self.assertEqual(
            len(self.policy.trainable_variables),
            2 * (len(self.hidden_layer_sizes) + 1))

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
        self.assertEqual(
            tuple(diagnostics.keys()),
            ('shifts-mean',
             'shifts-std',
             'log_scale_diags-mean',
             'log_scale_diags-std',
             '-log-pis-mean',
             '-log-pis-std',
             'raw-actions-mean',
             'raw-actions-std',
             'actions-mean',
             'actions-std',
             'actions-min',
             'actions-max'))

        for value in diagnostics.values():
            self.assertTrue(np.isscalar(value))

    def test_serialize_deserialize(self):
        observation1_np = self.env.reset()
        observation2_np = self.env.step(self.env.action_space.sample())[0]

        observations_np = {}
        for key in observation1_np.keys():
            observations_np[key] = np.stack((
                observation1_np[key], observation2_np[key]
            )).astype(np.float32)
        observations_np = flatten_input_structure(observations_np)

        weights = self.policy.get_weights()
        actions_np = self.policy.actions_np(observations_np)
        log_pis_np = self.policy.log_pis_np(observations_np, actions_np)

        serialized = pickle.dumps(self.policy)
        deserialized = pickle.loads(serialized)

        weights_2 = deserialized.get_weights()
        log_pis_np_2 = deserialized.log_pis_np(observations_np, actions_np)

        for weight, weight_2 in zip(weights, weights_2):
            np.testing.assert_array_equal(weight, weight_2)

        np.testing.assert_array_equal(log_pis_np, log_pis_np_2)
        np.testing.assert_equal(
            actions_np.shape, deserialized.actions_np(observations_np).shape)

    def test_latent_smoothing(self):
        observation_np = self.env.reset()
        smoothed_policy = FeedforwardGaussianPolicy(
            input_shapes=self.env.observation_shape,
            output_shape=self.env.action_space.shape,
            hidden_layer_sizes=self.hidden_layer_sizes,
            smoothing_coefficient=0.5,
            observation_keys=self.env.observation_keys)

        np.testing.assert_equal(smoothed_policy._smoothing_x, 0.0)
        self.assertEqual(smoothed_policy._smoothing_alpha, 0.5)
        self.assertEqual(
            smoothed_policy._smoothing_beta,
            np.sqrt((1.0 - np.power(smoothed_policy._smoothing_alpha, 2.0)))
            / (1.0 - smoothed_policy._smoothing_alpha))

        smoothing_x_previous = smoothed_policy._smoothing_x
        for i in range(5):
            observations_np = flatten_input_structure({
                key: value[None, :] for key, value in observation_np.items()
            })
            action_np = smoothed_policy.actions_np(observations_np)[0]
            observation_np = self.env.step(action_np)[0]

            self.assertFalse(np.all(np.equal(
                smoothing_x_previous,
                smoothed_policy._smoothing_x)))
            smoothing_x_previous = smoothed_policy._smoothing_x

        smoothed_policy.reset()

        np.testing.assert_equal(smoothed_policy._smoothing_x, 0.0)


if __name__ == '__main__':
    tf.test.main()
