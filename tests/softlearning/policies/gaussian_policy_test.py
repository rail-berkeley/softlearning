import pickle
from collections import OrderedDict

import numpy as np
import tensorflow as tf

import gym

from softlearning.policies.gaussian_policy import FeedforwardGaussianPolicy


class GaussianPolicyTest(tf.test.TestCase):
    def setUp(self):
        self.env = gym.envs.make('Swimmer-v3')
        self.hidden_layer_sizes = (128, 128)
        self.policy = FeedforwardGaussianPolicy(
            input_shapes=(self.env.observation_space.shape, ),
            output_shape=self.env.action_space.shape,
            hidden_layer_sizes=self.hidden_layer_sizes)

    def test_actions_and_log_pis_symbolic(self):
        observation1_np = self.env.reset()
        observation2_np = self.env.step(self.env.action_space.sample())[0]
        observations_np = np.stack((observation1_np, observation2_np))

        observations_tf = tf.constant(observations_np, dtype=tf.float32)

        actions = self.policy.actions([observations_tf])
        log_pis = self.policy.log_pis([observations_tf], actions)

        self.assertEqual(actions.shape, (2, *self.env.action_space.shape))
        self.assertEqual(log_pis.shape, (2, 1))

        self.evaluate(tf.global_variables_initializer())

        actions_np = self.evaluate(actions)
        log_pis_np = self.evaluate(log_pis)

        self.assertEqual(actions_np.shape, (2, *self.env.action_space.shape))
        self.assertEqual(log_pis_np.shape, (2, 1))

    def test_actions_and_log_pis_numeric(self):
        observation1_np = self.env.reset()
        observation2_np = self.env.step(self.env.action_space.sample())[0]
        observations_np = np.stack((observation1_np, observation2_np))

        actions_np = self.policy.actions_np([observations_np])
        log_pis_np = self.policy.log_pis_np([observations_np], actions_np)

        self.assertEqual(actions_np.shape, (2, *self.env.action_space.shape))
        self.assertEqual(log_pis_np.shape, (2, 1))

    def test_env_step_with_actions(self):
        observation1_np = self.env.reset()
        action = self.policy.actions_np(observation1_np[None])[0, ...]
        self.env.step(action)

    def test_trainable_variables(self):
        self.assertEqual(
            len(self.policy.trainable_variables),
            2 * (len(self.hidden_layer_sizes) + 1))

    def test_get_diagnostics(self):
        observation1_np = self.env.reset()
        observation2_np = self.env.step(self.env.action_space.sample())[0]
        observations_np = np.stack((observation1_np, observation2_np))

        diagnostics = self.policy.get_diagnostics([observations_np])

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
             'actions-std'))

        for value in diagnostics.values():
            self.assertTrue(np.isscalar(value))

    def test_serialize_deserialize(self):
        observation1_np = self.env.reset()
        observation2_np = self.env.step(self.env.action_space.sample())[0]
        observations_np = np.stack(
            (observation1_np, observation2_np)
        ).astype(np.float32)

        weights = self.policy.get_weights()
        actions_np = self.policy.actions_np([observations_np])
        log_pis_np = self.policy.log_pis_np([observations_np], actions_np)

        serialized = pickle.dumps(self.policy)
        deserialized = pickle.loads(serialized)

        weights_2 = deserialized.get_weights()
        log_pis_np_2 = deserialized.log_pis_np([observations_np], actions_np)

        for weight, weight_2 in zip(weights, weights_2):
            np.testing.assert_array_equal(weight, weight_2)

        np.testing.assert_array_equal(log_pis_np, log_pis_np_2)
        np.testing.assert_equal(
            actions_np.shape, deserialized.actions_np([observations_np]).shape)

    def test_latent_smoothing(self):
        observation_np = self.env.reset()
        smoothed_policy = FeedforwardGaussianPolicy(
            input_shapes=(self.env.observation_space.shape, ),
            output_shape=self.env.action_space.shape,
            hidden_layer_sizes=self.hidden_layer_sizes,
            smoothing_coefficient=0.5)

        np.testing.assert_equal(smoothed_policy._smoothing_x, 0.0)
        self.assertEqual(smoothed_policy._smoothing_alpha, 0.5)
        self.assertEqual(
            smoothed_policy._smoothing_beta,
            np.sqrt((1.0 - np.power(smoothed_policy._smoothing_alpha, 2.0)))
            / (1.0 - smoothed_policy._smoothing_alpha))

        smoothing_x_previous = smoothed_policy._smoothing_x
        for i in range(5):
            action_np = smoothed_policy.actions_np([
                observation_np[None, :]])[0]
            observation_np = self.env.step(action_np)[0]

            self.assertFalse(np.all(np.equal(
                smoothing_x_previous,
                smoothed_policy._smoothing_x)))
            smoothing_x_previous = smoothed_policy._smoothing_x

        smoothed_policy.reset()

        np.testing.assert_equal(smoothed_policy._smoothing_x, 0.0)


if __name__ == '__main__':
    tf.test.main()
