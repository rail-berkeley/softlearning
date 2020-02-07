import pickle
from collections import OrderedDict

import numpy as np
import pytest
import tensorflow as tf
import tree

from softlearning.environments.utils import get_environment
from softlearning.policies.gaussian_policy import FeedforwardGaussianPolicy


class GaussianPolicyTest(tf.test.TestCase):
    def setUp(self):
        self.env = get_environment('gym', 'Swimmer', 'v3', {})
        self.hidden_layer_sizes = (16, 16)

        self.policy = FeedforwardGaussianPolicy(
            input_shapes=self.env.observation_shape,
            output_shape=self.env.action_space.shape,
            action_range=(
                self.env.action_space.low,
                self.env.action_space.high,
            ),
            hidden_layer_sizes=self.hidden_layer_sizes,
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

            self.assertEqual(actions.shape, (2, *self.env.action_space.shape))
            self.assertEqual(log_pis.shape, (2, 1))

    def test_env_step_with_actions(self):
        observation_np = self.env.reset()
        action = self.policy.action(observation_np).numpy()
        self.env.step(action)

    def test_trainable_variables(self):
        self.assertEqual(
            tuple(self.policy.trainable_variables),
            tuple(self.policy.shift_and_scale_model.trainable_variables))

        self.assertEqual(
            len(self.policy.trainable_variables),
            2 * (len(self.hidden_layer_sizes) + 1))

    def test_get_diagnostics(self):
        observation1_np = self.env.reset()
        observation2_np = self.env.step(self.env.action_space.sample())[0]

        observations_np = type(observation1_np)((
            (key, np.stack((
                observation1_np[key], observation2_np[key]
            ), axis=0).astype(np.float32))
            for key in observation1_np.keys()
        ))

        diagnostics = self.policy.get_diagnostics_np(observations_np)

        self.assertTrue(isinstance(diagnostics, OrderedDict))
        self.assertEqual(
            tuple(diagnostics.keys()),
            ('shifts-mean',
             'shifts-std',
             'scales-mean',
             'scales-std',
             'entropy-mean',
             'entropy-std',
             'actions-mean',
             'actions-std',
             'actions-min',
             'actions-max'))

        for value in diagnostics.values():
            self.assertTrue(np.isscalar(value))

    def test_serialize_deserialize(self):
        observation1_np = self.env.reset()
        observation2_np = self.env.step(self.env.action_space.sample())[0]

        observations_np = type(observation1_np)((
            (key, np.stack((
                observation1_np[key], observation2_np[key]
            ), axis=0).astype(np.float32))
            for key in observation1_np.keys()
        ))

        weights = self.policy.get_weights()
        actions_np = self.policy.actions(observations_np).numpy()
        log_pis_np = self.policy.log_probs(observations_np, actions_np).numpy()

        serialized = pickle.dumps(self.policy)
        deserialized = pickle.loads(serialized)

        weights_2 = deserialized.get_weights()
        log_pis_np_2 = deserialized.log_probs(
            observations_np, actions_np).numpy()

        for weight, weight_2 in zip(weights, weights_2):
            np.testing.assert_array_equal(weight, weight_2)

        np.testing.assert_array_equal(log_pis_np, log_pis_np_2)
        np.testing.assert_equal(
            actions_np.shape,
            deserialized.actions(observations_np).numpy().shape)

    @pytest.mark.skip("Latent smoothing is temporarily disabled.")
    def test_latent_smoothing(self):
        observation_np = self.env.reset()
        smoothed_policy = FeedforwardGaussianPolicy(
            input_shapes=self.env.observation_shape,
            output_shape=self.env.action_space.shape,
            action_range=(
                self.env.action_space.low,
                self.env.action_space.high,
            ),
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
            action_np = smoothed_policy.action(observation_np).numpy()
            observation_np = self.env.step(action_np)[0]

            self.assertFalse(np.all(np.equal(
                smoothing_x_previous,
                smoothed_policy._smoothing_x)))
            smoothing_x_previous = smoothed_policy._smoothing_x

        smoothed_policy.reset()

        np.testing.assert_equal(smoothed_policy._smoothing_x, 0.0)


if __name__ == '__main__':
    tf.test.main()
