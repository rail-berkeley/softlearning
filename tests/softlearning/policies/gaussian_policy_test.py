import numpy as np
import tensorflow as tf

import gym

from softlearning.policies.gaussian_policy import GaussianPolicy


class GaussianPolicyTest(tf.test.TestCase):
    def setUp(self):
        self.env = gym.envs.make('Swimmer-v2')
        self.policy = GaussianPolicy(
            input_shapes=(self.env.observation_space.shape, ),
            output_shape=self.env.action_space.shape,
            hidden_layer_sizes=(128, 128))

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


if __name__ == '__main__':
    tf.test.main()
