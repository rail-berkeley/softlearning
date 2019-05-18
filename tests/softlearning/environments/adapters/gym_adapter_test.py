import unittest
import pickle

import numpy as np

from .softlearning_env_test import AdapterTestClass
from softlearning.environments.adapters.gym_adapter import (
    GymAdapter, CUSTOM_GYM_ENVIRONMENTS)


SKIP_ENVIRONMENTS = (
    ('Pusher2d', 'ImageDefault-v0'),
    ('Pusher2d', 'ImageReach-v0'),
    ('Pusher2d', 'BlindReach-v0'))


class TestGymAdapter(unittest.TestCase, AdapterTestClass):
    def create_adapter(self, domain='Swimmer', task='v3', *args, **kwargs):
        return GymAdapter(domain, task, *args, **kwargs)

    def test_environments(self):
        # Make sure that all the environments are creatable
        def verify_reset_and_step(domain, task):
            env = GymAdapter(domain=domain, task=task)
            env.reset()
            env.step(env.action_space.sample())

        for domain, tasks in CUSTOM_GYM_ENVIRONMENTS.items():
            for task in tasks:
                if (domain, task) in SKIP_ENVIRONMENTS:
                    continue
                print("testing: ", domain, task)
                verify_reset_and_step(domain, task)

    def test_serialize_deserialize(self):
        domain, task = 'Swimmer', 'v3'
        env_kwargs = {
            'forward_reward_weight': 0,
            'ctrl_cost_weight': 0,
            'reset_noise_scale': 0,
            'exclude_current_positions_from_observation': False,
        }
        env1 = self.create_adapter(domain=domain, task=task, **env_kwargs)
        env1.reset()

        env2 = pickle.loads(pickle.dumps(env1))

        self.assertEqual(env1.observation_keys, env2.observation_keys)
        for key, value in env_kwargs.items():
            self.assertEqual(getattr(env1.unwrapped, f'_{key}'), value)
            self.assertEqual(getattr(env2.unwrapped, f'_{key}'), value)

        domain, task = 'HandReach', 'v0'
        gym_adapter_kwargs = {
            'observation_keys': ('observation', 'desired_goal')
        }
        env_kwargs = {
            'distance_threshold': 0.123123,
            'reward_type': 'dense',
        }
        env1 = self.create_adapter(
            domain=domain, task=task, **gym_adapter_kwargs, **env_kwargs)
        env1.reset()
        env2 = env1.copy()

        for key, value in gym_adapter_kwargs.items():
            self.assertEqual(getattr(env1, key), value)
            self.assertEqual(getattr(env2, key), value)

        for key, value in env_kwargs.items():
            self.assertEqual(getattr(env1.unwrapped, key), value)
            self.assertEqual(getattr(env2.unwrapped, key), value)

    def test_copy_environments(self):
        domain, task = 'Swimmer', 'v3'
        env_kwargs = {
            'forward_reward_weight': 0,
            'ctrl_cost_weight': 0,
            'reset_noise_scale': 0,
            'exclude_current_positions_from_observation': False,
        }
        env1 = self.create_adapter(domain=domain, task=task, **env_kwargs)
        env1.reset()
        env2 = env1.copy()

        self.assertEqual(env1.observation_keys, env2.observation_keys)
        for key, value in env_kwargs.items():
            self.assertEqual(getattr(env1.unwrapped, f'_{key}'), value)
            self.assertEqual(getattr(env2.unwrapped, f'_{key}'), value)

        domain, task = 'HandReach', 'v0'
        gym_adapter_kwargs = {
            'observation_keys': ('observation', 'desired_goal')
        }
        env_kwargs = {
            'distance_threshold': 0.123123,
            'reward_type': 'dense',
        }
        env1 = self.create_adapter(
            domain=domain, task=task, **gym_adapter_kwargs, **env_kwargs)
        env1.reset()
        env2 = env1.copy()

        for key, value in gym_adapter_kwargs.items():
            self.assertEqual(getattr(env1, key), value)
            self.assertEqual(getattr(env2, key), value)

        for key, value in env_kwargs.items():
            self.assertEqual(getattr(env1.unwrapped, key), value)
            self.assertEqual(getattr(env2.unwrapped, key), value)

    def test_environment_kwargs(self):
        env_kwargs = {
            'forward_reward_weight': 0.0,
            'ctrl_cost_weight': 0.0,
            'reset_noise_scale': 0.0,
        }

        env = GymAdapter(
            domain='Swimmer', task='Parameterizable-v3', **env_kwargs)

        observation1, reward, done, info = env.step(env.action_space.sample())

        self.assertAlmostEqual(reward, 0.0)

        for key, expected_value in env_kwargs.items():
            actual_value = getattr(env.unwrapped, f'_{key}')
            self.assertEqual(actual_value, expected_value)

    def test_render_rgb_array(self):
        env = self.create_adapter()
        with self.assertRaisesRegex(
                RuntimeError, "Failed to initialize OpenGL"):
            result = env.render(mode='rgb_array')
            self.assertIsInstance(result, np.ndarray)


if __name__ == '__main__':
    unittest.main()
