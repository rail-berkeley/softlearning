import pickle
import unittest

import numpy as np

from .softlearning_env_test import AdapterTestClass
from softlearning.environments.adapters.robosuite_adapter import (
    RobosuiteAdapter)


class TestRobosuiteAdapter(unittest.TestCase, AdapterTestClass):
    def create_adapter(self, domain='Sawyer', task='Lift', *args, **kwargs):
        return RobosuiteAdapter(
            domain,
            task,
            *args,
            **{
                'has_renderer': False,
                'has_offscreen_renderer': False,
                'use_camera_obs': False,
                **kwargs
            })

    def test_environments(self):
        # Make sure that all the environments are creatable
        TEST_ENVIRONMENTS = [('Sawyer', 'Lift')]

        def verify_reset_and_step(domain, task):
            env = RobosuiteAdapter(
                domain=domain,
                task=task,
                has_renderer=True,
                has_offscreen_renderer=True,
                use_camera_obs=False)
            env.reset()
            env.step(env.action_space.sample())

        for domain, task in TEST_ENVIRONMENTS:
            verify_reset_and_step(domain, task)

    def test_serialize_deserialize(self):
        domain, task = 'Sawyer', 'Lift'
        env_kwargs = {
            'has_renderer': False,
            'has_offscreen_renderer': False,
            'use_camera_obs': False,
            'reward_shaping': True,
        }
        env1 = self.create_adapter(domain=domain, task=task, **env_kwargs)
        env1.reset()

        env2 = pickle.loads(pickle.dumps(env1))

        self.assertEqual(env1.observation_keys, env2.observation_keys)
        for key, value in env_kwargs.items():
            self.assertEqual(getattr(env1.unwrapped, f'{key}'), value)
            self.assertEqual(getattr(env2.unwrapped, f'{key}'), value)

    def test_copy_environments(self):
        domain, task = 'Sawyer', 'Lift'
        env_kwargs = {
            "gripper_type": "TwoFingerGripper",
            "table_full_size": (0.8, 0.8, 0.8)
        }
        env1 = self.create_adapter(domain=domain, task=task, **env_kwargs)
        env1.reset()
        env2 = env1.copy()

        self.assertEqual(env1.observation_keys, env2.observation_keys)
        for key, value in env_kwargs.items():
            self.assertEqual(getattr(env1.unwrapped, key), value)
            self.assertEqual(getattr(env2.unwrapped, key), value)

        domain, task = 'Sawyer', 'Lift'
        robosuite_adapter_kwargs = {
            'observation_keys': ('joint_pos', 'joint_vel')
        }
        env_kwargs = {
            "gripper_type": "TwoFingerGripper",
            "table_full_size": (0.8, 0.8, 0.8)
        }
        env1 = self.create_adapter(
            domain=domain, task=task, **robosuite_adapter_kwargs, **env_kwargs)
        env1.reset()
        env2 = env1.copy()

        for key, value in robosuite_adapter_kwargs.items():
            self.assertEqual(getattr(env1, key), value)
            self.assertEqual(getattr(env2, key), value)

        for key, value in env_kwargs.items():
            self.assertEqual(getattr(env1.unwrapped, key), value)
            self.assertEqual(getattr(env2.unwrapped, key), value)

    def test_fails_with_invalid_environment_kwargs(self):
        domain, task = 'Sawyer', 'Lift'
        robosuite_adapter_kwargs = {
            'observation_keys': ('joint_pos', 'invalid_key')
        }
        with self.assertRaises(AssertionError):
            env = self.create_adapter(
                domain=domain, task=task, **robosuite_adapter_kwargs)

    def test_environment_kwargs(self):
        env_kwargs = {
            "has_renderer": False,
            "has_offscreen_renderer": False,
            "use_camera_obs": False,
            "control_freq": 10,
            "horizon": 1000
        }

        env = RobosuiteAdapter(
            domain='Sawyer', task='Lift', **env_kwargs)

        observation1, reward, done, info = env.step(env.action_space.sample())

        self.assertAlmostEqual(reward, 0.0)

        for key, expected_value in env_kwargs.items():
            actual_value = getattr(env.unwrapped, key)
            self.assertEqual(actual_value, expected_value)

    def test_render_rgb_array(self):
        env = self.create_adapter(
            has_renderer=False,
            has_offscreen_renderer=True)
        env.render(mode='rgb_array', camera_id=0, width=32, height=32)

    def test_render_human(self):
        env = self.create_adapter(
            has_renderer=True,
            has_offscreen_renderer=False)
        env.render(mode='human')

    def test_fails_with_unnormalized_action_spec(self):
        from robosuite.environments.sawyer_lift import SawyerLift

        class UnnormalizedEnv(SawyerLift):
            @property
            def dof(self):
                return 5

            @property
            def action_spec(self):
                low, high = np.ones(self.dof) * -2.0, np.ones(self.dof) * 2.0
                return low, high

        env = UnnormalizedEnv(
                has_renderer=False,
                has_offscreen_renderer=False,
                use_camera_obs=False)
        with self.assertRaises(AssertionError):
            adapter = RobosuiteAdapter(domain=None, task=None, env=env)


if __name__ == '__main__':
    unittest.main()
