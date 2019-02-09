import unittest

from .softlearning_env_test import AdapterTestClass
from softlearning.environments.adapters.gym_adapter import (
    GymAdapter, CUSTOM_GYM_ENVIRONMENTS)


class TestGymAdapter(unittest.TestCase, AdapterTestClass):
    # TODO(hartikainen): This is a terrible way of testing the envs.
    # All the envs should be tested independently.

    def create_adapter(self, domain='Swimmer', task='v2', *args, **kwargs):
        return GymAdapter(domain, task, *args, **kwargs)

    def test_environments(self):
        # Make sure that all the environments are creatable
        SKIP_ENVIRONMENTS = (
            ('Pusher2d', 'ImageDefault-v0'),
            ('Pusher2d', 'ImageReach-v0'),
            ('Pusher2d', 'BlindReach-v0'))

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

    def test_environment_kwargs(self):
        env_kwargs = {
            'forward_reward_weight': 0.0,
            'ctrl_cost_weight': 0.0,
            'reset_noise_scale': 0.0,
        }

        env = GymAdapter(
            domain='Swimmer', task='Parameterizable-v0', **env_kwargs)

        observation1, reward, done, info = env.step(env.action_space.sample())

        self.assertAlmostEqual(reward, 0.0)

        for key, expected_value in env_kwargs.items():
            actual_value = getattr(env.unwrapped, f'_{key}')
            self.assertEqual(actual_value, expected_value)


if __name__ == '__main__':
    unittest.main()
