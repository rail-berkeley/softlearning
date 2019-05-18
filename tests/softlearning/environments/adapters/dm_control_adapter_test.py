import pickle
import unittest

from .softlearning_env_test import AdapterTestClass
from softlearning.environments.adapters.dm_control_adapter import (
    DmControlAdapter)


class TestDmControlAdapter(unittest.TestCase, AdapterTestClass):
    def create_adapter(self,
                       domain='cartpole',
                       task='swingup',
                       *args,
                       **kwargs):
        return DmControlAdapter(domain, task, *args, **kwargs)

    def test_environments(self):
        # Make sure that all the environments are creatable
        TEST_ENVIRONMENTS = (
            ('cartpole', 'swingup'),
        )

        def verify_reset_and_step(domain, task):
            env = DmControlAdapter(domain=domain, task=task)
            env.reset()
            env.step(env.action_space.sample())

        for domain, task in TEST_ENVIRONMENTS:
            print("testing: ", domain, task)
            verify_reset_and_step(domain, task)

    def test_render_human(self):
        env = self.create_adapter()
        with self.assertRaises(NotImplementedError):
            result = env.render(mode='human')
            self.assertIsNone(result)

    def test_environment_kwargs(self):
        # TODO(hartikainen): Figure this out later.
        pass

    def test_serialize_deserialize(self):
        domain, task = 'hopper', 'hop'
        env_kwargs = {
            'environment_kwargs': {
                'flat_observation': True,
            }
        }
        env1 = self.create_adapter(domain=domain, task=task, **env_kwargs)
        env1.reset()

        env2 = pickle.loads(pickle.dumps(env1))

        self.assertEqual(env1.observation_keys, env2.observation_keys)
        for key, value in env_kwargs['environment_kwargs'].items():
            self.assertEqual(getattr(env1.unwrapped, f'_{key}'), value)
            self.assertEqual(getattr(env2.unwrapped, f'_{key}'), value)

    def test_copy_environments(self):
        domain, task = 'cartpole', 'swingup'
        env_kwargs = {
            'environment_kwargs': {
                'flat_observation': False,
            }
        }
        env1 = self.create_adapter(domain=domain, task=task, **env_kwargs)
        env1.reset()
        env2 = env1.copy()

        self.assertEqual(env1.observation_keys, env2.observation_keys)
        for key, value in env_kwargs['environment_kwargs'].items():
            self.assertEqual(getattr(env1.unwrapped, f'_{key}'), value)
            self.assertEqual(getattr(env2.unwrapped, f'_{key}'), value)


if __name__ == '__main__':
    unittest.main()
