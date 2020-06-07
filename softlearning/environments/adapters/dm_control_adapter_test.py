import pickle
import unittest

import numpy as np
from gym import spaces
import pytest

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

    def test_rescale_action(self):
        environment_kwargs = {
            'domain': 'quadruped',
            'task': 'run',
        }
        environment = DmControlAdapter(**environment_kwargs, rescale_action_range=None)
        new_low, new_high = -1.0, 1.0

        assert isinstance(environment.action_space, spaces.Box)
        assert np.any(environment.action_space.low != new_low)
        assert np.any(environment.action_space.high != new_high)

        rescaled_environment = DmControlAdapter(
            **environment_kwargs, rescale_action_range=(new_low, new_high))

        np.testing.assert_allclose(
            rescaled_environment.action_space.low, new_low)
        np.testing.assert_allclose(
            rescaled_environment.action_space.high, new_high)

    def test_rescale_observation_raises_exception(self):
        environment_kwargs = {
            'domain': 'quadruped',
            'task': 'run',
            'rescale_observation_range': (-1.0, 1.0),
        }
        with pytest.raises(
                NotImplementedError, match=r"Observation rescaling .*"):
            environment = DmControlAdapter(**environment_kwargs)


if __name__ == '__main__':
    unittest.main()
