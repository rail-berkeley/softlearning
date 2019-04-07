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


if __name__ == '__main__':
    unittest.main()
