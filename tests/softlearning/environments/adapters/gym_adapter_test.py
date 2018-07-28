import unittest

from .softlearning_env_test import TestAdapterClass
from softlearning.environments.adapters.gym_adapter import (
    GYM_ENVIRONMENTS,
    GymAdapter)


class TestGymAdapter(unittest.TestCase, TestAdapterClass):
    ENVIRONMENTS = GYM_ENVIRONMENTS
    EXPECTED_ENVIRONMENTS = [
        '<GymAdapter(domain=swimmer, task=default) <<SwimmerEnv<Swimmer-v2>>>>',

        '<GymAdapter(domain=ant, task=default) <<AntEnv<Ant-v2>>>>',

        '<GymAdapter(domain=humanoid, task=default) '
        '<<HumanoidEnv<Humanoid-v2>>>>',

        '<GymAdapter(domain=humanoid, task=standup) '
        '<<HumanoidStandupEnv<HumanoidStandup-v2>>>>',

        '<GymAdapter(domain=hopper, task=default) <<HopperEnv<Hopper-v2>>>>',

        '<GymAdapter(domain=half-cheetah, task=default) '
        '<<HalfCheetahEnv<HalfCheetah-v2>>>>',

        '<GymAdapter(domain=walker, task=default) <<Walker2dEnv<Walker2d-v2>>>>'
    ]

    def create_adapter(self, domain='swimmer', task='default'):
        return GymAdapter(domain=domain, task=task)

    def test___str__(self):
        env = self.create_adapter()
        result = str(env)
        self.assertEqual(
            result,
            ('<GymAdapter(domain=swimmer, task=default)'
             ' <<SwimmerEnv<Swimmer-v2>>>>'))


if __name__ == '__main__':
    unittest.main()
