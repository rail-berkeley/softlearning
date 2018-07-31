import unittest

from .softlearning_env_test import TestAdapterClass
from softlearning.environments.adapters.rllab_adapter import (
    RLLAB_ENVIRONMENTS,
    RllabAdapter)


class TestRllabAdapter(unittest.TestCase, TestAdapterClass):
    ENVIRONMENTS = RLLAB_ENVIRONMENTS
    EXPECTED_ENVIRONMENTS = [
        '<RllabAdapter(domain=swimmer, task=default) '
        '<<rllab.envs.mujoco.swimmer_env.SwimmerEnv object at 0x10a241c18>>>',

        '<RllabAdapter(domain=swimmer, task=multi-direction) '
        '<<softlearning.environments.rllab.multi_direction_env.MultiDirectionSwimmerEnv '
        'object at 0x10a241c18>>>',

        '<RllabAdapter(domain=ant, task=default) <<rllab.envs.mujoco.ant_env.AntEnv '
        'object at 0x10a241c18>>>',

        '<RllabAdapter(domain=ant, task=multi-direction) '
        '<<softlearning.environments.rllab.multi_direction_env.MultiDirectionAntEnv '
        'object at 0x10a241c18>>>',

        '<RllabAdapter(domain=ant, task=cross-maze) '
        '<<softlearning.environments.rllab.cross_maze_ant_env.CrossMazeAntEnv object '
        'at 0x10a241c18>>>',

        '<RllabAdapter(domain=humanoid, task=default) '
        '<<rllab.envs.mujoco.humanoid_env.HumanoidEnv object at 0x10a2498d0>>>',

        '<RllabAdapter(domain=humanoid, task=multi-direction) '
        '<<softlearning.environments.rllab.multi_direction_env.MultiDirectionHumanoidEnv '
        'object at 0x10a2498d0>>>'
    ]

    def create_adapter(self, domain='swimmer', task='default'):
        return RllabAdapter(domain=domain, task=task)

    def test___str__(self):
        env = self.create_adapter()
        result = str(env)
        self.assertIn((
            '<RllabAdapter(domain=swimmer, task=default)'
            ' <<rllab.envs.mujoco.swimmer_env.SwimmerEnv'
            ' object at '),
            result)


if __name__ == '__main__':
    unittest.main()
