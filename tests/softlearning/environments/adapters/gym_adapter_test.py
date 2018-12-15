import unittest

from .softlearning_env_test import AdapterTestClass
from softlearning.environments.adapters.gym_adapter import (
    GYM_ENVIRONMENTS,
    GymAdapter)


class TestGymAdapter(unittest.TestCase, AdapterTestClass):
    # TODO(hartikainen): This is a terrible way of testing the envs.
    # All the envs should be tested independently.

    ENVIRONMENTS = GYM_ENVIRONMENTS
    EXPECTED_ENVIRONMENTS = [
        '<GymAdapter(domain=Swimmer, task=v2) '
        '<<NormalizeActionWrapper<SwimmerEnv<Swimmer-v2>>>>>',
        '<GymAdapter(domain=Swimmer, task=CustomDefault) '
        '<<NormalizeActionWrapper<SwimmerEnv instance>>>>',
        '<GymAdapter(domain=Swimmer, task=Default) '
        '<<NormalizeActionWrapper<SwimmerEnv<Swimmer-v2>>>>>',
        '<GymAdapter(domain=Ant, task=v2) <<NormalizeActionWrapper<AntEnv<Ant-v2>>>>>',
        '<GymAdapter(domain=Ant, task=Custom) <<NormalizeActionWrapper<AntEnv '
        'instance>>>>',
        '<GymAdapter(domain=Ant, task=Default) '
        '<<NormalizeActionWrapper<AntEnv<Ant-v2>>>>>',
        '<GymAdapter(domain=Humanoid, task=v2) '
        '<<NormalizeActionWrapper<HumanoidEnv<Humanoid-v2>>>>>',
        '<GymAdapter(domain=Humanoid, task=Standup-v2) '
        '<<NormalizeActionWrapper<HumanoidStandupEnv<HumanoidStandup-v2>>>>>',
        '<GymAdapter(domain=Humanoid, task=Custom) '
        '<<NormalizeActionWrapper<HumanoidEnv instance>>>>',
        '<GymAdapter(domain=Humanoid, task=Default) '
        '<<NormalizeActionWrapper<HumanoidEnv<Humanoid-v2>>>>>',
        '<GymAdapter(domain=Hopper, task=v2) '
        '<<NormalizeActionWrapper<HopperEnv<Hopper-v2>>>>>',
        '<GymAdapter(domain=Hopper, task=Custom) <<NormalizeActionWrapper<HopperEnv '
        'instance>>>>',
        '<GymAdapter(domain=Hopper, task=Default) '
        '<<NormalizeActionWrapper<HopperEnv<Hopper-v2>>>>>',
        '<GymAdapter(domain=HalfCheetah, task=v2) '
        '<<NormalizeActionWrapper<HalfCheetahEnv<HalfCheetah-v2>>>>>',
        '<GymAdapter(domain=HalfCheetah, task=Default) '
        '<<NormalizeActionWrapper<HalfCheetahEnv<HalfCheetah-v2>>>>>',
        '<GymAdapter(domain=Walker, task=v2) '
        '<<NormalizeActionWrapper<Walker2dEnv<Walker2d-v2>>>>>',
        '<GymAdapter(domain=Walker, task=Custom) <<NormalizeActionWrapper<Walker2dEnv '
        'instance>>>>',
        '<GymAdapter(domain=Walker, task=Default) '
        '<<NormalizeActionWrapper<Walker2dEnv<Walker2d-v2>>>>>',
        '<GymAdapter(domain=Pusher2d, task=Default) '
        '<<NormalizeActionWrapper<Pusher2dEnv instance>>>>',
        '<GymAdapter(domain=Pusher2d, task=DefaultReach) '
        '<<NormalizeActionWrapper<ForkReacherEnv instance>>>>',
        '<GymAdapter(domain=Point2DEnv, task=Default) '
        '<<NormalizeActionWrapper<Point2DEnv instance>>>>',
        '<GymAdapter(domain=Point2DEnv, task=Wall) '
        '<<NormalizeActionWrapper<Point2DWallEnv instance>>>>',
        '<GymAdapter(domain=HandManipulatePen, task=v0) '
        '<<NormalizeActionWrapper<HandPenEnv<HandManipulatePen-v0>>>>>',
        '<GymAdapter(domain=HandManipulatePen, task=Dense-v0) '
        '<<NormalizeActionWrapper<HandPenEnv<HandManipulatePenDense-v0>>>>>',
        '<GymAdapter(domain=HandManipulatePen, task=Default) '
        '<<NormalizeActionWrapper<HandPenEnv<HandManipulatePen-v0>>>>>',
        '<GymAdapter(domain=HandManipulateEgg, task=v0) '
        '<<NormalizeActionWrapper<HandEggEnv<HandManipulateEgg-v0>>>>>',
        '<GymAdapter(domain=HandManipulateEgg, task=Dense-v0) '
        '<<NormalizeActionWrapper<HandEggEnv<HandManipulateEggDense-v0>>>>>',
        '<GymAdapter(domain=HandManipulateEgg, task=Default) '
        '<<NormalizeActionWrapper<HandEggEnv<HandManipulateEgg-v0>>>>>',
        '<GymAdapter(domain=HandManipulateBlock, task=v0) '
        '<<NormalizeActionWrapper<HandBlockEnv<HandManipulateBlock-v0>>>>>',
        '<GymAdapter(domain=HandManipulateBlock, task=Dense-v0) '
        '<<NormalizeActionWrapper<HandBlockEnv<HandManipulateBlockDense-v0>>>>>',
        '<GymAdapter(domain=HandManipulateBlock, task=Default) '
        '<<NormalizeActionWrapper<HandBlockEnv<HandManipulateBlock-v0>>>>>',
        '<GymAdapter(domain=HandReach, task=v0) '
        '<<NormalizeActionWrapper<HandReachEnv<HandReach-v0>>>>>',
        '<GymAdapter(domain=HandReach, task=Dense-v0) '
        '<<NormalizeActionWrapper<HandReachEnv<HandReachDense-v0>>>>>',
        '<GymAdapter(domain=HandReach, task=Default) '
        '<<NormalizeActionWrapper<HandReachEnv<HandReach-v0>>>>>',
        '<GymAdapter(domain=InvertedDoublePendulum, task=Default) '
        '<<NormalizeActionWrapper<InvertedDoublePendulumEnv<InvertedDoublePendulum-v2>>>>>',
        '<GymAdapter(domain=InvertedDoublePendulum, task=v2) '
        '<<NormalizeActionWrapper<InvertedDoublePendulumEnv<InvertedDoublePendulum-v2>>>>>',
        '<GymAdapter(domain=Reacher, task=Default) '
        '<<NormalizeActionWrapper<ReacherEnv<Reacher-v2>>>>>',
        '<GymAdapter(domain=Reacher, task=v2) '
        '<<NormalizeActionWrapper<ReacherEnv<Reacher-v2>>>>>',
        '<GymAdapter(domain=InvertedPendulum, task=Default) '
        '<<NormalizeActionWrapper<InvertedPendulumEnv<InvertedPendulum-v2>>>>>',
        '<GymAdapter(domain=InvertedPendulum, task=v2) '
        '<<NormalizeActionWrapper<InvertedPendulumEnv<InvertedPendulum-v2>>>>>',
        '<GymAdapter(domain=MultiGoal, task=Default) '
        '<<NormalizeActionWrapper<MultiGoalEnv instance>>>>',
    ]

    def create_adapter(self, domain='Swimmer', task='Default'):
        return GymAdapter(domain=domain, task=task)

    def test___str__(self):
        env = self.create_adapter()
        result = str(env)
        self.assertEqual(
            result,
            ('<GymAdapter(domain=Swimmer, task=Default)'
             ' <<NormalizeActionWrapper<SwimmerEnv<Swimmer-v2>>>>>'))


if __name__ == '__main__':
    unittest.main()
