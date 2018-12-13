import unittest

from .softlearning_env_test import AdapterTestClass
from softlearning.environments.adapters.gym_adapter import (
    GYM_ENVIRONMENTS,
    GymAdapter)


class TestGymAdapter(unittest.TestCase, AdapterTestClass):
    # TODO(hartikainen): This is a terrible way of testing the envs.
    # All the envs should be tested independently.

    ENVIRONMENTS = GYM_ENVIRONMENTS
    EXPECTED_ENVIRONMENTS = (
        '<GymAdapter(domain=swimmer, task=default) '
        '<<NormalizeActionWrapper<SwimmerEnv<Swimmer-v2>>>>>',
        '<GymAdapter(domain=swimmer, task=custom-default) '
        '<<NormalizeActionWrapper<SwimmerEnv instance>>>>',
        '<GymAdapter(domain=ant, task=default) '
        '<<NormalizeActionWrapper<AntEnv<Ant-v2>>>>>',
        '<GymAdapter(domain=ant, task=custom-default) <<NormalizeActionWrapper<AntEnv '
        'instance>>>>',
        '<GymAdapter(domain=humanoid, task=default) '
        '<<NormalizeActionWrapper<HumanoidEnv<Humanoid-v2>>>>>',
        '<GymAdapter(domain=humanoid, task=standup) '
        '<<NormalizeActionWrapper<HumanoidStandupEnv<HumanoidStandup-v2>>>>>',
        '<GymAdapter(domain=humanoid, task=custom-default) '
        '<<NormalizeActionWrapper<HumanoidEnv instance>>>>',
        '<GymAdapter(domain=hopper, task=default) '
        '<<NormalizeActionWrapper<HopperEnv<Hopper-v2>>>>>',
        '<GymAdapter(domain=hopper, task=custom-default) '
        '<<NormalizeActionWrapper<HopperEnv instance>>>>',
        '<GymAdapter(domain=half-cheetah, task=default) '
        '<<NormalizeActionWrapper<HalfCheetahEnv<HalfCheetah-v2>>>>>',
        '<GymAdapter(domain=walker, task=default) '
        '<<NormalizeActionWrapper<Walker2dEnv<Walker2d-v2>>>>>',
        '<GymAdapter(domain=walker, task=custom-default) '
        '<<NormalizeActionWrapper<Walker2dEnv instance>>>>',
        '<GymAdapter(domain=pusher-2d, task=default) '
        '<<NormalizeActionWrapper<Pusher2dEnv instance>>>>',
        '<GymAdapter(domain=pusher-2d, task=default-reach) '
        '<<NormalizeActionWrapper<ForkReacherEnv instance>>>>',
        '<GymAdapter(domain=Point2DEnv, task=default) '
        '<<NormalizeActionWrapper<Point2DEnv instance>>>>',
        '<GymAdapter(domain=Point2DEnv, task=wall) '
        '<<NormalizeActionWrapper<Point2DWallEnv instance>>>>',
        '<GymAdapter(domain=HandManipulatePen, task=v0) '
        '<<NormalizeActionWrapper<HandPenEnv<HandManipulatePen-v0>>>>>',
        '<GymAdapter(domain=HandManipulatePen, task=Dense-v0) '
        '<<NormalizeActionWrapper<HandPenEnv<HandManipulatePenDense-v0>>>>>',
        '<GymAdapter(domain=HandManipulatePen, task=default) '
        '<<NormalizeActionWrapper<HandPenEnv<HandManipulatePen-v0>>>>>',
        '<GymAdapter(domain=HandManipulateEgg, task=v0) '
        '<<NormalizeActionWrapper<HandEggEnv<HandManipulateEgg-v0>>>>>',
        '<GymAdapter(domain=HandManipulateEgg, task=Dense-v0) '
        '<<NormalizeActionWrapper<HandEggEnv<HandManipulateEggDense-v0>>>>>',
        '<GymAdapter(domain=HandManipulateEgg, task=default) '
        '<<NormalizeActionWrapper<HandEggEnv<HandManipulateEgg-v0>>>>>',
        '<GymAdapter(domain=HandManipulateBlock, task=v0) '
        '<<NormalizeActionWrapper<HandBlockEnv<HandManipulateBlock-v0>>>>>',
        '<GymAdapter(domain=HandManipulateBlock, task=Dense-v0) '
        '<<NormalizeActionWrapper<HandBlockEnv<HandManipulateBlockDense-v0>>>>>',
        '<GymAdapter(domain=HandManipulateBlock, task=default) '
        '<<NormalizeActionWrapper<HandBlockEnv<HandManipulateBlock-v0>>>>>',
        '<GymAdapter(domain=HandReach, task=v0) '
        '<<NormalizeActionWrapper<HandReachEnv<HandReach-v0>>>>>',
        '<GymAdapter(domain=HandReach, task=Dense-v0) '
        '<<NormalizeActionWrapper<HandReachEnv<HandReachDense-v0>>>>>',
        '<GymAdapter(domain=HandReach, task=default) '
        '<<NormalizeActionWrapper<HandReachEnv<HandReach-v0>>>>>',
        '<GymAdapter(domain=InvertedDoublePendulum, task=v2) '
        '<<NormalizeActionWrapper<InvertedDoublePendulumEnv<InvertedDoublePendulum-v2>>>>>',
        '<GymAdapter(domain=Reacher, task=v2) '
        '<<NormalizeActionWrapper<ReacherEnv<Reacher-v2>>>>>',
        '<GymAdapter(domain=InvertedPendulum, task=v2) '
        '<<NormalizeActionWrapper<InvertedPendulumEnv<InvertedPendulum-v2>>>>>',
        '<GymAdapter(domain=MultiGoal, task=default) '
        '<<NormalizeActionWrapper<MultiGoalEnv instance>>>>'
    )

    def create_adapter(self, domain='swimmer', task='default'):
        return GymAdapter(domain=domain, task=task)

    def test___str__(self):
        env = self.create_adapter()
        result = str(env)
        self.assertEqual(
            result,
            ('<GymAdapter(domain=swimmer, task=default)'
             ' <<NormalizeActionWrapper<SwimmerEnv<Swimmer-v2>>>>>'))


if __name__ == '__main__':
    unittest.main()
