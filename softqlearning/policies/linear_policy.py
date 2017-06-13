import numpy as np

from rllab.misc.overrides import overrides
from rllab.policies.base import Policy
from rllab.core.serializable import Serializable


class LinearPolicy(Policy, Serializable):
    def __init__(self, env_spec, waypoints, time_inds):
        Serializable.quick_init(self, locals())
        super().__init__(env_spec)
        self._counter = 0

        self._waypoints = waypoints
        self._time_inds = np.array(time_inds)

    @overrides
    def get_params_internal(self, **tags):
        return list()

    @overrides
    def get_action(self, observation):
        self._counter += 1

        waypoint_index = np.where(self._counter <= self._time_inds)[0][0]
        waypoint = self._waypoints[waypoint_index]

        qpos = observation[:7]
        diff = waypoint - qpos
        action = 0.3*diff

        return action, None

    @overrides
    def reset(self):
        self._counter = 0


# TODO: Some test stuff below, to be removed.
def test():
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    from softqlearning.misc.sampler import rollout
    from softqlearning.envs.mujoco.pusher import PusherEnv
    from rllab.envs.normalized_env import normalize
    from rllab.tf.envs.base import TfEnv

    # from gym.envs.mujoco.pusher import PusherEnv

    object_pos = (-0.2, 0.2)
    target_pos = (0.1, 0)

    waypoints = _get_waypoints2()

    env = TfEnv(normalize(PusherEnv(cylinder_pos=object_pos,
                                    target_pos=target_pos,
                                    guide_cost_coeff=0.0,
                                    ctrl_cost_coeff=0.1,
                                    tgt_cost_coeff=1.0)))
    policy = LinearPolicy(env.spec, waypoints, [20, 99999])

    while True:
        path = rollout(env, policy, path_length=100, render=True, speedup=10)
        print(path['rewards'].sum())


def _get_waypoints2():
    pt1 = np.array([
        0.00,
        1.00,
        0.00,
        -1.00,
        0.00,
        0.00,
        0.00
    ])

    pt2 = np.array([
        0.85,
        1.00,
        0.00,
        -1.00,
        0.00,
        0.00,
        0.00
    ])

    return pt1, pt2


def _get_waypoints():
    pt1 = np.array([
        0.85,
        1.00,
        0.00,
        -1.00,
        0.00,
        0.00,
        0.00
    ])

    pt2 = np.array([
       0.85,
       0.6,
       0.00,
       -0.60,
       0.00,
       0.00,
       0.00
    ])

    return pt1, pt2


# joint limits:
# 0: -2.2854 1.714602
# 1: -0.5236 1.3963
# 2: -1.5 1.7
# 3: -2.3213 0
# 4: -1.5 1.5
# 5: -1.094 0
# 6: -1.5 1.5


def set_pose():
    from softqlearning.envs.mujoco.pusher import PusherEnv
    from rllab.envs.normalized_env import normalize
    from rllab.tf.envs.base import TfEnv

    env = TfEnv(normalize(PusherEnv()))

    qpos = env._base_env.init_qpos
    qvel = env._base_env.init_qvel

    np.savetxt('qpos.txt', qpos)

    while True:
        try:
            qpos = np.loadtxt('qpos.txt')
        except:
            pass
        env._base_env.set_state(qpos, qvel)
        env.render()
        # try:
        # except KeyboardInterrupt:
        #     import ipdb; ipdb.set_trace()
        #     env._base_env.set_state(qpos, qvel)


if __name__ == '__main__':
    test()
    # set_pose()

