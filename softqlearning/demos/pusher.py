import numpy as np

from softqlearning.misc.replay_pool import SimpleReplayPool
from softqlearning.policies.linear_policy import LinearPolicy
from softqlearning.misc.sampler import rollout
from softqlearning.envs.mujoco.pusher import PusherEnv
from rllab.envs.normalized_env import normalize
from rllab.tf.envs.base import TfEnv


class PusherDemo(SimpleReplayPool):

    def __init__(self, env_kwargs, render=False):
        path_length = 100

        env = TfEnv(normalize(PusherEnv(**env_kwargs)))

        super().__init__(path_length + 1,
                         observation_dim=env.observation_space.flat_dim,
                         action_dim=env.action_space.flat_dim)

        # TODO: The instructions makes sense only for a particular set of
        # env_kwargs.
        instructions = _get_pusher_instructions()
        policy = LinearPolicy(env.spec, instructions)
        path = rollout(env, policy, path_length=path_length,
                       render=render, speedup=10)

        self.add_path(path['observations'],
                      path['actions'],
                      path['rewards'],
                      path['dones'],
                      path['last_obs'])


def _get_pusher_instructions():
    pt1 = np.array([0.00, 1.00, 0.00, -1.00, 0.00, 0.00, 0.00])
    pt2 = np.array([0.85, 1.00, 0.00, -1.00, 0.00, 0.00, 0.00])
    times = np.array([20, 99999])

    return dict(
        waypoints=(pt1, pt2),
        times=times
    )
