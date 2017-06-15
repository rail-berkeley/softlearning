from rllab.envs.gym_env import GymEnv, NoVideoSchedule, CappedCubicVideoSchedule, convert_gym_space
from rllab.core.serializable import Serializable
from sandbox.rocky.tf.spaces.box import Box
import cv2
import numpy as np
import gym
import gym.wrappers
import gym.envs
import gym.spaces
from gym.monitoring import monitor_manager
import os
from rllab.envs.base import Env, Step
from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
from rllab.misc import logger
import logging

class CroppedGymEnv(GymEnv):
    def __init__(self, env_name, record_video=True, video_schedule=None, log_dir=None, record_log=True,
                 force_reset=False, screen_width=84, screen_height=84):
        if log_dir is None:
            if logger.get_snapshot_dir() is None:
                logger.log("Warning: skipping Gym environment monitoring since snapshot_dir not configured.")
            else:
                log_dir = os.path.join(logger.get_snapshot_dir(), "gym_log")
        Serializable.quick_init(self, locals())

        env = gym.envs.make(env_name)
        if 'Doom' in env_name:
            from ppaquette_gym_doom.wrappers.action_space import ToDiscrete
            wrapper = ToDiscrete('minimal')
            env = wrapper(env)

        self.env = env
        self.env_id = env.spec.id

        monitor_manager.logger.setLevel(logging.WARNING)

        assert not (not record_log and record_video)

        if log_dir is None or record_log is False:
            self.monitoring = False
        else:
            if not record_video:
                video_schedule = NoVideoSchedule()
            else:
                if video_schedule is None:
                    video_schedule = CappedCubicVideoSchedule()
            self.env = gym.wrappers.Monitor(self.env, log_dir, video_callable=video_schedule, force=True)
            self.monitoring = True

        self._observation_space = convert_gym_space(env.observation_space)
        self._action_space = convert_gym_space(env.action_space)
        self._horizon = env.spec.timestep_limit
        self._log_dir = log_dir
        self._force_reset = force_reset
        self.screen_width = screen_width
        self.screen_height = screen_height
        self._observation_space = Box(low=0, high=1, shape=(screen_width, screen_height, 1))


    def reshape(self, obs):
        return np.expand_dims(cv2.resize(cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY), (self.screen_width, self.screen_height)), -1)

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        next_obs = self.reshape(next_obs)

        return Step(next_obs, reward, done, **info)

    def reset(self):
        if self._force_reset and self.monitoring:
            recorder = self.env._monitor.stats_recorder
            if recorder is not None:
                recorder.done = True
        return self.reshape(self.env.reset())
