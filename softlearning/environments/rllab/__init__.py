"""Custom rllab environments.

Every class inside this module should extend a rllab.Env class. The file
structure should be similar to gym.envs file structure, e.g. if you're
implementing a mujoco env, you would implement it under rllab.mujoco submodule.
"""

from .multigoal import MultiGoalEnv
from .delayed_env import DelayedEnv
from .multi_direction_env import (
    MultiDirectionSwimmerEnv,
    MultiDirectionAntEnv,
    MultiDirectionHumanoidEnv)

from .cross_maze_ant_env import CrossMazeAntEnv
from .hierarchy_proxy_env import HierarchyProxyEnv
from .meta_env import FixedOptionEnv
