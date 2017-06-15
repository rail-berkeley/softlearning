import datetime
import os
import os.path as osp
import random

import dateutil.tz
import numpy as np
import tensorflow as tf

from railrl.envs.env_utils import gym_env
from railrl.envs.memory.one_char_memory import OneCharMemory
from rllab import config
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.mujoco.ant_env import AntEnv
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.inverted_double_pendulum_env import \
    InvertedDoublePendulumEnv
from rllab.envs.normalized_env import normalize
from rllab.misc import logger


def get_env_settings(env_id="", normalize_env=True, gym_name="",
                     env_params=None):
    if env_params is None:
        env_params = {}

    if env_id == 'cart':
        env = CartpoleEnv()
        name = "Cartpole"
    elif env_id == 'cheetah':
        env = HalfCheetahEnv()
        name = "HalfCheetah"
    elif env_id == 'ant':
        env = AntEnv()
        name = "Ant"
    elif env_id == 'point':
        env = gym_env("OneDPoint-v0")
        name = "OneDPoint"
    elif env_id == 'reacher':
        env = gym_env("Reacher-v1")
        name = "Reacher"
    elif env_id == 'idp':
        env = InvertedDoublePendulumEnv()
        name = "InvertedDoublePendulum"
    elif env_id == 'ocm':
        env = OneCharMemory(**env_params)
        name = "OneCharMemory"
    elif env_id == 'gym':
        if gym_name == "":
            raise Exception("Must provide a gym name")
        env = gym_env(gym_name)
        name = gym_name
    else:
        raise Exception("Unknown env: {0}".format(env_id))
    if normalize_env:
        env = normalize(env)
        name += "-normalized"
    return dict(
        env=env,
        name=name,
        was_env_normalized=normalize_env,
    )


def run_experiment_here(
        experiment_function,
        exp_prefix="default",
        variant=None,
        exp_count=0,
        seed=0,
):
    """
    Run an experiment locally without any serialization.

    :param experiment_function: Function. `variant` will be passed in as its
    only argument.
    :param exp_prefix: Experiment prefix for the save file.
    :param variant: Dictionary passed in to `experiment_function`.
    :param exp_count: Experiment count. Should be unique across all
    experiments. Note that one experiment may correspond to multiple seeds,.
    :param seed: Seed used for this experiment.
    :return:
    """
    if variant is None:
        variant = {}
    setup_logger(
        exp_prefix=exp_prefix,
        variant=variant,
        exp_count=exp_count,
        seed=seed,
    )
    experiment_function(variant)
    reset_execution_environment()


now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')


def create_exp_name(exp_prefix="default", exp_count=0, seed=0):
    """
    Create a semi-unique experiment name that has a timestamp
    :param exp_prefix:
    :param exp_count:
    :param seed: Experiment seed
    :return:
    """
    return "%s_%s_%04d--s-%d" % (exp_prefix, timestamp, exp_count, seed)


def create_log_dir(exp_prefix="default", exp_count=0, seed=0):
    """
    Creates and returns a unique log directory.

    :param exp_prefix: All experiments with this prefix will have log
    directories be under this directory.
    :param exp_count: Different exp_counts will be in different directories.
    :param seed: Experiment seed
    :return:
    """
    exp_name = create_exp_name(exp_prefix=exp_prefix, exp_count=exp_count,
                               seed=seed)
    log_dir = osp.join(
        config.LOG_DIR,
        'local',
        exp_prefix.replace("_", "-"),
        exp_name,
    )
    if osp.exists(log_dir):
        raise Exception(
            "Log directory already exists. Will not overwrite: {0}".format(
                log_dir
            )
        )
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def setup_logger(
        exp_prefix=None,
        exp_count=0,
        seed=0,
        variant=None,
        log_dir=None,
        text_log_file="debug.log",
        variant_log_file="variant.json",
        tabular_log_file="progress.csv",
        snapshot_mode="last",
        log_tabular_only=False,
        snapshot_gap=1,
):
    """
    Set up logger to have some reasonable default settings.

    :param exp_prefix:
    :param exp_count:
    :param seed: Experiment seed
    :param variant:
    :param log_dir:
    :param text_log_file:
    :param variant_log_file:
    :param tabular_log_file:
    :param snapshot_mode:
    :param log_tabular_only:
    :param snapshot_gap:
    :return:
    """
    if log_dir is None:
        assert exp_prefix is not None
        log_dir = create_log_dir(exp_prefix, exp_count=exp_count, seed=seed)
    tabular_log_path = osp.join(log_dir, tabular_log_file)
    text_log_path = osp.join(log_dir, text_log_file)

    if variant is not None:
        variant_log_path = osp.join(log_dir, variant_log_file)
        logger.log_variant(variant_log_path, variant)

    logger.add_text_output(text_log_path)
    logger.add_tabular_output(tabular_log_path)
    logger.set_snapshot_dir(log_dir)
    logger.set_snapshot_mode(snapshot_mode)
    logger.set_snapshot_gap(snapshot_gap)
    logger.set_log_tabular_only(log_tabular_only)


def set_seed(seed):
    """
    Set the seed for all the possible random number generators.

    :param seed:
    :return: None
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def reset_execution_environment():
    """
    Call this between calls to separate experiments.
    :return:
    """
    tf.reset_default_graph()
    import importlib
    importlib.reload(logger)
