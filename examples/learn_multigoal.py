""" Example script to perform soft Q-learning in the multigoal environment. """
import os

import numpy as np

from rllab import config
from rllab.misc import logger
from rllab.envs.normalized_env import normalize

from softqlearning.algorithms import SQL
from softqlearning.misc.kernel import adaptive_isotropic_gaussian_kernel
from softqlearning.environments import MultiGoalEnv
from softqlearning.replay_buffers import SimpleReplayBuffer
from softqlearning.value_functions import NNQFunction
from softqlearning.misc.plotter import QFPolicyPlotter
from softqlearning.policies import StochasticNNPolicy

snapshot_dir = os.path.join(config.LOG_DIR, 'multigoal')
tabular_log_file = os.path.join(snapshot_dir, 'eval.log')
logger.add_tabular_output(tabular_log_file)
logger.set_snapshot_dir(snapshot_dir)


def test():

    env = normalize(MultiGoalEnv())

    pool = SimpleReplayBuffer(
        env_spec=env.spec,
        max_replay_buffer_size=1E6,
    )

    base_kwargs = dict(
        epoch_length=100,
        min_pool_size=100,
        n_epochs=100,
        max_path_length=30,
        batch_size=64,
        eval_render=True,
    )

    q_plot_settings = dict(
        xlim=(-1., 1.),
        ylim=(-1., 1.),
        obs_lst=np.array([[-2.5, 0.0],
                          [0.0, 0.0],
                          [2.5, 2.5]]),
        action_dims=(0, 1),
    )

    env_plot_settings = dict(
        xlim=(-7, 7),
        ylim=(-7, 7),
        title="paths",
        xlabel="x",
        ylabel="y",
    )

    M = 128
    qf = NNQFunction(
        env_spec=env.spec,
        hidden_layer_sizes=[M, M]
    )

    policy = StochasticNNPolicy(
        env.spec,
        hidden_layer_sizes=(M, M),
        squash=True
    )

    plotter = QFPolicyPlotter(
        qf=qf,
        policy=policy,
        obs_lst=np.array([[-2.5, 0.0],
                          [0.0, 0.0],
                          [2.5, 2.5]]),
        default_action=[np.nan, np.nan],
        n_samples=100
    )

    algorithm = SQL(
        base_kwargs=base_kwargs,
        env=env,
        pool=pool,
        qf=qf,
        policy=policy,
        plotter=plotter,

        kernel_fn=adaptive_isotropic_gaussian_kernel,
        kernel_n_particles=32,
        kernel_update_ratio=0.5,

        value_n_particles=16,
        qf_lr=0.001,
        td_target_update_interval=1000,

        policy_lr=0.001,

        discount=0.99,
        reward_scale=0.1,
    )
    algorithm.train()

if __name__ == "__main__":
    test()
