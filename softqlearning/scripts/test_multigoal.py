""" Example script to perform soft Q-learning in the multigoal environment. """
import os

import numpy as np

from rllab import config
from rllab.misc import logger
from rllab.envs.normalized_env import normalize
from rllab.tf.envs.base import TfEnv

from softqlearning.algos.softqlearning import SoftQLearning
from softqlearning.core.kernel import AdaptiveIsotropicGaussianKernel
from softqlearning.core.nn import NeuralNetwork, StochasticNeuralNetwork
from softqlearning.envs.multi_goal_env import MultiGoalEnv

tabular_log_file = os.path.join(config.LOG_DIR, 'tests', 'multigoal',
                                'eval.log')
snapshot_dir = os.path.join(config.LOG_DIR, 'tests', 'multigoal')
logger.add_tabular_output(tabular_log_file)
logger.set_snapshot_dir(snapshot_dir)


def test():

    env = TfEnv(normalize(MultiGoalEnv()))

    base_kwargs = dict(
        epoch_length=100,
        min_pool_size=100,
        n_epochs=100,
        max_path_length=30,
        batch_size=64,
        scale_reward=0.1,
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
    )

    qf_kwargs = dict(
        layer_sizes=(100, 100, 1),
        output_nonlinearity=None,
    )

    policy_kwargs = dict(
        layer_sizes=(100, 100, env.action_dim),
        output_nonlinearity=None,
    )

    algorithm = SoftQLearning(
        base_kwargs=base_kwargs,
        env=env,

        kernel_class=AdaptiveIsotropicGaussianKernel,
        kernel_n_particles=32,
        kernel_update_ratio=0.5,

        qf_class=NeuralNetwork,
        qf_kwargs=qf_kwargs,
        qf_target_n_particles=16,
        qf_lr=0.001,
        qf_target_update_interval=1000,

        policy_class=StochasticNeuralNetwork,
        policy_kwargs=policy_kwargs,
        policy_lr=0.001,

        discount=0.99,
        alpha=1,

        n_eval_episodes=10,
        q_plot_settings=q_plot_settings,
        env_plot_settings=env_plot_settings,
    )
    algorithm.train()

if __name__ == "__main__":
    test()
