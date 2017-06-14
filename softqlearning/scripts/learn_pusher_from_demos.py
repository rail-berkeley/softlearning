"""
Example script that applies soft Q-learning to the pusher environment.
"""
import os
from rllab import config

from rllab.misc import logger
from rllab.envs.normalized_env import normalize
from rllab.tf.envs.base import TfEnv

from softqlearning.algos.softqlearning import SoftQLearning
from softqlearning.core.kernel import AdaptiveIsotropicGaussianKernel
from softqlearning.core.nn import NeuralNetwork, StochasticNeuralNetwork
from softqlearning.envs.mujoco.pusher import PusherEnv
from softqlearning.demos.pusher import PusherDemo

snapshot_dir = os.path.join(config.LOG_DIR, 'pusher')
tabular_log_file = os.path.join(snapshot_dir, 'eval.log')
logger.add_tabular_output(tabular_log_file)
logger.set_snapshot_dir(snapshot_dir)


def test():

    env = TfEnv(normalize(PusherEnv()))

    base_kwargs = dict(
        epoch_length=1000,
        min_pool_size=1000,
        n_epochs=200,
        max_path_length=100,
        batch_size=64,
        scale_reward=30,
        demo_pool=PusherDemo(),
        demo_ratio=0.1,
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
        kernel_n_particles=16,
        kernel_update_ratio=0.5,

        qf_class=NeuralNetwork,
        qf_kwargs=qf_kwargs,
        qf_target_n_particles=16,
        qf_lr=0.001,
        qf_target_update_interval=3000,

        policy_class=StochasticNeuralNetwork,
        policy_kwargs=policy_kwargs,
        policy_lr=0.001,

        discount=0.99,
        alpha=1,

        n_eval_episodes=10,
        q_plot_settings=None,
        env_plot_settings=None,
    )
    algorithm.train()

if __name__ == "__main__":
    test()
