"""
Example script that applies soft Q-learning to the minimaze environment
"""
import numpy as np

from rllab.envs.normalized_env import normalize
# from softqlearning.misc.instrument import run_experiment_lite, stub

from sandbox.rocky.tf.envs.base import TfEnv

from softqlearning.algos.softqlearning import SoftQLearning
from softqlearning.core.kernel import AdaptiveIsotropicGaussianKernel
from softqlearning.core.nn import NeuralNetwork, StochasticNeuralNetwork
from softqlearning.envs.minimaze_env import MinimazeEnv
# from softqlearning.misc.utils import timestamp


def main():

    env = TfEnv(normalize(MinimazeEnv(
    )))

    base_kwargs = dict(
        max_path_length=50,
        epoch_length=500,
        min_pool_size=50,
        n_epochs=100,
        batch_size=64,
        scale_reward=10,
    )

    qf_kwargs = dict(
        layer_sizes=(40, 40, 1),
        output_nonlinearity=None,
    )

    policy_kwargs = dict(
        layer_sizes=(40, 40, env.action_dim),
        output_nonlinearity=None,
    )

    env_plot_settings = dict(
        xlim=(-7, 7),
        ylim=(-7, 7),
        title="paths",
        xlabel="x",
        ylabel="y",
    )

    q_plot_settings = dict(
        xlim=(-1., 1.),
        ylim=(-1., 1.),
        obs_lst=np.array([[0.0, 0.0, 0.0, 0.0]]),
        action_dims=(0, 1),
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
        qf_target_update_interval=1000,

        policy_class=StochasticNeuralNetwork,
        policy_kwargs=policy_kwargs,
        policy_lr=0.001,

        discount=0.99,
        alpha=1,

        eval_n_episodes=10,
        q_plot_settings=q_plot_settings,
        env_plot_settings=env_plot_settings,
    )

    algorithm.train()

    # exp_prefix = 'minimaze/minimaze'
    # exp_name = format(timestamp())

    # run_experiment_lite(
    #     algorithm.train(),
    #     exp_name=exp_name,
    #     exp_prefix=exp_prefix,
    #     n_parallel=0,
    #     terminate_machine=False,
    #     mode='local',
    #     use_cloudpickle=False,
    # )

if __name__ == "__main__":
    # stub(globals())
    main()
