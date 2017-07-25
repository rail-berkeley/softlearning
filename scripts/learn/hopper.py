"""
Example script that applies soft Q-learning to the hopper environment.
"""
from rllab.envs.normalized_env import normalize
from softqlearning.misc.instrument import run_experiment_lite, stub

from sandbox.rocky.tf.envs.base import TfEnv

from softqlearning.algos.softqlearning import SoftQLearning
from softqlearning.core.kernel import AdaptiveIsotropicGaussianKernel
from softqlearning.core.nn import NeuralNetwork, StochasticNeuralNetwork
from softqlearning.envs.mujoco.hopper_env import HopperEnv
from softqlearning.misc.utils import timestamp


def main():

    env = TfEnv(normalize(HopperEnv(
        alive_coeff=0.5,
    )))

    base_kwargs = dict(
        epoch_length=10000,
        # min_pool_size=10000,
        n_epochs=4000,
        max_path_length=500,
        batch_size=64,
        scale_reward=10,
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
        qf_target_update_interval=1000,

        policy_class=StochasticNeuralNetwork,
        policy_kwargs=policy_kwargs,
        policy_lr=0.001,

        discount=0.99,
        alpha=1,

        eval_n_episodes=10,
        q_plot_settings=None,
        env_plot_settings=None,
    )

    exp_prefix = 'hopper/hopper'
    exp_name = format(timestamp())

    algorithm.train()
    run_experiment_lite(
        algorithm.train(),
        exp_name=exp_name,
        exp_prefix=exp_prefix,
        n_parallel=0,
        terminate_machine=False,
        mode='local',
        use_cloudpickle=False,
    )

if __name__ == "__main__":
    stub(globals())
    main()
