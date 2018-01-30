""" Example script to perform soft Q-learning in the multigoal environment. """
import numpy as np

from rllab.envs.normalized_env import normalize

from softqlearning.algorithms import SQL
from softqlearning.misc.kernel import adaptive_isotropic_gaussian_kernel
from softqlearning.environments import MultiGoalEnv
from softqlearning.replay_buffers import SimpleReplayBuffer
from softqlearning.value_functions import NNQFunction
from softqlearning.misc.plotter import QFPolicyPlotter
from softqlearning.policies import StochasticNNPolicy
from softqlearning.misc.sampler import SimpleSampler


def test():

    env = normalize(MultiGoalEnv())

    pool = SimpleReplayBuffer(env_spec=env.spec, max_replay_buffer_size=1e6)

    sampler = SimpleSampler(
        max_path_length=30, min_pool_size=100, batch_size=64)

    base_kwargs = {
        'sampler': sampler,
        'epoch_length': 100,
        'n_epochs': 1000,
        'n_train_repeat': 1,
        'eval_render': True,
        'eval_n_episodes': 10
    }

    M = 128
    policy = StochasticNNPolicy(
        env.spec, hidden_layer_sizes=(M, M), squash=True)

    qf = NNQFunction(env_spec=env.spec, hidden_layer_sizes=[M, M])

    plotter = QFPolicyPlotter(
        qf=qf,
        policy=policy,
        obs_lst=np.array([[-2.5, 0.0], [0.0, 0.0], [2.5, 2.5]]),
        default_action=[np.nan, np.nan],
        n_samples=100)

    algorithm = SQL(
        base_kwargs=base_kwargs,
        env=env,
        pool=pool,
        qf=qf,
        policy=policy,
        plotter=plotter,
        policy_lr=3e-4,
        qf_lr=3e-4,
        value_n_particles=16,
        td_target_update_interval=1000,
        kernel_fn=adaptive_isotropic_gaussian_kernel,
        kernel_n_particles=32,
        kernel_update_ratio=0.5,
        discount=0.99,
        reward_scale=0.1,
        save_full_state=False)

    algorithm.train()


if __name__ == "__main__":
    test()
