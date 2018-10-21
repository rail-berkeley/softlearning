""" Example script to perform soft Q-learning in the multigoal environment. """
import numpy as np

from softlearning.algorithms import SQL
from softlearning.misc.kernel import adaptive_isotropic_gaussian_kernel
from softlearning.environments.utils import get_environment
from softlearning.replay_pools import SimpleReplayPool
from softlearning.value_functions.utils import get_Q_function_from_variant
from softlearning.misc.plotter import QFPolicyPlotter
from softlearning.policies import StochasticNNPolicy
from softlearning.samplers import SimpleSampler


def test():
    env = get_environment('rllab', 'multigoal', 'default', {
        'actuation_cost_coeff': 1,
        'distance_cost_coeff': 0.1,
        'goal_reward': 1,
        'init_sigma': 0.1,
    })

    pool = SimpleReplayPool(
        observation_shape=env.active_observation_shape,
        action_shape=env.action_space.shape,
        max_size=1e6)

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
        observation_shape=env.active_observation_shape,
        action_shape=env.action_space.shape,
        hidden_layer_sizes=(M, M),
        squash=True)

    Q = get_Q_function_from_variant({
        'Q_params': {'hidden_layer_sizes': (M, M)}
    }, env)

    plotter = QFPolicyPlotter(
        Q=Q,
        policy=policy,
        obs_lst=np.array([[-2.5, 0.0], [0.0, 0.0], [2.5, 2.5]]),
        default_action=[np.nan, np.nan],
        n_samples=100)

    algorithm = SQL(
        base_kwargs=base_kwargs,
        env=env,
        pool=pool,
        Q=Q,
        policy=policy,
        plotter=plotter,
        policy_lr=3e-4,
        Q_lr=3e-4,
        value_n_particles=16,
        td_target_update_interval=1000,
        kernel_fn=adaptive_isotropic_gaussian_kernel,
        kernel_n_particles=32,
        kernel_update_ratio=0.5,
        discount=0.99,
        reward_scale=0.1,
        save_full_state=False)

    # Do the training
    for epoch, mean_return in algorithm.train():
        pass


if __name__ == "__main__":
    test()
