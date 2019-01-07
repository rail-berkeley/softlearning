import os

import numpy as np

from softlearning.algorithms.utils import get_algorithm_from_variant
from softlearning.environments.utils import get_environment
from softlearning.misc.plotter import QFPolicyPlotter
from softlearning.samplers import SimpleSampler
from softlearning.policies.utils import get_policy_from_variant
from softlearning.replay_pools import SimpleReplayPool
from softlearning.value_functions.utils import get_Q_function_from_variant
from softlearning.misc.utils import initialize_tf_variables
from examples.utils import get_parser, launch_experiments_ray
from examples.multi_goal.variants import get_variant_spec


def run_experiment(variant, reporter):
    env = get_environment('gym', 'MultiGoal', 'Default', {
        'actuation_cost_coeff': 1,
        'distance_cost_coeff': 0.1,
        'goal_reward': 1,
        'init_sigma': 0.1,
    })

    pool = SimpleReplayPool(
        observation_space=env.observation_space,
        action_space=env.action_space,
        max_size=1e6)

    sampler = SimpleSampler(
        max_path_length=30, min_pool_size=100, batch_size=64)

    Qs = get_Q_function_from_variant(variant, env)
    policy = get_policy_from_variant(variant, env, Qs)
    plotter = QFPolicyPlotter(
        Q=Qs[0],
        policy=policy,
        obs_lst=np.array(((-2.5, 0.0),
                          (0.0, 0.0),
                          (2.5, 2.5),
                          (-2.5, -2.5))),
        default_action=(np.nan, np.nan),
        n_samples=100)

    algorithm = get_algorithm_from_variant(
        variant=variant,
        env=env,
        policy=policy,
        Qs=Qs,
        pool=pool,
        sampler=sampler,
        plotter=plotter
    )

    initialize_tf_variables(algorithm._session, only_uninitialized=True)

    for train_result in algorithm.train():
        reporter(**train_result)


def main():
    args = get_parser().parse_args()

    universe, domain, task = 'general', 'multigoal', 'default'
    local_dir = os.path.join(
        '~/ray_results', universe, domain, task)

    variant_spec = get_variant_spec(universe, domain, task, args.policy, local_dir, args.algorithm)

    launch_experiments_ray([variant_spec], args, local_dir, run_experiment)


if __name__ == "__main__":
    main()
