from softlearning.environments.utils import get_environment_from_variant
from softlearning.algorithms.utils import get_algorithm_from_variant
from softlearning.policies.utils import get_policy_from_variant, get_policy
from softlearning.preprocessors.utils import get_preprocessor_from_variant
from softlearning.replay_pools.utils import get_replay_pool_from_variant
from softlearning.samplers.utils import get_sampler_from_variant
from softlearning.value_functions.utils import (
    get_Q_function_from_variant,
    get_V_function_from_variant)

from softlearning.misc.utils import set_seed, setup_rllab_logger

from examples.variants import get_variant_spec, get_variant_spec_image
from examples.utils import (
    parse_universe_domain_task,
    get_parser,
    launch_experiments_rllab)


def run_experiment(variant, reporter=None):
    if 'ray' in variant['mode']:
        setup_rllab_logger(variant)
        set_seed(variant['run_params']['seed'])

    env = get_environment_from_variant(variant)
    replay_pool = get_replay_pool_from_variant(variant, env)
    sampler = get_sampler_from_variant(variant)
    preprocessor = get_preprocessor_from_variant(variant)
    Qs = get_Q_function_from_variant(variant, env)
    V = get_V_function_from_variant(variant, env)
    policy = get_policy_from_variant(variant, env, Qs, preprocessor)
    initial_exploration_policy = get_policy('UniformPolicy', env)

    algorithm = get_algorithm_from_variant(
        algorithm_type='SAC',
        variant=variant,
        env=env,
        policy=policy,
        initial_exploration_policy=initial_exploration_policy,
        q_functions=Qs,
        vf=V,
        pool=replay_pool,
        sampler=sampler,
    )

    # Do the training
    for epoch, mean_return in algorithm.train():
        if reporter is not None:
            reporter(timesteps_total=epoch, mean_accuracy=mean_return)


def main():
    args = get_parser().parse_args()

    universe, domain, task = parse_universe_domain_task(args)

    if ('image' in task.lower()
        or 'blind' in task.lower()
        or 'image' in domain.lower()):
        variant_spec = get_variant_spec_image(
            universe, domain, task, args.policy)
    else:
        variant_spec = get_variant_spec(universe, domain, task, args.policy)

    variant_spec['mode'] = args.mode
    launch_experiments_rllab(variant_spec, args, run_experiment)


if __name__ == '__main__':
    main()
