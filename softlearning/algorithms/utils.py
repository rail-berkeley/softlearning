from .sac import SAC


def create_metric_learner_algorithm(variant,
                                    env,
                                    policy,
                                    q_functions,
                                    vf,
                                    initial_exploration_policy,
                                    replay_pool,
                                    sampler,
                                    metric_learner,
                                    *args, **kwargs):
    policy_params = variant['policy_params']
    algorithm_params = variant['algorithm_params']
    algorithm_params['base_kwargs']['sampler'] = sampler

    algorithm_args = dict(
        *args,
        env=env,
        policy=policy,
        initial_exploration_policy=initial_exploration_policy,
        pool=replay_pool,
        q_functions=q_functions,
        vf=vf,
        reparameterize=policy_params['reparameterize'],
        **algorithm_params,
        **kwargs)

    algorithm = MetricLearnerAlgorithm(
        metric_learner=metric_learner,
        **algorithm_args)

    return algorithm


def create_SAC_algorithm(variant, *args, sampler, **kwargs):
    algorithm_params = variant['algorithm_params']
    policy_params = variant['policy_params']

    base_kwargs = {
        'sampler': sampler,
        **algorithm_params.pop('base_kwargs'),
    }

    algorithm = SAC(
        base_kwargs=base_kwargs,
        **algorithm_params,
        reparameterize=policy_params['reparameterize'],
        **kwargs)

    return algorithm


ALGORITHM_CLASSES = {
    'MetricLearnerAlgorithm': create_metric_learner_algorithm,
    'SAC': create_SAC_algorithm,
}


def get_algorithm_from_variant(variant,
                               *args,
                               **kwargs):
    algorithm_params = variant['algorithm_params']
    algorithm_type = algorithm_params.pop('type')
    algorithm = ALGORITHM_CLASSES[algorithm_type](
        variant, *args, **kwargs)

    return algorithm
