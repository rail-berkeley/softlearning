from softlearning.algorithms import MetricLearnerAlgorithm


def create_metric_learner_algorithm(variant,
                                    env,
                                    policy,
                                    Qs,
                                    V,
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
        q_functions=Qs,
        vf=V,
        reparameterize=policy_params['reparameterize'],
        action_prior=policy_params['action_prior'],
        save_full_state=False,
        **algorithm_params,
        **kwargs)

    algorithm = MetricLearnerAlgorithm(
        metric_learner=metric_learner,
        **algorithm_args)

    return algorithm


ALGORITHM_CLASSES = {
    'MetricLearnerAlgorithm': create_metric_learner_algorithm
}


def get_algorithm_from_variant(algorithm_type,
                               variant,
                               *args,
                               **kwargs):
    algorithm = ALGORITHM_CLASSES[algorithm_type](
        variant, *args, **kwargs)

    return algorithm
