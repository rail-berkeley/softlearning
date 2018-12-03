def build_metric_learner_from_variant(variant, env, evaluation_data):
    sampler_params = variant['sampler_params']
    metric_learner_params = variant['metric_learner_params']
    metric_learner_params.update({
        'observation_shape': env.observation_space.shape,
        'max_distance': sampler_params['kwargs']['max_path_length'],
        'evaluation_data': evaluation_data
    })

    metric_learner = MetricLearner(**metric_learner_params)
    return metric_learner


def get_model_from_variant(variant, env, *args, **kwargs):
    pass
