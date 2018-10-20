import os
import ray
from ray import tune

from softlearning.environments.utils import get_environment_from_variant
from softlearning.algorithms import SAC

from softlearning.misc.utils import set_seed, setup_rllab_logger
from softlearning.policies import (
    GaussianPolicy,
    LatentSpacePolicy,
    GMMPolicy,
    UniformPolicy)
from softlearning.samplers.utils import get_sampler_from_variant
from softlearning.replay_pools import (
    SimpleReplayPool,
    ExtraPolicyInfoReplayPool)
from softlearning.value_functions import NNQFunction, NNVFunction
from softlearning.preprocessors import PREPROCESSOR_FUNCTIONS
from examples.variants import get_variant_spec_image, get_variant_spec
from examples.utils import (
    parse_universe_domain_task,
    get_parser,
    launch_experiments_local,
    launch_experiments_ray)


def run_experiment(variant, reporter):
    # Setup the rllab logger manually
    # TODO.hartikainen: We should change the logger to use some standard logger
    setup_rllab_logger(variant)
    set_seed(variant['run_params']['seed'])

    policy_params = variant['policy_params']
    value_fn_params = variant['value_fn_params']
    preprocessor_params = variant['preprocessor_params']
    algorithm_params = variant['algorithm_params']
    replay_pool_params = variant['replay_pool_params']

    preprocessor_kwargs = preprocessor_params.get('kwargs', {})
    if 'num_conv_layers' in preprocessor_kwargs:
        num_conv_layers = preprocessor_kwargs.pop('num_conv_layers')
        filters_per_layer = preprocessor_kwargs.pop('filters_per_layer')
        kernel_size_per_layer = preprocessor_kwargs.pop('kernel_size_per_layer')

        conv_filters = (filters_per_layer, ) * num_conv_layers
        conv_kernel_sizes = (kernel_size_per_layer, ) * num_conv_layers
        preprocessor_kwargs['conv_filters'] = conv_filters
        preprocessor_kwargs['conv_kernel_sizes'] = conv_kernel_sizes
    if 'num_dense_layers' in preprocessor_kwargs:
        num_dense_layers = preprocessor_kwargs.pop(
            'num_dense_layers')
        dense_hidden_units_per_layer = preprocessor_kwargs.pop(
            'dense_hidden_units_per_layer')

        dense_hidden_layer_sizes = (
            dense_hidden_units_per_layer, ) * num_dense_layers
        preprocessor_kwargs[
            'dense_hidden_layer_sizes'] = dense_hidden_layer_sizes

    env = get_environment_from_variant(variant)
    sampler = get_sampler_from_variant(variant)

    if algorithm_params['store_extra_policy_info']:
        pool = ExtraPolicyInfoReplayPool(
            observation_shape=env.active_observation_shape,
            action_shape=env.action_space.shape,
            **replay_pool_params)
    else:
        pool = SimpleReplayPool(
            observation_shape=env.active_observation_shape,
            action_shape=env.action_space.shape,
            **replay_pool_params)

    base_kwargs = dict(algorithm_params['base_kwargs'], sampler=sampler)

    M = value_fn_params['layer_size']
    q_functions = tuple(
        NNQFunction(
            observation_shape=env.active_observation_shape,
            action_shape=env.action_space.shape,
            hidden_layer_sizes=(M, M),
            name='qf{}'.format(i))
        for i in range(2)
    )
    vf = NNVFunction(
        observation_shape=env.active_observation_shape,
        hidden_layer_sizes=(M, M))
    initial_exploration_policy = UniformPolicy(
        observation_shape=env.active_observation_shape,
        action_shape=env.action_space.shape)

    if policy_params['type'] == 'gaussian':
        policy = GaussianPolicy(
            observation_shape=env.active_observation_shape,
            action_shape=env.action_space.shape,
            hidden_layer_sizes=(M, M),
            reparameterize=policy_params['reparameterize'],
            reg=1e-3,
        )
    elif policy_params['type'] == 'lsp':
        if preprocessor_params:
            preprocessor_fn = PREPROCESSOR_FUNCTIONS[
                preprocessor_params.get('function_name')]
            preprocessor = preprocessor_fn(
                *preprocessor_params.get('args', []),
                **preprocessor_params.get('kwargs', {}))
        else:
            preprocessor = None

        policy_s_t_layers = policy_params['s_t_layers']
        policy_s_t_units = policy_params['s_t_units']
        s_t_hidden_sizes = [policy_s_t_units] * policy_s_t_layers

        bijector_config = {
            'num_coupling_layers': policy_params['coupling_layers'],
            'translation_hidden_sizes': s_t_hidden_sizes,
            'scale_hidden_sizes': s_t_hidden_sizes,
        }

        policy = LatentSpacePolicy(
            observation_shape=env.active_observation_shape,
            action_shape=env.action_space.shape,
            squash=policy_params['squash'],
            bijector_config=bijector_config,
            reparameterize=policy_params['reparameterize'],
            q_function=q_functions[0],
            observations_preprocessor=preprocessor)
    elif policy_params['type'] == 'gmm':
        assert not policy_params['reparameterize'], (
            "reparameterize should be False when using a GMMPolicy")
        policy = GMMPolicy(
            observation_shape=env.active_observation_shape,
            action_shape=env.action_space.shape,
            K=policy_params['K'],
            hidden_layer_sizes=(M, M),
            reparameterize=policy_params['reparameterize'],
            qf=q_functions[0],
            reg=1e-3,
        )
    else:
        raise NotImplementedError(policy_params['type'])

    algorithm = SAC(
        base_kwargs=base_kwargs,
        env=env,
        policy=policy,
        initial_exploration_policy=initial_exploration_policy,
        pool=pool,
        q_functions=q_functions,
        vf=vf,
        lr=algorithm_params['lr'],
        target_entropy=algorithm_params['target_entropy'],
        reward_scale=algorithm_params['reward_scale'],
        discount=algorithm_params['discount'],
        tau=algorithm_params['tau'],
        reparameterize=policy_params['reparameterize'],
        target_update_interval=algorithm_params['target_update_interval'],
        action_prior=algorithm_params['action_prior'],
        save_full_state=False,
        store_extra_policy_info=algorithm_params['store_extra_policy_info'])

    for epoch, mean_return in algorithm.train():
        reporter(timesteps_total=epoch, mean_accuracy=mean_return)


def main():
    args = get_parser().parse_args()

    universe, domain, task = parse_universe_domain_task(args)

    variant_spec = get_variant_spec(universe, domain, task, args.policy)

    local_dir = os.path.join('~/ray_results', universe, domain, task)

    if args.mode == 'local':
        launch_experiments_local(variant_spec, args, run_experiment)
    else:
        launch_experiments_ray([variant_spec], args, local_dir, run_experiment)


if __name__ == '__main__':
    main()
