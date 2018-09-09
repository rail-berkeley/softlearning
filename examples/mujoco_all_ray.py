import os
import ray
from ray import tune

from softlearning.environments.utils import get_environment
from softlearning.algorithms import SAC

from softlearning.misc.utils import set_seed, datestamp
from softlearning.policies import (
    GaussianPolicy,
    LatentSpacePolicy,
    GMMPolicy,
    UniformPolicy)
from softlearning.samplers import get_sampler_from_params
from softlearning.replay_pools import (
    SimpleReplayPool,
    ExtraPolicyInfoReplayPool)
from softlearning.value_functions import NNQFunction, NNVFunction
from softlearning.preprocessors import PREPROCESSOR_FUNCTIONS
from examples.variants import get_variant_spec_image, get_variant_spec
from examples.utils import (
    parse_universe_domain_task,
    get_parser,
    setup_rllab_logger)


def run_experiment(variant, reporter):
    # Setup the rllab logger manually
    # TODO.hartikainen: We should change the logger to use some standard logger
    setup_rllab_logger(variant)
    set_seed(variant['run_params']['seed'])

    env_params = variant['env_params']
    policy_params = variant['policy_params']
    value_fn_params = variant['value_fn_params']
    preprocessor_params = variant['preprocessor_params']
    algorithm_params = variant['algorithm_params']
    replay_pool_params = variant['replay_pool_params']
    sampler_params = variant['sampler_params']

    universe = variant['universe']
    task = variant['task']
    domain = variant['domain']

    # Unfortunately we have to do hack like this because ray logger fails
    # if our variant has parentheses.
    if 'image_size' in env_params:
        env_params['image_size'] = tuple(
            int(dim) for dim in env_params['image_size'].split('x'))

    preprocessor_kwargs = preprocessor_params.get('kwargs', {})
    if 'image_size' in preprocessor_kwargs:
        preprocessor_kwargs['image_size'] = tuple(
            int(dim) for dim in preprocessor_kwargs['image_size'].split('x'))
    if 'hidden_layer_sizes' in preprocessor_kwargs:
        preprocessor_kwargs['hidden_layer_sizes'] = tuple(
            int(dim) for dim in preprocessor_kwargs['hidden_layer_sizes'].split('x'))
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

    env = get_environment(universe, domain, task, env_params)

    sampler = get_sampler_from_params(sampler_params)

    if algorithm_params['store_extra_policy_info']:
        pool = ExtraPolicyInfoReplayPool(
            observation_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            **replay_pool_params)
    else:
        pool = SimpleReplayPool(
            observation_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            **replay_pool_params)

    base_kwargs = dict(algorithm_params['base_kwargs'], sampler=sampler)

    M = value_fn_params['layer_size']
    q_functions = tuple(
        NNQFunction(
            observation_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            hidden_layer_sizes=(M, M),
            name='qf{}'.format(i))
        for i in range(2)
    )
    vf = NNVFunction(
        observation_shape=env.observation_space.shape,
        hidden_layer_sizes=(M, M))
    initial_exploration_policy = UniformPolicy(
        observation_shape=env.observation_space.shape,
        action_shape=env.action_space.shape)

    if policy_params['type'] == 'gaussian':
        policy = GaussianPolicy(
            observation_shape=env.observation_space.shape,
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
            observation_shape=env.observation_space.shape,
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
            observation_shape=env.observation_space.shape,
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
        action_prior=policy_params['action_prior'],
        save_full_state=False,
        store_extra_policy_info=algorithm_params['store_extra_policy_info'])

    for epoch, mean_return in algorithm.train():
        reporter(timesteps_total=epoch, mean_accuracy=mean_return)


def main():
    parser = get_parser(allow_policy_list=True)
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--cpus', type=int, default=None)
    args = parser.parse_args()

    universe, domain, task = parse_universe_domain_task(args)

    tune.register_trainable('mujoco-runner', run_experiment)

    cpus = (args.cpus
            if args.cpus is not None
            else {'local': 8}.get(args.mode, 16))

    if args.mode == 'local':
        ray.init()
        trial_resources = {'cpu': cpus}
    else:
        ray.init(redis_address=ray.services.get_node_ip_address() + ':6379')
        trial_resources = {'cpu': cpus}

    if args.gpus > 0:
        trial_resources['gpu'] = args.gpus

    local_dir = os.path.join('~/ray_results', universe, domain, task)

    variant_specs = []
    for policy in args.policy:
        if ('image' in task.lower()
            or 'blind' in task.lower()
            or 'image' in domain.lower()):
            variant_spec = get_variant_spec_image(
                universe, domain, task, policy)
        else:
            variant_spec = get_variant_spec(universe, domain, task, policy)

        variant_spec['run_params']['local_dir'] = local_dir
        variant_specs.append(variant_spec)

    date_prefix = datestamp()
    experiment_id = '-'.join((date_prefix, args.exp_name))

    tune.run_experiments({
        "{}-{}".format(experiment_id, policy): {
            'run': 'mujoco-runner',
            'trial_resources': trial_resources,
            'config': variant_spec,
            'local_dir': local_dir,
            'upload_dir': 'gs://sac-ray-test/ray/results'
        }
        for policy, variant_spec in zip(args.policy, variant_specs)
    })


if __name__ == '__main__':
    main()
