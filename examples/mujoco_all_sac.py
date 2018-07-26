import tensorflow as tf

from ray.tune.variant_generator import generate_variants

from softlearning.environments.utils import get_environment
from softlearning.algorithms import SAC

from softlearning.misc.instrument import launch_experiment
from softlearning.policies import (
    GaussianPolicy,
    LatentSpacePolicy,
    GMMPolicy,
    UniformPolicy)
from softlearning.samplers import SimpleSampler
from softlearning.samplers import ExtraPolicyInfoSampler
from softlearning.replay_pools import SimpleReplayPool
from softlearning.replay_pools import ExtraPolicyInfoReplayPool
from softlearning.value_functions import NNQFunction, NNVFunction
from softlearning.preprocessors import MLPPreprocessor
from examples.variants import get_variant_spec
from examples.utils import (
    parse_universe_domain_task,
    get_parser)


def run_experiment(variant):
    env_params = variant['env_params']
    policy_params = variant['policy_params']
    value_fn_params = variant['value_fn_params']
    algorithm_params = variant['algorithm_params']
    replay_pool_params = variant['replay_pool_params']
    sampler_params = variant['sampler_params']

    universe = variant['universe']
    task = variant['task']
    domain = variant['domain']

    env = get_environment(universe, domain, task, env_params)

    if algorithm_params['store_extra_policy_info']:
        sampler = ExtraPolicyInfoSampler(**sampler_params)
        pool = ExtraPolicyInfoReplayPool(env_spec=env.spec,
                                         **replay_pool_params)
    else:
        sampler = SimpleSampler(**sampler_params)
        pool = SimpleReplayPool(env_spec=env.spec, **replay_pool_params)

    base_kwargs = dict(algorithm_params['base_kwargs'], sampler=sampler)

    M = value_fn_params['layer_size']
    qf1 = NNQFunction(env_spec=env.spec, hidden_layer_sizes=(M, M), name='qf1')
    qf2 = NNQFunction(env_spec=env.spec, hidden_layer_sizes=(M, M), name='qf2')
    vf = NNVFunction(env_spec=env.spec, hidden_layer_sizes=(M, M))
    initial_exploration_policy = UniformPolicy(env_spec=env.spec)

    if policy_params['type'] == 'gaussian':
        policy = GaussianPolicy(
            env_spec=env.spec,
            hidden_layer_sizes=(M, M),
            reparameterize=policy_params['reparameterize'],
            reg=1e-3,
        )
    elif policy_params['type'] == 'lsp':
        preprocessing_layer_sizes = policy_params.get(
            'preprocessing_layer_sizes')
        if preprocessing_layer_sizes is not None:
            nonlinearity = {
                None: None,
                'relu': tf.nn.relu,
                'tanh': tf.nn.tanh
            }[policy_params['preprocessing_output_nonlinearity']]

            observations_preprocessor = MLPPreprocessor(
                env_spec=env.spec,
                layer_sizes=preprocessing_layer_sizes,
                output_nonlinearity=nonlinearity)
        else:
            observations_preprocessor = None

        policy_s_t_layers = policy_params['s_t_layers']
        policy_s_t_units = policy_params['s_t_units']
        s_t_hidden_sizes = [policy_s_t_units] * policy_s_t_layers

        bijector_config = {
            'num_coupling_layers': policy_params['coupling_layers'],
            'translation_hidden_sizes': s_t_hidden_sizes,
            'scale_hidden_sizes': s_t_hidden_sizes,
        }

        policy = LatentSpacePolicy(
            env_spec=env.spec,
            squash=policy_params['squash'],
            bijector_config=bijector_config,
            reparameterize=policy_params['reparameterize'],
            q_function=qf1,
            observations_preprocessor=observations_preprocessor)
    elif policy_params['type'] == 'gmm':
        # reparameterize should always be False if using a GMMPolicy
        policy = GMMPolicy(
            env_spec=env.spec,
            K=policy_params['K'],
            hidden_layer_sizes=(M, M),
            reparameterize=policy_params['reparameterize'],
            qf=qf1,
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
        qf1=qf1,
        qf2=qf2,
        vf=vf,
        lr=algorithm_params['lr'],
        target_entropy=algorithm_params['target_entropy'],
        reward_scale=algorithm_params['reward_scale'],
        discount=algorithm_params['discount'],
        tau=algorithm_params['tau'],
        reparameterize=policy_params['reparameterize'],
        target_update_interval=algorithm_params['target_update_interval'],
        action_prior=policy_params['action_prior'],
        store_extra_policy_info=algorithm_params['store_extra_policy_info'],
        save_full_state=False,
    )

    # Do the training
    for epoch, mean_return in algorithm.train():
        pass


def launch_experiments(variants, args):
    num_experiments = len(variants)

    print('Launching {} experiments.'.format(num_experiments))

    for i, variant in enumerate(variants):
        print("Experiment: {}/{}".format(i, num_experiments))
        run_params = variant['run_params']

        experiment_prefix = variant['prefix'] + '/' + args.exp_name
        experiment_name = '{prefix}-{exp_name}-{i:02}'.format(
            prefix=variant['prefix'], exp_name=args.exp_name, i=i)

        launch_experiment(
            run_experiment,
            mode=args.mode,
            variant=variant,
            exp_prefix=experiment_prefix,
            exp_name=experiment_name,
            n_parallel=1,
            seed=run_params['seed'],
            terminate_machine=True,
            log_dir=args.log_dir,
            snapshot_mode=run_params['snapshot_mode'],
            snapshot_gap=run_params['snapshot_gap'],
            sync_s3_pkl=run_params['sync_pkl'],
        )


def main():
    args = get_parser().parse_args()

    universe, domain, task = parse_universe_domain_task(args)

    variant_spec = get_variant_spec(universe, domain, task, args.policy)

    variant_spec['mode'] = args.mode
    variants = [x[1] for x in generate_variants(variant_spec)]
    launch_experiments(variants, args)


if __name__ == '__main__':
    main()
