import argparse

from rllab.envs.normalized_env import normalize
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.mujoco.ant_env import AntEnv
from rllab.envs.mujoco.humanoid_env import HumanoidEnv

from ray.tune.variant_generator import generate_variants

from softlearning.algorithms import SAC
from softlearning.environments import (
    GymEnv,
    PusherEnv,
    ImagePusherEnv,
    MultiDirectionSwimmerEnv,
    MultiDirectionAntEnv,
    MultiDirectionHumanoidEnv,
    CrossMazeAntEnv)
from softlearning.environments.image_pusher import ImageForkReacherEnv

from softlearning.misc.instrument import launch_experiment
from softlearning.misc.utils import timestamp
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
from softlearning.preprocessors import (
    FeedforwardNetPreprocessor,
    PREPROCESSOR_FUNCTIONS,
)
from examples.variants import parse_domain_and_task, get_variant_spec


ENVIRONMENTS = {
    'swimmer-gym': {
        'default': lambda: GymEnv('Swimmer-v1'),
    },
    'swimmer-rllab': {
        'default': SwimmerEnv,
        'multi-direction': MultiDirectionSwimmerEnv,
    },
    'ant-gym': {
        'default': lambda: GymEnv('Ant-v1'),
    },
    'ant-rllab': {
        'default': AntEnv,
        'multi-direction': MultiDirectionAntEnv,
        'cross-maze': CrossMazeAntEnv
    },
    'humanoid-gym': {
        'default': lambda: GymEnv('Humanoid-v1'),
        'standup': lambda: GymEnv('HumanoidStandup-v1')
    },
    'humanoid-rllab': {
        'default': HumanoidEnv,
        'multi-direction': MultiDirectionHumanoidEnv,
    },
    'hopper': {
        'default': lambda: GymEnv('Hopper-v1')
    },
    'half-cheetah': {
        'default': lambda: GymEnv('HalfCheetah-v1')
    },
    'walker': {
        'default': lambda: GymEnv('Walker2d-v1')
    },
    'pusher': {
        'default': PusherEnv,
        'image': ImagePusherEnv,
        'reach': ImageForkReacherEnv,
    },
}

DEFAULT_DOMAIN = DEFAULT_ENV = 'swimmer-rllab'
AVAILABLE_DOMAINS = set(ENVIRONMENTS.keys())
AVAILABLE_TASKS = set(y for x in ENVIRONMENTS.values() for y in x.keys())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain',
                        type=str,
                        choices=AVAILABLE_DOMAINS,
                        default=None)
    parser.add_argument('--task',
                        type=str,
                        choices=AVAILABLE_TASKS,
                        default='default')
    parser.add_argument('--policy',
                        type=str,
                        choices=('gaussian', 'gmm', 'lsp'),
                        default='gaussian')
    parser.add_argument('--env', type=str, default=DEFAULT_ENV)
    parser.add_argument('--exp_name', type=str, default=timestamp())
    parser.add_argument('--mode', type=str, default='local')
    parser.add_argument('--log_dir', type=str, default=None)

    args = parser.parse_args()

    return args


def run_experiment(variant):
    env_params = variant['env_params']
    policy_params = variant['policy_params']
    value_fn_params = variant['value_fn_params']
    preprocessor_params = variant['preprocessor_params']
    algorithm_params = variant['algorithm_params']
    replay_pool_params = variant['replay_pool_params']
    sampler_params = variant['sampler_params']

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

    env = normalize(ENVIRONMENTS[domain][task](**env_params))

    if algorithm_params['store_extra_policy_info']:
        sampler = ExtraPolicyInfoSampler(**sampler_params)
        pool = ExtraPolicyInfoReplayPool(env_spec=env.spec, **replay_pool_params)
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
            hidden_layer_sizes=[policy_params['hidden_layer_width']]*2,
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
            env_spec=env.spec,
            squash=policy_params['squash'],
            bijector_config=bijector_config,
            reparameterize=policy_params['reparameterize'],
            q_function=qf1,
            observations_preprocessor=preprocessor)
    elif policy_params['type'] == 'gmm':
        assert not policy_params['reparameterize'], (
            "reparameterize should be False when using a GMMPolicy")
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
    args = parse_args()

    domain, task = args.domain, args.task
    if (not domain) or (not task):
        domain, task = parse_domain_and_task(args.env)

    variant_spec = get_variant_spec(
        domain=domain, task=task, policy=args.policy)
    variant_spec['mode'] = args.mode
    variants = [x[1] for x in generate_variants(variant_spec)]
    launch_experiments(variants, args)


if __name__ == '__main__':
    main()
