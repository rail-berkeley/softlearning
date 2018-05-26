import argparse
import os

import tensorflow as tf
import ray
from ray import tune

from rllab.envs.normalized_env import normalize
from rllab.envs.mujoco.gather.ant_gather_env import AntGatherEnv
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.mujoco.ant_env import AntEnv
from rllab.envs.mujoco.humanoid_env import HumanoidEnv

from sac.algos import SAC
from sac.envs import (
    GymEnv,
    MultiDirectionSwimmerEnv,
    MultiDirectionAntEnv,
    MultiDirectionHumanoidEnv,
    CrossMazeAntEnv,
)

from sac.misc.instrument import run_sac_experiment
from sac.misc.utils import timestamp, unflatten
from sac.policies import LatentSpacePolicy, GMMPolicy
from sac.misc.sampler import SimpleSampler
from sac.replay_buffers import SimpleReplayBuffer
from sac.value_functions import NNQFunction, NNVFunction
from sac.preprocessors import MLPPreprocessor
from .variants import parse_domain_and_task, get_variants

ENVIRONMENTS = {
    'swimmer': {
        'default': SwimmerEnv,
        'multi-direction': MultiDirectionSwimmerEnv,
    },
    'ant': {
        'default': AntEnv,
        'multi-direction': MultiDirectionAntEnv,
        'cross-maze': CrossMazeAntEnv
    },
    'humanoid': {
        'default': HumanoidEnv,
        'multi-direction': MultiDirectionHumanoidEnv,
    },
    'hopper': {
        'default': lambda: normalize(GymEnv('Hopper-v1'))
    },
    'half-cheetah': {
        'default': lambda: normalize(GymEnv('HalfCheetah-v1'))
    },
    'walker': {
        'default': lambda: normalize(GymEnv('Walker2d-v1'))
    },
}

DEFAULT_DOMAIN = DEFAULT_ENV = 'swimmer'
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
                        choices=('lsp', 'gmm'),
                        default='lsp')
    parser.add_argument('--env', type=str, default=DEFAULT_ENV)
    parser.add_argument('--exp_name', type=str, default=timestamp())
    parser.add_argument('--mode', type=str, default='local')
    parser.add_argument('--log_dir', type=str, default=None)
    args = parser.parse_args()

    return args

DEFAULT_SNAPSHOT_DIR = '~/ray/results'
DEFAULT_SNAPSHOT_MODE = 'none'
DEFAULT_SNAPSHOT_GAP = 1000

def setup_rllab_logger(variant):
    """Temporary setup for rllab logger previously handled by run_experiment.

    TODO.hartikainen: Remove this once we have gotten rid of rllab logger.
    """

    from rllab.misc import logger

    run_params = variant['run_params']

    ray_log_dir = os.getcwd()
    log_dir = os.path.join(ray_log_dir, 'rllab-logger')

    tabular_log_file = os.path.join(log_dir, 'progress.csv')
    text_log_file = os.path.join(log_dir, 'debug.log')
    params_log_file = os.path.join(log_dir, 'params.json')
    variant_log_file = os.path.join(log_dir, 'variant.json')

    logger.log_variant(variant_log_file, variant)
    logger.add_text_output(text_log_file)
    logger.add_tabular_output(tabular_log_file)

    logger.set_snapshot_dir(log_dir)
    logger.set_snapshot_mode(
        run_params.get('snapshot_mode', DEFAULT_SNAPSHOT_MODE))
    logger.set_snapshot_gap(
        run_params.get('snapshot_gap', DEFAULT_SNAPSHOT_GAP))
    logger.set_log_tabular_only(False)

    # TODO.hartikainen: need to remove something, or push_prefix, pop_prefix?
    # logger.push_prefix("[%s] " % args.exp_name)


def run_experiment(variant, reporter):
    # Setup the rllab logger manually
    # TODO.hartikainen: We should change the logger to use some standard logger

    setup_rllab_logger(variant)

    env_params = variant['env_params']
    policy_params = variant['policy_params']
    value_fn_params = variant['value_fn_params']
    algorithm_params = variant['algorithm_params']
    replay_buffer_params = variant['replay_buffer_params']
    sampler_params = variant['sampler_params']

    task = variant['task']
    domain = variant['domain']

    env = normalize(ENVIRONMENTS[domain][task](**env_params))

    pool = SimpleReplayBuffer(env_spec=env.spec, **replay_buffer_params)

    sampler = SimpleSampler(**sampler_params)

    base_kwargs = dict(algorithm_params['base_kwargs'], sampler=sampler)

    M = value_fn_params['layer_size']
    qf = NNQFunction(env_spec=env.spec, hidden_layer_sizes=(M, M))
    vf = NNVFunction(env_spec=env.spec, hidden_layer_sizes=(M, M))

    if policy_params['type'] == 'lsp':
        nonlinearity = {
            None: None,
            'relu': tf.nn.relu,
            'tanh': tf.nn.tanh
        }[policy_params['preprocessing_output_nonlinearity']]

        preprocessing_hidden_sizes = policy_params.get('preprocessing_hidden_sizes')
        if preprocessing_hidden_sizes is not None:
            observations_preprocessor = MLPPreprocessor(
                env_spec=env.spec,
                layer_sizes=preprocessing_hidden_sizes,
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
            q_function=qf,
            observations_preprocessor=observations_preprocessor)
    elif policy_params['type'] == 'gmm':
        policy = GMMPolicy(
            env_spec=env.spec,
            K=policy_params['K'],
            hidden_layer_sizes=(M, M),
            qf=qf,
            reg=1e-3,
        )
    else:
        raise NotImplementedError(policy_params['type'])

    algorithm = SAC(
        base_kwargs=base_kwargs,
        env=env,
        policy=policy,
        pool=pool,
        qf=qf,
        vf=vf,
        lr=algorithm_params['lr'],
        scale_reward=algorithm_params['scale_reward'],
        discount=algorithm_params['discount'],
        tau=algorithm_params['tau'],
        target_update_interval=algorithm_params['target_update_interval'],
        action_prior=policy_params['action_prior'],
        save_full_state=False,
    )

    algorithm._sess.run(tf.global_variables_initializer())

    for epoch, mean_return in algorithm.train(as_iterable=True):
        reporter(timesteps_total=epoch, mean_accuracy=mean_return)

def main():
    args = parse_args()

    domain, task = args.domain, args.task
    if (not domain) or (not task):
        domain, task = parse_domain_and_task(args.env)

    variants = get_variants(domain=domain, task=task, policy=args.policy)

    local_dir = '~/ray_results/{}/{}'.format(domain, task)
    variants['run_params']['local_dir'] = local_dir

    tune.register_trainable('mujoco-runner', run_experiment)
    if args.mode == 'local':
        ray.init()
    else:
        ray.init(redis_address=ray.services.get_node_ip_address() + ':6379')

    tune.run_experiments({
        args.exp_name: {
            'run': 'mujoco-runner',
            'trial_resources': {'cpu': 8},
            'config': variants,
            'local_dir': local_dir,
            # 'upload_dir': 'gs://<your-bucket>/ray_results'
        }
    })

if __name__ == '__main__':
    main()
