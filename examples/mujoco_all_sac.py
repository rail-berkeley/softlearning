import argparse
import os

import tensorflow as tf
import numpy as np

from rllab.envs.normalized_env import normalize
from rllab.envs.mujoco.gather.ant_gather_env import AntGatherEnv
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.mujoco.ant_env import AntEnv
from rllab.envs.mujoco.humanoid_env import HumanoidEnv
from rllab.misc.instrument import VariantGenerator

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

def run_experiment(variant):
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

    algorithm.train()


def launch_experiments(variant_generator, args):
    variants = variant_generator.variants()
    # TODO: Remove unflatten. Our variant generator should support nested params
    variants = [unflatten(variant, separator='.') for variant in variants]

    num_experiments = len(variants)
    print('Launching {} experiments.'.format(num_experiments))

    for i, variant in enumerate(variants):
        print("Experiment: {}/{}".format(i, num_experiments))
        run_params = variant['run_params']

        experiment_prefix = variant['prefix'] + '/' + args.exp_name
        experiment_name = '{prefix}-{exp_name}-{i:02}'.format(
            prefix=variant['prefix'], exp_name=args.exp_name, i=i)

        run_sac_experiment(
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

    variant_generator = get_variants(domain=domain, task=task, policy=args.policy)
    launch_experiments(variant_generator, args)


if __name__ == '__main__':
    main()
