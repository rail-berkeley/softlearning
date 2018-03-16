import re
import argparse
import json
import joblib
import os

import tensorflow as tf
import numpy as np

from rllab.envs.normalized_env import normalize
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.mujoco.ant_env import AntEnv
from rllab.envs.mujoco.humanoid_env import HumanoidEnv
from rllab.misc.instrument import VariantGenerator

from sac.core.serializable import deep_clone
from sac.algos import SACV2
from sac.envs import (
    RandomGoalSwimmerEnv, RandomGoalAntEnv, RandomGoalHumanoidEnv,
    HierarchyProxyEnv, SimpleMazeAntEnv, CrossMazeAntEnv)
from sac.misc.instrument import run_sac_experiment
from sac.misc.utils import timestamp
from sac.policies import RealNVPPolicy
from sac.replay_buffers import SimpleReplayBuffer
from sac.value_functions import NNQFunction, NNVFunction
from sac.preprocessors import MLPPreprocessor
from sac.misc import tf_utils

try:
    import git
    repo = git.Repo(os.getcwd())
    git_rev = repo.active_branch.commit.name_rev
except:
    git_rev = None

COMMON_PARAMS = {
    'seed': 'random',
    'lr': 3e-4,
    'policy_lr': 3e-4,
    'discount': 0.99,
    'target_update_interval': 1,
    'tau': 1e-2,
    'layer_size': 128,
    'batch_size': 128,
    'max_pool_size': 1e6,
    'n_train_repeat': 1,
    'epoch_length': 1000,
    'snapshot_mode': 'gap',
    'snapshot_gap': 2000,
    'sync_pkl': True,
    # real nvp configs
    'policy_coupling_layers': 2,
    'policy_s_t_layers': 1,
    'policy_prior_regularization': 0.0,
    'regularize_actions': True,
    'preprocessing_hidden_sizes': None,
    'preprocessing_output_nonlinearity': 'relu',

    'git_sha': git_rev
}


ENV_PARAMS = {
   'simple-maze-ant-env': {  # 21 DoF
        'prefix': 'simple-maze-ant-env',
        'env_name': 'simple-maze-ant',

        'epoch_length': 1000,
        'max_path_length': 1000,
        'n_epochs': int(10e3 + 1),
        'scale_reward': 3,

        'preprocessing_hidden_sizes': (128, 128, 16),
        'policy_s_t_units': 8,
        'policy_fix_h_on_reset': True,

        'snapshot_gap': 2000,

        'env_reward_type': ['sparse'],
        'discount': 0.999,
        'env_terminate_at_goal': True,
        'env_goal_reward_weight': 1000,
        'env_goal_radius': 2,
        'env_velocity_reward_weight': 0,
        'env_ctrl_cost_coeff': 0, # 1e-2,
        'env_contact_cost_coeff': 0, # 1e-3,
        'env_survive_reward': 0, # 5e-2,
        'env_goal_distance': np.linalg.norm([6, -6]),
        'env_goal_angle_range': (0, 2*np.pi),

        'pre_trained_policy_path': [
            'random-goal-ant-ablation-single-level-low-level-1-00/itr_10000.pkl',
            'random-goal-ant-ablation-single-level-low-level-1-01/itr_10000.pkl',
            'random-goal-ant-ablation-single-level-low-level-1-02/itr_10000.pkl',
            'random-goal-ant-ablation-single-level-low-level-1-03/itr_10000.pkl',
            'random-goal-ant-ablation-single-level-low-level-1-04/itr_10000.pkl',
            'random-goal-ant-ablation-single-level-low-level-1-05/itr_10000.pkl',
        ]
    },
}

ENV_PARAMS['cross-maze-ant-env'] = dict(
    ENV_PARAMS['simple-maze-ant-env'],
    **{
        'prefix': 'cross-maze-ant-env',
        'env_name': 'cross-maze-ant',
        'env_goal_distance': 12, # np.linalg.norm([6,-6]), # (np.linalg.norm([6,-6]), 12),

        'env_fixed_goal_position': [[6, -6], [6, 6], [12, 0]],
    }
)

DEFAULT_ENV = 'cross-maze-ant-env'
AVAILABLE_ENVS = list(ENV_PARAMS.keys())

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',
                        type=str,
                        choices=AVAILABLE_ENVS,
                        default=DEFAULT_ENV)
    parser.add_argument('--exp_name',type=str, default=timestamp())
    parser.add_argument('--mode', type=str, default='local')
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--pre_trained_policy_path', '-p',
                        type=str, default=None)
    args = parser.parse_args()

    return args


def get_variants(args):
    env_params = ENV_PARAMS[args.env]
    params = COMMON_PARAMS
    params.update(env_params)

    if args.mode == 'local':
        trained_policies_base = os.path.join(os.getcwd(), 'sac/policies/trained_policies')
    elif args.mode == 'ec2':
        trained_policies_base = '/root/code/rllab/sac/policies/trained_policies'

    params['pre_trained_policy_path'] = [
      os.path.join(trained_policies_base, p)
      for p in params['pre_trained_policy_path']
    ]

    vg = VariantGenerator()
    for key, val in params.items():
        if isinstance(val, list):
            vg.add(key, val)
        else:
            vg.add(key, [val])

    return vg


def load_pre_trained_policy(policy_path):
    with tf_utils.get_default_session().as_default():
        snapshot = joblib.load(policy_path)

    policy = snapshot['policy']
    qf = snapshot['qf']
    vf = snapshot['vf']

    return policy, qf, vf

def run_experiment(variant):
    policy, qf, vf = load_pre_trained_policy(
        policy_path=variant['pre_trained_policy_path'])

    env_name = variant['env_name']

    env_args = {
        name.replace('env_', '', 1): value
        for name, value in variant.items()
        if name.startswith('env_') and name != 'env_name'
    }

    if 'cross-maze-ant' == env_name:
        EnvClass = CrossMazeAntEnv
    else:
        raise NotImplementedError

    env = normalize(EnvClass(**env_args))

    pool = SimpleReplayBuffer(
        env_spec=env.spec,
        max_replay_buffer_size=variant['max_pool_size'],
    )

    base_kwargs = dict(
        min_pool_size=variant['max_path_length'],
        epoch_length=variant['epoch_length'],
        n_epochs=variant['n_epochs'],
        max_path_length=variant['max_path_length'],
        batch_size=variant['batch_size'],
        n_train_repeat=variant['n_train_repeat'],
        eval_render=False,
        eval_n_episodes=1,
        eval_deterministic=True,
    )

    algorithm = SACV2(
        base_kwargs=base_kwargs,
        env=env,
        policy=policy,
        pool=pool,
        qf=qf,
        vf=vf,

        policy_lr=variant["policy_lr"],
        lr=variant['lr'],
        scale_reward=variant['scale_reward'],
        discount=variant['discount'],
        tau=variant['tau'],
        target_update_interval=variant['target_update_interval'],
        regularize_actions=variant['regularize_actions'],

        save_full_state=False,
    )

    # algorithm._pool = pool
    # algorithm._env = env
    # algorithm._save_full_state = False

    # with tf_utils.get_default_session().as_default():
    #     uninitialized_vars = [
    #         v for v in tf.global_variables()
    #         if not tf.is_variable_initialized(v).eval()
    #     ]

    tf_utils.get_default_session().run(tf.variables_initializer([
        variable for variable in tf.global_variables()
        if ('Adam' in variable.name
            or 'target/vf' in variable.name
            or 'beta1_power' in variable.name
            or 'beta2_power' in variable.name)
    ]))

    algorithm.train()

def launch_experiments(variant_generator):
    variants = variant_generator.variants()

    num_experiments = len(variants)
    print('Launching {} experiments.'.format(num_experiments))

    for i, variant in enumerate(variants):
        print("Experiment: {}/{}".format(i, num_experiments))

        if variant['seed'] == 'random':
            variant['seed'] = np.random.randint(1, 100)

        variant['pre_trained_policy_path_short'] = '/'.join(
            variant['pre_trained_policy_path'].split('/')[-2:])

        local_trained_policies_base = os.path.join(os.getcwd(), 'sac/policies/trained_policies')
        pre_train_variant_path = os.path.join(
            local_trained_policies_base,
            os.path.basename(os.path.dirname(variant['pre_trained_policy_path'])),
            'variant.json')

        with open(pre_train_variant_path, 'r') as f:
            pre_train_variant = json.load(f)
        variant['pre_train_variant'] = pre_train_variant
        variant['pre_train_iterations'] = int(re.search(
            r'(?<=itr_)(\d+)', variant['pre_trained_policy_path']).group())
        variant['pre_train_experiment'] = '-'.join(
            pre_train_variant['exp_name'].split('-')[:-1])

        experiment_prefix = variant['prefix'] + '/' + args.exp_name
        experiment_name = (variant['prefix']
                           + '-' + args.exp_name
                           + '-' + str(i).zfill(2))

        run_sac_experiment(
            run_experiment,
            mode=args.mode,
            variant=variant,
            exp_prefix=experiment_prefix,
            exp_name=experiment_name,
            n_parallel=1,
            seed=variant['seed'],
            terminate_machine=True,
            log_dir=args.log_dir,
            snapshot_mode=variant['snapshot_mode'],
            snapshot_gap=variant['snapshot_gap'],
            sync_s3_pkl=variant['sync_pkl'],
        )

if __name__ == '__main__':
    args = parse_args()
    variant_generator = get_variants(args)
    launch_experiments(variant_generator)
