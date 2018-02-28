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
from rllab.envs.mujoco.gather.ant_gather_env import AntGatherEnv
from rllab.misc.instrument import VariantGenerator

from sac.algos import SAC
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
    'snapshot_gap': 1000,
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
    'random-goal-swimmer': {  # 2 DoF
        'prefix': 'random-goal-swimmer',
        'env_name': 'random-goal-swimmer',
        'epoch_length': 1000,
        'max_path_length': 1000,
        'n_epochs': int(5e3 + 1),
        'scale_reward': 100.0,

        'preprocessing_hidden_sizes': (128, 128, 4),
        'policy_s_t_units': 2,

        'snapshot_gap': 500,

        'env_reward_type': ['dense'],
        'env_terminate_at_goal': False,
        'env_goal_reward_weight': 3e-1,
        'env_goal_radius': 0.25,
        'env_goal_distance': 5,
        'env_goal_angle_range': (-0.25*np.pi, 0.25*np.pi),

        'low_level_policy_path': [
            'multi-direction-swimmer-low-level-policy-3-00/itr_100.pkl',
            'multi-direction-swimmer-low-level-policy-3-01/itr_100.pkl',
            'multi-direction-swimmer-low-level-policy-3-02/itr_100.pkl',
            'multi-direction-swimmer-low-level-policy-3-03/itr_100.pkl',
            'multi-direction-swimmer-low-level-policy-3-04/itr_100.pkl',
        ]
    },
    'random-goal-ant': {  # 8 DoF
        'prefix': 'random-goal-ant',
        'env_name': 'random-goal-ant',
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
        'env_goal_radius': 5,
        'env_velocity_reward_weight': 0,
        'env_ctrl_cost_coeff': 0, # 1e-2,
        'env_contact_cost_coeff': 0, # 1e-3,
        'env_survive_reward': 0, # 5e-2,
        'env_goal_distance': (5, 25),
        'env_goal_angle_range': (0, 2*np.pi),

        'low_level_policy_path': [
            'multi-direction-ant-low-level-policy-polynomial-decay-2-12/itr_10000.pkl',
            'multi-direction-ant-low-level-policy-polynomial-decay-2-13/itr_10000.pkl',
            'multi-direction-ant-low-level-policy-polynomial-decay-2-14/itr_10000.pkl',
            'multi-direction-ant-low-level-policy-polynomial-decay-2-15/itr_10000.pkl',
            'multi-direction-ant-low-level-policy-polynomial-decay-2-16/itr_10000.pkl',
            'multi-direction-ant-low-level-policy-polynomial-decay-2-17/itr_10000.pkl',
        ]
    },
    'random-goal-humanoid': {  # 21 DoF
        'prefix': 'random-goal-humanoid',
        'env_name': 'random-goal-humanoid',
        'epoch_length': 1000,
        'max_path_length': 1000,
        'n_epochs': int(2e5 + 1),
        'scale_reward': 3.0,

        'preprocessing_hidden_sizes': (128, 128, 42),
        'policy_s_t_units': 21,

        'snapshot_gap': 1000,

        'env_reward_type': ['dense'],
        'env_terminate_at_goal': False,
        'env_goal_reward_weight': 3e-1,
        'env_goal_radius': 0.25,
        'env_goal_distance': 5,
        'env_goal_angle_range': (0, 2*np.pi),

        'low_level_policy_path': [
            'multi-direction-humanoid-low-level-policy-2-00/itr_20000.pkl',
            'multi-direction-humanoid-low-level-policy-2-01/itr_20000.pkl',
            'multi-direction-humanoid-low-level-policy-2-02/itr_20000.pkl',
            'multi-direction-humanoid-low-level-policy-2-03/itr_20000.pkl',
            'multi-direction-humanoid-low-level-policy-2-04/itr_20000.pkl',
        ]
    },
    'ant-resume-training': {  # 8 DoF
        'prefix': 'ant-resume-training',
        'env_name': 'ant-rllab',
        'max_path_length': 1000,
        'n_epochs': int(4e3 + 1),
        'scale_reward': 3.0,

        'preprocessing_hidden_sizes': (128, 128, 16),
        'policy_s_t_units': 8,

        'snapshot_gap': 1000,

        'low_level_policy_path': [
            'ant-rllab-real-nvp-final-00-00/itr_6000.pkl',
            'ant-rllab-real-nvp-final-00-01/itr_6000.pkl',
            'ant-rllab-real-nvp-final-00-02/itr_6000.pkl',
            'ant-rllab-real-nvp-final-00-03/itr_6000.pkl',
            'ant-rllab-real-nvp-final-00-04/itr_6000.pkl',
        ]
    },
    'humanoid-resume-training': {  # 21 DoF
        'prefix': 'humanoid-resume-training',
        'env_name': 'humanoid-rllab',
        'max_path_length': 1000,
        'n_epochs': int(1e4 + 1),
        'scale_reward': 3.0,

        'preprocessing_hidden_sizes': (128, 128, 42),
        'policy_s_t_units': 21,

        'snapshot_gap': 2000,

        'low_level_policy_path': [
            'humanoid-real-nvp-final-01b-00/itr_10000.pkl',
            'humanoid-real-nvp-final-01b-01/itr_10000.pkl',
            'humanoid-real-nvp-final-01b-02/itr_10000.pkl',
            'humanoid-real-nvp-final-01b-03/itr_10000.pkl',
            'humanoid-real-nvp-final-01b-04/itr_10000.pkl',
        ]
    },

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
        'discount': [0.99],
        'control_interval': [3],
        'env_terminate_at_goal': True,
        'env_goal_reward_weight': [1000],
        'env_goal_radius': 2,
        'env_velocity_reward_weight': 0,
        'env_ctrl_cost_coeff': 0, # 1e-2,
        'env_contact_cost_coeff': 0, # 1e-3,
        'env_survive_reward': 0, # 5e-2,
        'env_goal_distance': np.linalg.norm([6, -6]),
        'env_goal_angle_range': (0, 2*np.pi),

        'low_level_policy_path': [
            'multi-direction-ant-low-level-policy-3-{:02}/itr_{}000.pkl'.format(i, j)
            for i in [12,13,14,15,16,17]
            for j in [4] # [2,4,6,8,10]
        ]
    },
    'ant-gather-env': {  # 21 DoF
        'prefix': 'ant-gather-env',
        'env_name': 'ant-gather-env',

        'epoch_length': 1000,
        'max_path_length': 1000,
        'n_epochs': int(10e3 + 1),
        'scale_reward': [1, 10, 100],

        'preprocessing_hidden_sizes': (128, 128, 16),
        'policy_s_t_units': 8,
        'policy_fix_h_on_reset': [False, True],

        'snapshot_gap': 2000,

        'discount': [0.99],
        'regularize_actions': False,
        'control_interval': [1,3,10],

        'env_activity_range': 20,
        'env_sensor_range': 20,
        'env_n_bombs': 12,
        'env_n_apples': 12,
        'env_sensor_span': 2*np.pi,

        'low_level_policy_path': [
            'multi-direction-ant-low-level-policy-3-{:02}/itr_{}000.pkl'.format(i, j)
            for i in [12,13]
            for j in [10] # [2,4,6,8,10]
        ]
    },
}

ENV_PARAMS['cross-maze-ant-env'] = dict(
    ENV_PARAMS['simple-maze-ant-env'],
    **{
        'prefix': 'cross-maze-ant-env',
        'env_name': 'cross-maze-ant',
        'env_goal_distance': 12,

        'env_fixed_goal_position': [[6, -6], [6, 6], [12, 0]],
    }
)

DEFAULT_ENV = 'random-goal-swimmer'
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
    parser.add_argument('--low_level_policy_path', '-p',
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

    params['low_level_policy_path'] = [
      os.path.join(trained_policies_base, p)
      for p in params['low_level_policy_path']
    ]

    vg = VariantGenerator()
    for key, val in params.items():
        if isinstance(val, list) or callable(val):
            vg.add(key, val)
        else:
            vg.add(key, [val])

    return vg


def load_low_level_policy(policy_path):
    with tf_utils.get_default_session().as_default():
        with tf.variable_scope("low_level_policy", reuse=False):
            snapshot = joblib.load(policy_path)

    policy = snapshot["policy"]

    return policy

RANDOM_GOAL_ENVS = {
    'swimmer': RandomGoalSwimmerEnv,
    'ant': RandomGoalAntEnv,
    'humanoid': RandomGoalHumanoidEnv,
}

RLLAB_ENVS = {
    'ant-rllab': AntEnv,
    'humanoid-rllab': HumanoidEnv
}

def run_experiment(variant):
    low_level_policy = load_low_level_policy(
        policy_path=variant['low_level_policy_path'])

    env_name = variant['env_name']
    env_type = env_name.split('-')[-1]

    env_args = {
        name.replace('env_', '', 1): value
        for name, value in variant.items()
        if name.startswith('env_') and name != 'env_name'
    }

    if 'random-goal' in env_name:
        EnvClass = RANDOM_GOAL_ENVS[env_type]
    elif 'simple-maze-ant' == env_name:
        EnvClass = SimpleMazeAntEnv
    elif 'cross-maze-ant' == env_name:
        EnvClass = CrossMazeAntEnv
    elif 'ant-gather-env' == env_name:
        EnvClass = AntGatherEnv
    elif 'rllab' in env_name:
        EnvClass = RLLAB_ENVS[variant['env_name']]
    else:
        raise NotImplementedError

    base_env = normalize(EnvClass(**env_args))
    env = HierarchyProxyEnv(wrapped_env=base_env,
                            low_level_policy=low_level_policy)

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
        control_interval=variant.get('control_interval', 1),
        n_train_repeat=variant['n_train_repeat'],
        eval_render=False,
        eval_n_episodes=1,
        eval_deterministic=True,
    )

    M = variant['layer_size']
    qf = NNQFunction(
        env_spec=env.spec,
        hidden_layer_sizes=[M, M],
    )

    vf = NNVFunction(
        env_spec=env.spec,
        hidden_layer_sizes=[M, M],
    )

    preprocessing_hidden_sizes = variant.get('preprocessing_hidden_sizes')
    observations_preprocessor = (
        MLPPreprocessor(env_spec=env.spec,
                        layer_sizes=preprocessing_hidden_sizes,
                        name='high_level_observations_preprocessor')
        if preprocessing_hidden_sizes is not None
        else None
    )

    policy_s_t_layers = variant['policy_s_t_layers']
    policy_s_t_units = variant['policy_s_t_units']
    s_t_hidden_sizes = [policy_s_t_units] * policy_s_t_layers

    real_nvp_config = {
        "prior_regularization": 0.0,
        "num_coupling_layers": variant['policy_coupling_layers'],
        "translation_hidden_sizes": s_t_hidden_sizes,
        "scale_hidden_sizes": s_t_hidden_sizes,
    }

    policy = RealNVPPolicy(
        env_spec=env.spec,
        mode="train",
        squash=False,
        real_nvp_config=real_nvp_config,
        fix_h_on_reset=variant.get('policy_fix_h_on_reset', False),
        observations_preprocessor=observations_preprocessor,
        name="high_level_policy"
    )

    algorithm = SAC(
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

    tf_utils.get_default_session().run(tf.variables_initializer([
        variable for variable in tf.global_variables()
        if 'low_level_policy' not in variable.name
    ]))

    algorithm.train()

def launch_experiments(variant_generator):
    variants = variant_generator.variants()

    num_experiments = len(variants)
    print('Launching {} experiments.'.format(num_experiments))

    seen_seeds = set()
    for i, variant in enumerate(variants):
        print("Experiment: {}/{}".format(i, num_experiments))

        if variant['seed'] == 'random':
            variant['seed'] = np.random.randint(1, 10000)
            while  variant['seed'] in seen_seeds:
                variant['seed'] = np.random.randint(1, 10000)
            seen_seeds.add(variant['seed'])

        variant['low_level_policy_path_short'] = '/'.join(
            variant['low_level_policy_path'].split('/')[-2:])

        local_trained_policies_base = os.path.join(os.getcwd(), 'sac/policies/trained_policies')
        low_level_variant_path = os.path.join(
            local_trained_policies_base,
            os.path.basename(os.path.dirname(variant['low_level_policy_path'])),
            'variant.json')

        with open(low_level_variant_path, 'r') as f:
            low_level_variant = json.load(f)
        variant['low_level_variant'] = low_level_variant
        variant['low_level_iterations'] = int(re.search(
            r'(?<=itr_)(\d+)', variant['low_level_policy_path']).group())
        variant['low_level_experiment'] = '-'.join(
            low_level_variant['exp_name'].split('-')[:-1])

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
