import os
import joblib

import tensorflow as tf
import numpy as np
from ray import tune

from rllab.envs.normalized_env import normalize
from rllab.envs.mujoco.ant_env import AntEnv
from rllab.envs.mujoco.humanoid_env import HumanoidEnv

try:
    from ray.tune.variant_generator import generate_variants
except ImportError:
    # TODO(hartikainen): generate_variants has moved in >0.5.0, and some of my
    # stuff uses newer version. Remove this once we bump up the version in
    # requirements.txt
    from ray.tune.suggest.variant_generator import generate_variants


from softlearning.algorithms import SAC
from softlearning.environments.rllab import HierarchyProxyEnv
from softlearning.policies import LatentSpacePolicy
from softlearning.replay_pools import SimpleReplayPool
from softlearning.value_functions import NNQFunction, NNVFunction
from softlearning.preprocessors import MLPPreprocessor
from softlearning.misc import tf_utils
from softlearning.misc.utils import get_git_rev
from examples.utils import (
    parse_universe_domain_task,
    get_parser,
    launch_experiments_rllab)


COMMON_PARAMS = {
    'seed': lambda spec: np.random.randint(1, 100),
    'lr': 3e-4,
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
    # lsp configs
    'policy_coupling_layers': 2,
    'policy_s_t_layers': 1,
    'policy_scale_regularization': 0.0,
    'regularize_actions': True,
    'preprocessing_layer_sizes': None,
    'preprocessing_output_nonlinearity': 'relu',

    'git_sha': get_git_rev()
}


ENV_PARAMS = {
    'random-goal-swimmer': {  # 2 DoF
        'env_name': 'random-goal-swimmer',
        'epoch_length': 1000,
        'max_path_length': 1000,
        'n_epochs': int(5e3 + 1),
        'target_entropy': -2.0,

        'preprocessing_layer_sizes': (128, 128, 4),
        'policy_s_t_units': 2,

        'snapshot_gap': 500,

        'env_reward_type': ['dense'],
        'env_terminate_at_goal': False,
        'env_goal_reward_weight': 3e-1,
        'env_goal_radius': 0.25,
        'env_goal_distance': 5,
        'env_goal_angle_range': (-0.25*np.pi, 0.25*np.pi),

        'low_level_policy_path': tune.grid_search([
            'multi-direction-swimmer-low-level-policy-3-00/itr_100.pkl',
            'multi-direction-swimmer-low-level-policy-3-01/itr_100.pkl',
            'multi-direction-swimmer-low-level-policy-3-02/itr_100.pkl',
            'multi-direction-swimmer-low-level-policy-3-03/itr_100.pkl',
            'multi-direction-swimmer-low-level-policy-3-04/itr_100.pkl',
        ])
    },
    'random-goal-ant': {  # 8 DoF
        'env_name': 'random-goal-ant',
        'epoch_length': 1000,
        'max_path_length': 1000,
        'n_epochs': int(1e5 + 1),
        'target_entropy': -8.0,

        'preprocessing_layer_sizes': (128, 128, 16),
        'policy_s_t_units': 8,

        'snapshot_gap': 1000,

        'env_reward_type': ['dense'],
        'env_terminate_at_goal': False,
        'env_goal_reward_weight': 3e-1,
        'env_goal_radius': 0.25,
        'env_goal_distance': 25,
        'env_goal_angle_range': (0, 2*np.pi),

        'low_level_policy_path': tune.grid_search([
            'multi-direction-ant-low-level-policy-1-00/itr_10000.pkl',
            'multi-direction-ant-low-level-policy-1-01/itr_10000.pkl',
            'multi-direction-ant-low-level-policy-1-02/itr_10000.pkl',
            'multi-direction-ant-low-level-policy-1-03/itr_10000.pkl',
            'multi-direction-ant-low-level-policy-1-04/itr_10000.pkl',
        ])
    },
    'random-goal-humanoid': {  # 21 DoF
        'env_name': 'random-goal-humanoid',
        'epoch_length': 1000,
        'max_path_length': 1000,
        'n_epochs': int(2e5 + 1),
        'target_entropy': -21.0,

        'preprocessing_layer_sizes': (128, 128, 42),
        'policy_s_t_units': 21,

        'snapshot_gap': 1000,

        'env_reward_type': ['dense'],
        'env_terminate_at_goal': False,
        'env_goal_reward_weight': 3e-1,
        'env_goal_radius': 0.25,
        'env_goal_distance': 5,
        'env_goal_angle_range': (0, 2*np.pi),

        'low_level_policy_path': tune.grid_search([
            'multi-direction-humanoid-low-level-policy-2-00/itr_20000.pkl',
            'multi-direction-humanoid-low-level-policy-2-01/itr_20000.pkl',
            'multi-direction-humanoid-low-level-policy-2-02/itr_20000.pkl',
            'multi-direction-humanoid-low-level-policy-2-03/itr_20000.pkl',
            'multi-direction-humanoid-low-level-policy-2-04/itr_20000.pkl',
        ])
    },
    'ant-resume-training': {  # 8 DoF
        'env_name': 'ant-rllab',
        'max_path_length': 1000,
        'n_epochs': int(4e3 + 1),
        'target_entropy': -8.0,

        'preprocessing_layer_sizes': (128, 128, 16),
        'policy_s_t_units': 8,

        'snapshot_gap': 1000,

        'low_level_policy_path': tune.grid_search([
            'ant-rllab-real-nvp-final-00-00/itr_6000.pkl',
            'ant-rllab-real-nvp-final-00-01/itr_6000.pkl',
            'ant-rllab-real-nvp-final-00-02/itr_6000.pkl',
            'ant-rllab-real-nvp-final-00-03/itr_6000.pkl',
            'ant-rllab-real-nvp-final-00-04/itr_6000.pkl',
        ])
    },
    'humanoid-resume-training': {  # 21 DoF
        'env_name': 'humanoid-rllab',
        'max_path_length': 1000,
        'n_epochs': int(1e4 + 1),
        'target_entropy': -21.0,

        'preprocessing_layer_sizes': (128, 128, 42),
        'policy_s_t_units': 21,

        'snapshot_gap': 2000,

        'low_level_policy_path': tune.grid_search([
            'humanoid-real-nvp-final-01b-00/itr_10000.pkl',
            'humanoid-real-nvp-final-01b-01/itr_10000.pkl',
            'humanoid-real-nvp-final-01b-02/itr_10000.pkl',
            'humanoid-real-nvp-final-01b-03/itr_10000.pkl',
            'humanoid-real-nvp-final-01b-04/itr_10000.pkl',
        ])
    },
}


def load_low_level_policy(policy_path):
    with tf_utils.get_default_session().as_default():
        with tf.variable_scope("low_level_policy", reuse=False):
            snapshot = joblib.load(policy_path)

    policy = snapshot["policy"]

    return policy


RLLAB_ENVS = {
    'ant-rllab': AntEnv,
    'humanoid-rllab': HumanoidEnv
}


def run_experiment(variant):
    low_level_policy = load_low_level_policy(
        policy_path=variant['low_level_policy_path_full'])

    env_name = variant['env_name']

    env_args = {
        name.replace('env_', '', 1): value
        for name, value in variant.items()
        if name.startswith('env_') and name != 'env_name'
    }
    if 'random-goal' in env_name:
        raise NotImplementedError
    elif 'rllab' in variant['env_name']:
        EnvClass = RLLAB_ENVS[variant['env_name']]
    else:
        raise NotImplementedError

    base_env = normalize(EnvClass(**env_args))
    env = HierarchyProxyEnv(wrapped_env=base_env,
                            low_level_policy=low_level_policy)
    pool = SimpleReplayPool(
        observation_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        max_size=variant['max_pool_size'],
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

    M = variant['layer_size']
    qf = NNQFunction(
        observation_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        hidden_layer_sizes=[M, M],
    )

    vf = NNVFunction(
        observation_shape=env.observation_space.shape,
        hidden_layer_sizes=[M, M],
    )

    preprocessing_layer_sizes = variant.get('preprocessing_layer_sizes')
    observations_preprocessor = (
        MLPPreprocessor(
            observation_shape=env.observation_space.shape,
            layer_sizes=preprocessing_layer_sizes,
            name='high_level_observations_preprocessor')
        if preprocessing_layer_sizes is not None
        else None
    )

    policy_s_t_layers = variant['policy_s_t_layers']
    policy_s_t_units = variant['policy_s_t_units']
    s_t_hidden_sizes = [policy_s_t_units] * policy_s_t_layers

    bijector_config = {
        "scale_regularization": 0.0,
        "num_coupling_layers": variant['policy_coupling_layers'],
        "translation_hidden_sizes": s_t_hidden_sizes,
        "scale_hidden_sizes": s_t_hidden_sizes,
    }

    policy = LatentSpacePolicy(
        observation_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        mode="train",
        squash=False,
        bijector_config=bijector_config,
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

        lr=variant['lr'],
        target_entropy=variant['target_entropy'],
        discount=variant['discount'],
        tau=variant['tau'],
        target_update_interval=variant['target_update_interval'],
        regularize_actions=variant['regularize_actions'],

        save_full_state=False,
    )

    # Do the training
    for epoch, mean_return in algorithm.train():
        pass


LOCAL_POLICIES_PATH = os.path.join(
    os.getcwd(), 'softlearning/trained_policies')
RLLAB_MOUNT_POLICIES_PATH = '/root/code/rllab/trained_policies'


def build_low_level_policy_path(spec):
    trained_policies_path = (
        LOCAL_POLICIES_PATH
        if spec['mode'] == 'local'
        else RLLAB_MOUNT_POLICIES_PATH)

    return os.path.join(trained_policies_path, spec['low_level_policy_path'])


def main():
    parser = get_parser()
    parser.add_argument(
        '--low_level_policy_path', '-p', type=str, default=None)
    args = parser.parse_args()

    universe, domain, task = parse_universe_domain_task(args)

    variant_spec = dict(
        COMMON_PARAMS,
        **ENV_PARAMS[args.env],
        **{
            'mode': args.mode,
            'low_level_policy_path_full': build_low_level_policy_path
        },
        **{
            'universe': universe,
            'task': task,
            'domain': domain,
        })

    variants = [x[1] for x in generate_variants(variant_spec)]
    launch_experiments_rllab(variants, args, run_experiment)


if __name__ == '__main__':
    main()
