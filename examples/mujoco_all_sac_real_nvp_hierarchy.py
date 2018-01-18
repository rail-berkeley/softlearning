import argparse

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import VariantGenerator

from sac.algos import SACV2
from sac.envs import (
    GymEnv,
    MultiDirectionSwimmerEnv,
    MultiDirectionAntEnv,
    RandomGoalSwimmerEnv)
from sac.misc.instrument import run_sac_experiment
from sac.misc.utils import timestamp
from sac.policies import HierarchyPolicy
from sac.replay_buffers import SimpleReplayBuffer
from sac.value_functions import NNQFunction, NNVFunction
from sac.preprocessors import MLPPreprocessor


COMMON_PARAMS = {
    "seed": [1],
    "lr": 3e-4,
    "discount": 0.99,
    "target_update_interval": 1,
    "tau": 1e-2,
    "layer_size": 128,
    "batch_size": 128,
    "max_pool_size": 1e6,
    "n_train_repeat": [4],
    "epoch_length": 1000,
    "snapshot_mode": 'gap',
    "snapshot_gap": 100,
    "sync_pkl": True,

    # real nvp configs
    "policy_lr":  3e-4,
    "policy_coupling_layers": [2],
    "policy_s_t_layers": [1],
    "policy_s_t_units": [128],

    "preprocessing_hidden_sizes": [(128, 16)],
}


ENV_PARAMS = {
    'random-goal-swimmer': { # 2 DoF
        'prefix': 'random-goal-swimmer',
        'env_name': 'random-goal-swimmer',
        'max_path_length': 1000,
        'n_epochs': 502,
        'scale_reward': 100.0,

        "preprocessing_hidden_sizes": None,
    },
}

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
    params['low_level_policy_path'] = args.low_level_policy_path

    vg = VariantGenerator()
    for key, val in params.items():
        if isinstance(val, list):
            vg.add(key, val)
        else:
            vg.add(key, [val])

    return vg


def load_low_level_policy(policy_path):
    return None


def run_experiment(variant):
    if variant['env_name'] == 'random-goal-swimmer':
        env = normalize(RandomGoalSwimmerEnv())

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
                        layer_sizes=preprocessing_hidden_sizes)
        if preprocessing_hidden_sizes is not None
        else None
    )

    policy_s_t_layers = variant['policy_s_t_layers']
    policy_s_t_units = variant['policy_s_t_units']
    s_t_hidden_sizes = [policy_s_t_units] * policy_s_t_layers

    real_nvp_config = {
        "scale_regularization": 0.0,
        "num_coupling_layers": variant['policy_coupling_layers'],
        "translation_hidden_sizes": s_t_hidden_sizes,
        "scale_hidden_sizes": s_t_hidden_sizes,
    }

    low_level_policy = load_low_level_policy(
        policy_path=variant['low_level_policy_path'])

    policy = HierarchyPolicy(
        env_spec=env.spec,
        low_level_policy=low_level_policy,
        control_interval=10,
        mode="train",
        squash=True,
        real_nvp_config=real_nvp_config,
        observations_preprocessor=observations_preprocessor
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

        save_full_state=False,
    )

    # algorithm.train()

def launch_experiments(variant_generator):
    variants = variant_generator.variants()

    num_experiments = len(variants)
    print('Launching {} experiments.'.format(num_experiments))

    for i, variant in enumerate(variants):
        print("Experiment: {}/{}".format(i, num_experiments))
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
