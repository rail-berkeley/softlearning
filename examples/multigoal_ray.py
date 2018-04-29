import argparse
import numpy as np

import ray
from ray import tune
from rllab.envs.normalized_env import normalize

from sac.algos import SAC
from sac.envs import MultiGoalEnv
from sac.misc.plotter import QFPolicyPlotter
from sac.misc.utils import timestamp
from sac.misc.sampler import SimpleSampler
from sac.policies import GMMPolicy, LatentSpacePolicy
from sac.replay_buffers import SimpleReplayBuffer
from sac.value_functions import NNQFunction, NNVFunction


def run(variant, reporter):
    env = normalize(MultiGoalEnv(
        actuation_cost_coeff=1,
        distance_cost_coeff=0.1,
        goal_reward=1,
        init_sigma=0.1,
    ))

    pool = SimpleReplayBuffer(
        max_replay_buffer_size=1e6,
        env_spec=env.spec,
    )

    sampler = SimpleSampler(
        max_path_length=30,
        min_pool_size=30,
        batch_size=64,
    )

    base_kwargs = dict(
        epoch_length=1000,
        n_epochs=1000,
        n_train_repeat=1,
        eval_render=False,
        eval_n_episodes=10,
        eval_deterministic=False,
        sampler=sampler
    )

    M = 128
    qf = NNQFunction(
        env_spec=env.spec,
        hidden_layer_sizes=[M, M]
    )

    vf = NNVFunction(
        env_spec=env.spec,
        hidden_layer_sizes=[M, M]
    )

    if variant['policy_type'] == 'gmm':
        policy = GMMPolicy(
            env_spec=env.spec,
            K=4,
            hidden_layer_sizes=[M, M],
            qf=qf,
            reg=0.001
        )
    elif variant['policy_type'] == 'lsp':
        bijector_config = {
            "scale_regularization": 0.0,
            "num_coupling_layers": 2,
            "translation_hidden_sizes": (M,),
            "scale_hidden_sizes": (M,),
        }

        policy = LatentSpacePolicy(
            env_spec=env.spec,
            mode="train",
            squash=True,
            bijector_config=bijector_config,
            observations_preprocessor=None
        )

    plotter = QFPolicyPlotter(
        qf=qf,
        policy=policy,
        obs_lst=np.array([[-2.5, 0.0],
                          [0.0, 0.0],
                          [2.5, 2.5]]),
        default_action=[np.nan, np.nan],
        n_samples=100
    )

    algorithm = SAC(
        base_kwargs=base_kwargs,
        env=env,
        policy=policy,
        pool=pool,
        qf=qf,
        vf=vf,
        plotter=plotter,

        lr=3e-4,
        scale_reward=3.0,
        discount=0.99,
        tau=1e-4,

        save_full_state=True
    )

    for epoch, mean_return in algorithm.train(as_iterable=True):
        reporter(timesteps_total=epoch, mean_accuracy=mean_return)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--policy-type', type=str, choices=('gmm', 'lsp'), default='gmm')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    variants = {
        'policy_type': tune.grid_search(['gmm', 'lsp'])
    }

    tune.register_trainable('multigoal-runner', run)
    ray.init(redis_address=ray.services.get_node_ip_address() + ':6379')


    tune.run_experiments({
        'multigoal-' + timestamp(): {
            'run': 'multigoal-runner',
            'config': variants,
            'local_dir': (
                '~/ray_results/multigoal/{}'.format(args.exp_name)),
            'upload_dir': (
                's3://sac-real-nvp/ray/multigoal/{}'.format(args.exp_name))
        }
    })

if __name__ == "__main__":
    main()
