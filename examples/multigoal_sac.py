import argparse
import numpy as np

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from sac.algos import SAC
from sac.envs import MultiGoalEnv
from sac.misc.plotter import QFPolicyPlotter
from sac.misc.utils import timestamp
from sac.policies import GMMPolicy, LatentSpacePolicy
from sac.replay_buffers import SimpleReplayBuffer
from sac.value_functions import NNQFunction, NNVFunction


def run(variant):
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

    base_kwargs = dict(
        min_pool_size=30,
        epoch_length=1000,
        n_epochs=1000,
        max_path_length=30,
        batch_size=64,
        n_train_repeat=1,
        eval_render=True,
        eval_n_episodes=10,
        eval_deterministic=False
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
    algorithm.train()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--policy-type', type=str, choices=('gmm', 'lsp'), default='gmm')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    variant = {
        'policy_type': args.policy_type
    }

    run_experiment_lite(
        run,
        exp_prefix='multigoal',
        exp_name=timestamp(),
        variant=variant,
        snapshot_mode='last',
        n_parallel=1,
        seed=1,
        mode='local',
    )



if __name__ == "__main__":
    main()
