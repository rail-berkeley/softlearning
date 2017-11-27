from rllab.envs.normalized_env import normalize
from rllab.envs.mujoco.swimmer_env import SwimmerEnv

from sac.misc.instrument import run_sac_experiment
from sac.algos.sac import SAC
from sac.misc.replay_pool import SimpleReplayPool
from sac.misc.utils import timestamp
from sac.policies.gmm import GMMPolicy
from sac.misc.value_function import NNQFunction, NNVFunction


def run(*_):
    env = normalize(SwimmerEnv())

    pool = SimpleReplayPool(
        env_spec=env.spec,
        max_pool_size=1E6
    )

    base_kwargs = dict(
        min_pool_size=1000,
        epoch_length=1000,
        n_epochs=1000,
        max_path_length=1000,
        batch_size=64,
        scale_reward=1,
        n_train_repeat=1,
        eval_render=False,
        eval_n_episodes=1,
        eval_deterministic=True,
    )

    M = 100
    qf = NNQFunction(
        env_spec=env.spec,
        hidden_layer_sizes=(M, M)
    )

    vf = NNVFunction(
        env_spec=env.spec,
        hidden_layer_sizes=(M, M)
    )

    policy = GMMPolicy(
        env_spec=env.spec,
        K=2,
        hidden_layers=(M, M),
        qf=qf,
        reg=0.001
    )

    algorithm = SAC(
        base_kwargs=base_kwargs,
        env=env,
        policy=policy,
        pool=pool,
        qf=qf,
        vf=vf,

        lr=3E-4,
        discount=0.99,
        tau=0.99,

        save_full_state=False
    )
    algorithm.train()

if __name__ == "__main__":
    run_sac_experiment(
        run,
        exp_prefix='swimmer',
        exp_name=timestamp(),
        n_parallel=1,
        seed=1,
        snapshot_mode='gap',
        snapshot_gap='10',
        mode='local',
    )
