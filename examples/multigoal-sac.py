import numpy as np

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite

from sac.algos.sac import SAC
from sac.envs.multigoal import MultiGoalEnv
from sac.misc.plotter import QFPolicyPlotter
from sac.misc.replay_pool import SimpleReplayPool
from sac.misc.utils import timestamp
from sac.policies.gmm import GMMPolicy
from sac.misc.value_function import NNQFunction, NNVFunction


def run(*_):
    env = normalize(MultiGoalEnv(
        actuation_cost_coeff=10,
        distance_cost_coeff=1,
        goal_reward=10,
    ))

    pool = SimpleReplayPool(
        env_spec=env.spec,
        max_pool_size=1E6
    )

    base_kwargs = dict(
        min_pool_size=30,
        epoch_length=1000,
        n_epochs=1000,
        max_path_length=30,
        batch_size=64,
        scale_reward=0.3,
        n_train_repeat=1,
        eval_render=True,
        eval_n_episodes=10,
        eval_deterministic=False
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
        K=8,
        hidden_layers=(M, M),
        qf=qf,
        reg=0.001
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

        lr=1E-3,

        discount=0.99,
        tau=0.999,

        save_full_state=True
    )
    algorithm.train()

if __name__ == "__main__":
    run_experiment_lite(
        run,
        exp_prefix='multigoal',
        exp_name=timestamp(),
        n_parallel=1,
        seed=1,
        mode='local',
    )
