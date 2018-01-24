import numpy as np

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from sac.algos import SAC
from sac.algos import SACV2
from sac.envs import MultiGoalEnv
from sac.misc.plotter import QFPolicyPlotter
from sac.misc.utils import timestamp
from sac.policies import RealNVPPolicy
from sac.replay_buffers import SimpleReplayBuffer
from sac.value_functions import NNQFunction, NNVFunction


def run(*_):
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
        n_train_repeat=2,
        eval_render=True,
        eval_n_episodes=10,
        eval_deterministic=True
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

    real_nvp_config = {
        "scale_regularization": 0.0,
        "num_coupling_layers": 2,
        "translation_hidden_sizes": (M,),
        "scale_hidden_sizes": (M,),
    }

    policy = RealNVPPolicy(
        env_spec=env.spec,
        mode="train",
        squash=True,
        real_nvp_config=real_nvp_config,
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

    algorithm = SACV2(
        base_kwargs=base_kwargs,
        env=env,
        policy=policy,
        pool=pool,
        qf=qf,
        vf=vf,
        plotter=plotter,

        lr=3e-4,
        scale_reward=3,
        discount=0.99,
        tau=1e-4,

        save_full_state=True
    )

    algorithm.train()

if __name__ == "__main__":
    run_experiment_lite(
        run,
        exp_prefix='multigoal',
        exp_name=timestamp(),
        snapshot_mode='last',
        n_parallel=1,
        seed=np.random.randint(0, 10),
        mode='local',
    )
