"""
"""
# from railrl.misc.scripts_util import timestamp
from railrl.policies.nn_policy import FeedForwardPolicy
from railrl.qfunctions.nn_qfunction import FeedForwardCritic
from railrl.algos.ddpg import DDPG

from rllab.exploration_strategies.ou_strategy import OUStrategy
# from rllab.misc.instrument import run_experiment_lite, stub
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.normalized_env import normalize

from softqlearning.envs.mujoco.pusher import PusherEnv


def main():
    env = TfEnv(normalize(PusherEnv()))
    es = OUStrategy(env_spec=env.spec)
    qf = FeedForwardCritic(
        name_or_scope="critic",
        env_spec=env.spec,
    )
    policy = FeedForwardPolicy(
        name_or_scope="actor",
        env_spec=env.spec,
    )
    default_ddpg_params = dict(
        batch_size=64,
        n_epochs=200,
        epoch_length=1000,
        eval_samples=1000,
        max_path_length=100,
        min_pool_size=1000,
    )
    algorithm = DDPG(
        env,
        es,
        policy,
        qf,
        **default_ddpg_params,
    )

    algorithm.train()
    # exp_prefix = 'ddpg-pusher-{0}'.format(timestamp())
    # run_experiment_lite(
    #     algorithm.train(),
    #     n_parallel=1,
    #     snapshot_mode="last",
    #     exp_prefix=exp_prefix,
    #     seed=1,
    # )


if __name__ == "__main__":
    # stub(globals())
    main()
