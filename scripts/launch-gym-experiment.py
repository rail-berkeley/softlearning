import argparse

from rllab.misc.instrument import VariantGenerator, variant
from rllab.envs.normalized_env import normalize
from rllab.envs.gym_env import GymEnv

from sandbox.rocky.tf.envs.base import TfEnv

from sac.misc.instrument import run_sac_experiment
from sac.algos.sac import SAC
from sac.misc.replay_pool import SimpleReplayPool
from sac.misc.utils import timestamp
from sac.policies.gmm import GMMPolicy
from sac.misc.value_function import NNQFunction, NNVFunction


fixed_args = dict(
    prefix='half-cheetah',
    env_name='HalfCheetah-v1',
    max_path_length=1000,
    epoch_length=1000,
    n_epochs=1000,
    snapshot_gap=100
)


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [1, 2, 3]

    @variant
    def qf_lr(self):
        return [3E-4]

    @variant
    def policy_lr(self):
        return [3E-4]

    @variant
    def vf_lr(self):
        return [3E-4]

    @variant
    def discount(self):
        return [0.99]

    @variant
    def scale_reward(self):
        return [1]

    @variant
    def K(self):
        return [2]

    @variant
    def layer_size(self):
        return [100]

    @variant
    def target_update(self):
        return [
            dict(tau=0.999, update_interval=1)
        ]

vg = VG()
for key, val in fixed_args.items():
    vg.add(key, [val])


def run(vv):
    env = TfEnv(normalize(GymEnv(
        vv['env_name'],
        force_reset=True,
        record_video=False,
        record_log=False,
    )))

    pool = SimpleReplayPool(
        env_spec=env.spec,
        max_pool_size=1E6
    )

    base_kwargs = dict(
        min_pool_size=vv['max_path_length'],
        epoch_length=vv['epoch_length'],
        n_epochs=vv['n_epochs'],
        max_path_length=vv['max_path_length'],
        batch_size=64,
        scale_reward=vv['scale_reward'],
        n_train_repeat=1,
        eval_render=False,
        eval_n_episodes=1,
        eval_deterministic=True,
    )

    M = vv['layer_size']
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
        K=vv['K'],
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
        plotter=None,

        policy_lr=vv['policy_lr'],
        qf_lr=vv['qf_lr'],
        vf_lr=vv['vf_lr'],

        discount=vv['discount'],
        tau=vv['target_update']['tau'],
        qf_target_update_interval=vv['target_update']['update_interval'],

        save_full_state=False
    )
    algorithm.train()

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='local')
parser.add_argument('--exp', type=str, default=timestamp())
args = parser.parse_args()

for i, v in enumerate(vg.variants()):
    print('Launching {} experiments.'.format(len(vg.variants())))
    run_sac_experiment(
        run,
        variant=v,
        exp_prefix=v['prefix'] + '/' + args.exp,
        exp_name=v['prefix'] + '-' + args.exp + '-' + str(i).zfill(2),
        n_parallel=1,
        seed=v['seed'],
        terminate_machine=True,
        snapshot_mode='gap',
        snapshot_gap=v['snapshot_gap'],
        mode=args.mode,
        sync_s3_pkl=True,
    )
