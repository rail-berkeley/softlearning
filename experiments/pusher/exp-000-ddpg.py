"""
Fixed cylinder position, sweeping over learning rates.
"""
import sys
import os
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.envs.base import TfEnv
from rllab import config
from rllab.ec2_info import instance_info, subnet_info
from rllab.exploration_strategies.ou_strategy import OUStrategy

from railrl.policies.nn_policy import FeedForwardPolicy
from railrl.qfunctions.nn_qfunction import FeedForwardCritic
from railrl.algos.ddpg import DDPG

from softqlearning.envs.mujoco.pusher import PusherEnv
from rllab.misc.instrument import stub, run_experiment_lite
# from softqlearning.misc.instrument import stub, run_experiment_lite

stub(globals())

# from softqlearning.misc.instrument import VariantGenerator, variant
from rllab.misc.instrument import VariantGenerator, variant

# exp setup --------------------------------------------------------
exp_id = os.path.basename(__file__).split('.')[0]  # exp_xxx
exp_prefix = "pusher/" + exp_id
exp_name_prefix = 'pusher-ddpg-' + exp_id
mode = "ec2"
# mode = "local_test"

subnet = "us-west-1b"
ec2_instance = "c4.2xlarge"
config.DOCKER_IMAGE = "haarnoja/rllab"  # needs psutils
config.AWS_IMAGE_ID = "ami-e7c2e087"  # with docker already pulled

n_task_per_instance = 1
n_parallel = 2  # only for local exp
snapshot_mode = "gap"
snapshot_gap = 10
plot = False


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [0, 100, 200]

    @variant
    def env_name(self):
        return [
            "softqlearning.envs.mujoco.pusher.PusherEnv"
        ]

    @variant
    def qf_lr(self):
        return [1E-2, 1E-3, 1E-4]

    @variant
    def policy_lr(self):
        return [1E-2, 1E-3, 1E-4]
        # return [3E-4]

    @variant
    def max_path_length(self):
        return [100]

    @variant
    def alpha(self):
        return [1]


variants = VG().variants()
batch_tasks = []
print("#Experiments: %d" % len(variants))
for itr, v in enumerate(variants):
    # Construct objects.
    env_kwargs = dict(
        cylinder_pos=(-0.2, 0.2),
        target_pos=(0.1, 0.0),
        tgt_cost_coeff=1.0,
        guide_cost_coeff=0.5,
        ctrl_cost_coeff=0.1,
    )
    env = TfEnv(normalize(PusherEnv(**env_kwargs)))

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
        n_epochs=1000,
        epoch_length=1000,
        eval_samples=1000,
        max_path_length=v['max_path_length'],
        min_pool_size=10000,
    )
    if 'test' in mode:
        default_ddpg_params.update(dict(
            max_path_length=100,
            epoch_length=2,
            min_pool_size=2,
            n_epochs=1
        ))

    algorithm = DDPG(
        env,
        es,
        policy,
        qf,
        **default_ddpg_params
    )

    # EC2 settings.
    exp_name = "{exp_name_prefix}-{itr:02d}".format(
        exp_name_prefix=exp_name_prefix,
        itr=itr,
    )
    if ("ec2" in mode) and (len(exp_name) > 64):
        print("Should not use experiment name with length %d > 64.\n"
              % len(exp_name)
              + "The experiment name is %s.\n Exit now."
              % exp_name)
        sys.exit(1)

    elif 'ec2' in mode:
        target = "ec2"
        # configure instance
        info = instance_info[ec2_instance]
        config.AWS_INSTANCE_TYPE = ec2_instance
        config.AWS_SPOT_PRICE = str(info["price"])
        n_parallel = int(info["vCPU"] / 2)

        # choose subnet
        config.AWS_NETWORK_INTERFACES = [
            dict(
                SubnetId=subnet_info[subnet]["SubnetID"],
                Groups=subnet_info[subnet]["Groups"],
                DeviceIndex=0,
                AssociatePublicIpAddress=True,
            )
        ]
    elif 'local' in mode:
        target = 'local'

    # Launch instances.
    print(v)
    batch_tasks.append(
        dict(
            stub_method_call=algorithm.train(),
            exp_name=exp_name,
            seed=v['seed'],
            snapshot_mode=snapshot_mode,
            snapshot_gap=snapshot_gap,
            variant=v,
            plot=plot,
            n_parallel=n_parallel,
        )
    )
    if len(batch_tasks) >= n_task_per_instance:
        run_experiment_lite(
            batch_tasks=batch_tasks,
            exp_prefix=exp_prefix,
            mode=target,
            sync_s3_pkl=True,
            sync_s3_log=True,
            sync_s3_png=True,
            sync_log_on_termination=True,
            sync_all_data_node_to_s3=True,
            terminate_machine=("test" not in mode),
            python_command="python3"
        )
        batch_tasks = []
        if "test" in mode:
            sys.exit(0)

if ("local" not in mode) and ("test" not in mode):
    os.system("chmod 444 %s" % __file__)
