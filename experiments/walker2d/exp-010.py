"""
Train OpenAI gym hopper.
"""
import sys
import os
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.envs.base import TfEnv
from rllab import config
from softqlearning.ec2_info import instance_info, subnet_info
from softqlearning.misc.instrument import stub, run_experiment_lite

from softqlearning.misc.env_chooser import EnvChooser
from softqlearning.core.kernel import AdaptiveIsotropicGaussianKernel
from softqlearning.core.nn import NeuralNetwork, StochasticNeuralNetwork
from softqlearning.algos.softqlearning import SoftQLearning

stub(globals())

from softqlearning.misc.instrument import VariantGenerator, variant

# exp setup --------------------------------------------------------
exp_id = os.path.basename(__file__).split('.')[0]  # exp_xxx
exp_prefix = "walker2d/" + exp_id
exp_name_prefix = 'walker2d-' + exp_id
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
            "softqlearning.envs.mujoco.walker2d.Walker2dEnv"
        ]

    @variant
    def qf_lr(self):
        return [1E-3]

    @variant
    def policy_lr(self):
        return [1E-3]

    @variant
    def qf_target_n_particles(self):
        return [16]

    @variant
    def kernel_n_particles(self):
        return [16]

    @variant
    def kernel_update_ratio(self):
        return [0.5]

    @variant
    def alpha(self):
        return [1]

    @variant
    def scale_reward(self):
        return [3, 10, 30]

    @variant
    def max_path_length(self):
        return [500]

    @variant
    def qf_target_update(self):
       return [
            {'tau': 1.0,
             'update_interval': 300},
            {'tau': 1.0,
             'update_interval': 1000},
            {'tau': 1.0,
             'update_interval': 3000},
        ]

variants = VG().variants()
batch_tasks = []
print("#Experiments: %d" % len(variants))
for itr, v in enumerate(variants):

    # Construct objects.
    env_kwargs = dict(
    )

    env_chooser = EnvChooser()
    env = TfEnv(normalize(env_chooser.choose_env(v['env_name'], **env_kwargs)))

    qf_kwargs = dict(
        layer_sizes=(100, 100, 1),
        output_nonlinearity=None,
    )

    policy_kwargs = dict(
        layer_sizes=(100, 100, 6),
        # layer_sizes=(100, 100, env.action_space.flat_dim()),
        output_nonlinearity=None,
    )

    kernel_kwargs = dict()

    base_kwargs = dict(
        epoch_length=1000,
        min_pool_size=10000,
        n_epochs=4000,
        max_path_length=v["max_path_length"],
        batch_size=64,
        scale_reward=v["scale_reward"],
    )

    if 'test' in mode:
        base_kwargs.update(dict(
            max_path_length=2,
            epoch_length=2,
            min_pool_size=2,
            n_epochs=1
        ))

    softqlearning_kwargs = dict(
        base_kwargs=base_kwargs,
        env=env,

        kernel_class=AdaptiveIsotropicGaussianKernel,
        kernel_n_particles=v["kernel_n_particles"],
        kernel_update_ratio=v["kernel_update_ratio"],

        qf_class=NeuralNetwork,
        qf_kwargs=qf_kwargs,
        qf_target_n_particles=v["qf_target_n_particles"],
        qf_lr=v["qf_lr"],
        qf_target_update_interval=v["qf_target_update"]["update_interval"],

        policy_class=StochasticNeuralNetwork,
        policy_kwargs=policy_kwargs,
        policy_lr=v["policy_lr"],

        discount=0.99,
        alpha=1,

        eval_n_episodes=10,
        q_plot_settings=None,
        env_plot_settings=None,
    )

    algorithm = SoftQLearning(**softqlearning_kwargs)

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

    env_vars = dict()
    if 'ec2' in mode:
        env_vars = dict(MUJOCO_PY_MJPRO_PATH='/root/code/rllab/vendor/mjpro131')

    batch_tasks.append(
        dict(
            stub_method_call=algorithm.train(),
            exp_name=exp_name,
            seed=v['seed'],
            snapshot_mode=snapshot_mode,
            snapshot_gap=snapshot_gap,
            variant=v,
            plot=plot,
            n_parallel=0,#n_parallel,
            env=env_vars
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
            python_command="python3",
        )
        batch_tasks = []
        if "test" in mode:
            sys.exit(0)

if ("local" not in mode) and ("test" not in mode):
    os.system("chmod 444 %s" % __file__)
