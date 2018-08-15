import pickle

from ray import tune
try:
    from ray.tune.variant_generator import generate_variants
except ImportError:
    # TODO(hartikainen): generate_variants has moved in >0.5.0, and some of my
    # stuff uses newer version. Remove this once we bump up the version in
    # requirements.txt
    from ray.tune.suggest.variant_generator import generate_variants


from rllab.envs.normalized_env import normalize

from softlearning.environments.rllab.pusher import PusherEnv
from softlearning.algorithms import SQL
from softlearning.misc.kernel import adaptive_isotropic_gaussian_kernel
from softlearning.replay_pools import SimpleReplayPool
from softlearning.value_functions import NNQFunction
from softlearning.policies import StochasticNNPolicy
from softlearning.samplers import SimpleSampler
from examples.utils import get_parser, launch_experiments_rllab


COMMON_PARAMS = {
    'seed': lambda spec: 1,
    'policy_lr': 3E-4,
    'qf_lr': 3E-4,
    'discount': 0.99,
    'layer_size': 128,
    'batch_size': 128,
    'max_size': 1E6,
    'n_train_repeat': 1,
    'epoch_length': 1000,
    'kernel_particles': 16,
    'kernel_update_ratio': 0.5,
    'value_n_particles': 16,
    'td_target_update_interval': 1000,
    'snapshot_mode': 'last',
    'snapshot_gap': 100,
    'save_full_state': True,
}

ENV_PARAMS = {
    'pusher': {
        'env_name': 'pusher',
        'max_path_length': 300,
        'n_epochs': 500,
        'reward_scale': 1,
        'goal': tune.grid_search([(-1, 'any'), ('any', -1)])
    }
}


def run_experiment(variant):
    if variant['env_name'] == 'pusher':
        # TODO: assumes `pusher.xml` is located in `rllab/models/` when
        # running on EC2.
        env = normalize(PusherEnv(goal=variant.get('goal')))
    else:
        raise ValueError

    pool = SimpleReplayPool(
        observation_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        max_size=variant['max_size'])

    sampler = SimpleSampler(
        max_path_length=variant['max_path_length'],
        min_pool_size=variant['max_path_length'],
        batch_size=variant['batch_size'])

    base_kwargs = dict(
        epoch_length=variant['epoch_length'],
        n_epochs=variant['n_epochs'],
        n_train_repeat=variant['n_train_repeat'],
        eval_render=False,
        eval_n_episodes=1,
        sampler=sampler)

    task_id = abs(pickle.dumps(variant).__hash__())

    M = variant['layer_size']
    qf = NNQFunction(
        observation_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        hidden_layer_sizes=(M, M),
        name='qf_{i}'.format(i=task_id))

    policy = StochasticNNPolicy(
        observation_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        hidden_layer_sizes=(M, M),
        name='policy_{i}'.format(i=task_id))

    algorithm = SQL(
        base_kwargs=base_kwargs,
        env=env,
        pool=pool,
        qf=qf,
        policy=policy,
        kernel_fn=adaptive_isotropic_gaussian_kernel,
        kernel_n_particles=variant['kernel_particles'],
        kernel_update_ratio=variant['kernel_update_ratio'],
        value_n_particles=variant['value_n_particles'],
        td_target_update_interval=variant['td_target_update_interval'],
        qf_lr=variant['qf_lr'],
        policy_lr=variant['policy_lr'],
        discount=variant['discount'],
        reward_scale=variant['reward_scale'],
        save_full_state=variant['save_full_state'])

    # Do the training
    for epoch, mean_return in algorithm.train():
        pass


def main():
    args = get_parser().parse_args()

    universe, domain, task = 'rllab', 'pusher', 'default'

    variant_spec = dict(
        COMMON_PARAMS,
        **ENV_PARAMS[domain],
        **{
            'universe': universe,
            'task': task,
            'domain': domain,
        })

    variants = [x[1] for x in generate_variants(variant_spec)]
    launch_experiments_rllab(variants, args, run_experiment)


if __name__ == '__main__':
    main()
