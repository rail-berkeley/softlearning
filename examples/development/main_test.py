import copy

import numpy as np
import tensorflow as tf

from examples.development.main import ExperimentRunner

CONFIG = {
    'Q_params': {
        'class_name': 'double_feedforward_Q_function',
        'config': {
            'hidden_layer_sizes': (10, 10),
            'observation_keys': None,
            'preprocessors': None,
        },
    },
    'algorithm_params': {
        'class_name': 'SAC',
        'config': {
            'Q_lr': 0.0003,
            'alpha_lr': 0.0003,
            'batch_size': 4,
            'discount': 0.99,
            'epoch_length': 20,
            'eval_n_episodes': 1,
            'eval_render_kwargs': {},
            'min_pool_size': 15,
            'n_epochs': 301,
            'n_train_repeat': 1,
            'num_warmup_samples': 10,
            'policy_lr': 0.0003,
            'reward_scale': 1.0,
            'target_entropy': 'auto',
            'target_update_interval': 1,
            'tau': 0.005,
            'train_every_n_steps': 1,
            'video_save_frequency': 0,
        },
    },
    'environment_params': {
        'training': {
            'domain': 'Swimmer',
            'kwargs': {},
            'task': 'v3',
            'universe': 'gym',
        },
    },
    'exploration_policy_params': {
        'class_name': 'UniformPolicy',
        'config': {
            'observation_keys': None,
        },
    },
    'git_sha': ('fb03db4b0ffafc61d8ea6d550e7fdebeecb34d15 '
                'refactor/pick-utils-changes'),
    'mode': 'local',
    'policy_params': {
        'class_name': 'FeedforwardGaussianPolicy',
        'config': {
            'hidden_layer_sizes': (10, 10),
            'observation_keys': None,
            'preprocessors': None,
            'squash': True,
        },
    },
    'replay_pool_params': {
        'class_name': 'SimpleReplayPool',
        'config': {
            'max_size': 1000
        },
    },
    'run_params': {
        'checkpoint_at_end': True,
        'checkpoint_frequency': 60,
        'seed': 5666
    },
    'sampler_params': {
        'class_name': 'SimpleSampler',
        'config': {
            'max_path_length': 10,
        },
    },
}


def assert_weights_not_equal(weights1, weights2):
    for weight1, weight2 in zip(weights1, weights2):
        assert not np.all(np.equal(weight1, weight2))


class TestExperimentRunner(tf.test.TestCase):

    def test_checkpoint_dict(self):
        config = copy.deepcopy(CONFIG)

        experiment_runner = ExperimentRunner(config=config)

        experiment_runner._build()

        self.assertEqual(experiment_runner.algorithm._epoch, 0)
        self.assertEqual(experiment_runner.algorithm._timestep, 0)
        self.assertEqual(experiment_runner.algorithm._total_timestep, 0)
        self.assertFalse(experiment_runner.algorithm._training_started)

        self.assertEqual(experiment_runner.replay_pool.size, 0)
        self.assertEqual(experiment_runner.algorithm._alpha.numpy(), 1.0)

        initial_policy_weights = experiment_runner.policy.get_weights()
        initial_Qs_weights = [Q.get_weights() for Q in experiment_runner.Qs]

        for i in range(10):
            experiment_runner.train()

        self.assertEqual(experiment_runner.algorithm._epoch, 9)
        self.assertEqual(experiment_runner.algorithm._timestep, 20)
        self.assertEqual(experiment_runner.algorithm._total_timestep, 200)
        self.assertTrue(experiment_runner.algorithm._training_started)
        self.assertNotEqual(
            experiment_runner.algorithm._alpha.numpy(), 1.0)

        self.assertEqual(experiment_runner.replay_pool.size, 210)

        policy_weights = experiment_runner.policy.get_weights()
        Qs_weights = [Q.get_weights() for Q in experiment_runner.Qs]

        # Make sure that the training changed all the weights
        assert_weights_not_equal(initial_policy_weights, policy_weights)

        for initial_Q_weights, Q_weights in zip(initial_Qs_weights, Qs_weights):
            assert_weights_not_equal(initial_Q_weights, Q_weights)

        experiment_runner.algorithm._log_alpha.assign(tf.math.log(5.0))
        expected_alpha_value = 5.0
        self.assertEqual(
            experiment_runner.algorithm._alpha.numpy(),
            expected_alpha_value)

        trainable_variables_1 = {
            'policy': experiment_runner.policy.trainable_variables,
            'Q0': experiment_runner.Qs[0].trainable_variables,
            'Q1': experiment_runner.Qs[1].trainable_variables,
            'target_Q0': (
                experiment_runner.algorithm._Q_targets[0].trainable_variables),
            'target_Q1': (
                experiment_runner.algorithm._Q_targets[1].trainable_variables),
            'alpha': [experiment_runner.algorithm._alpha],
        }
        trainable_variables_1_np = {
            key: [variable.numpy() for variable in variables]
            for key, variables in trainable_variables_1.items()
        }

        # TODO(hartikainen/tf2): check that all the right variables were saved.
        # This was commented at the time of tf2 migration.
        # assert set(
        #     variable
        #     for _, variables in trainable_variables_1.items()
        #     for variable in variables
        # ) == set(
        #     variable for variable in tf.trainable_variables()
        #     if 'save_counter' not in variable.name)

        optimizer_variables_1 = {
            'Q_optimizer_1': (
                experiment_runner.algorithm._Q_optimizers[0].variables()),
            'Q_optimizer_2': (
                experiment_runner.algorithm._Q_optimizers[1].variables()),
            'policy_optimizer': (
                experiment_runner.algorithm._policy_optimizer.variables()),
            'alpha_optimizer': (
                experiment_runner.algorithm._alpha_optimizer.variables()),
        }
        optimizer_variables_1_np = {
            key: [variable.numpy() for variable in variables]
            for key, variables in optimizer_variables_1.items()
        }

        checkpoint = experiment_runner.save()

        experiment_runner_2 = ExperimentRunner(config=config)
        self.assertFalse(experiment_runner_2._built)

        experiment_runner_2.restore(checkpoint)

        trainable_variables_2 = {
            'policy': experiment_runner_2.policy.trainable_variables,
            'Q0': experiment_runner_2.Qs[0].trainable_variables,
            'Q1': experiment_runner_2.Qs[1].trainable_variables,
            'target_Q0': (
                experiment_runner_2.algorithm._Q_targets[0].trainable_variables
            ),
            'target_Q1': (
                experiment_runner_2.algorithm._Q_targets[1].trainable_variables
            ),
            'alpha': [experiment_runner_2.algorithm._alpha],
        }
        trainable_variables_2_np = {
            key: [variable.numpy() for variable in variables]
            for key, variables in trainable_variables_2.items()
        }

        # TODO(hartikainen/tf2): check that all the right variables were saved.
        # This was commented at the time of tf2 migration.
        # assert set(
        #     variable
        #     for _, variables in trainable_variables_2.items()
        #     for variable in variables
        # ) == set(
        #     variable for variable in tf.trainable_variables()
        #     if 'save_counter' not in variable.name)

        optimizer_variables_2 = {
            'Q_optimizer_1': (
                experiment_runner_2.algorithm._Q_optimizers[0].variables()),
            'Q_optimizer_2': (
                experiment_runner_2.algorithm._Q_optimizers[1].variables()),
            'policy_optimizer': (
                experiment_runner_2.algorithm._policy_optimizer.variables()),
            'alpha_optimizer': (
                experiment_runner_2.algorithm._alpha_optimizer.variables()),
        }
        optimizer_variables_2_np = {
            key: [variable.numpy() for variable in variables]
            for key, variables in optimizer_variables_2.items()
        }

        for i, (key, variables_1_np) in enumerate(trainable_variables_1_np.items()):
            print()
            variables_1_tf = trainable_variables_1[key]
            variables_2_tf = trainable_variables_2[key]
            variables_2_np = trainable_variables_2_np[key]
            for j, (variable_1_np, variable_2_np,
                    variable_1_tf, variable_2_tf) in enumerate(
                    zip(variables_1_np, variables_2_np,
                        variables_1_tf, variables_2_tf)):
                allclose = np.allclose(variable_1_np, variable_2_np)
                variable_1_name = variable_1_tf.name
                variable_2_name = variable_2_tf.name

                print(f"i: {i}; j: {j}; {key};"
                      f" {allclose}; {variable_1_name}; {variable_2_name}")

                if 'target_Q' in key:
                    pass
                else:
                    np.testing.assert_allclose(variable_1_np, variable_2_np)

        # for optimizer_key in optimizer_variables_1_np.keys():
        #     variables_1_np = optimizer_variables_1_np[optimizer_key]
        #     variables_2_np = optimizer_variables_2_np[optimizer_key]
        #     for variable_1_np, variable_2_np in zip(
        #             variables_1_np, variables_2_np):
        #         np.testing.assert_allclose(variable_1_np, variable_2_np)

        for i in (0, 1):
            Q_variables_tf = trainable_variables_1[f'Q{i}']
            Q_variables_np = trainable_variables_1_np[f'Q{i}']
            target_Q_variables_tf = trainable_variables_2[f'target_Q{i}']
            target_Q_variables_np = trainable_variables_2_np[f'target_Q{i}']

            for j, (Q_np, target_Q_np, Q_tf, target_Q_tf) in enumerate(
                    zip(Q_variables_np, target_Q_variables_np,
                        Q_variables_tf, target_Q_variables_tf)):
                allclose = np.allclose(Q_np, target_Q_np)
                Q_name = Q_tf.name
                target_Q_name = target_Q_tf.name

                # print(f"i: {i}; {allclose}; {Q_name}; {target_Q_name}")

        self.assertEqual(experiment_runner_2.algorithm._epoch, 10)
        self.assertEqual(experiment_runner_2.algorithm._timestep, 0)
        self.assertEqual(
            experiment_runner_2.algorithm._alpha.numpy(),
            expected_alpha_value)

        for i in range(10):
            experiment_runner_2.train()

        self.assertEqual(experiment_runner_2.algorithm._epoch, 19)
        self.assertEqual(experiment_runner_2.algorithm._timestep, 20)
        self.assertEqual(experiment_runner_2.algorithm._total_timestep, 400)
        self.assertTrue(experiment_runner_2.algorithm._training_started)

    def test_checkpoint_pool_reconstruction(self):
        config = copy.deepcopy(CONFIG)

        config['run_params']['checkpoint_replay_pool'] = True
        experiment_runner = ExperimentRunner(config=config)

        experiment_runner._build()

        self.assertEqual(experiment_runner.algorithm._epoch, 0)
        self.assertEqual(experiment_runner.algorithm._timestep, 0)
        self.assertEqual(experiment_runner.algorithm._total_timestep, 0)
        self.assertFalse(experiment_runner.algorithm._training_started)

        self.assertEqual(experiment_runner.replay_pool.size, 0)
        self.assertEqual(experiment_runner.algorithm._alpha.numpy(), 1.0)

        checkpoints = []
        while (experiment_runner.replay_pool.size
               < experiment_runner.replay_pool._max_size):
            for i in range(10):
                experiment_runner.train()
            checkpoints.append(experiment_runner.save())

        experiment_runner_2 = ExperimentRunner(config=config)
        self.assertFalse(experiment_runner_2._built)

        experiment_runner_2.restore(checkpoints[-1])

        replay_pool_1 = experiment_runner.replay_pool
        replay_pool_2 = experiment_runner_2.replay_pool

        self.assertEqual(replay_pool_1._max_size, replay_pool_2._max_size)
        self.assertEqual(replay_pool_1.size, replay_pool_2.size)
        self.assertEqual(replay_pool_2._max_size, replay_pool_2.size)
        self.assertEqual(
            set(replay_pool_1.fields.keys()),
            set(replay_pool_2.fields.keys()))

        for field_name in replay_pool_1.fields.keys():
            np.testing.assert_array_equal(
                replay_pool_1.fields[field_name],
                replay_pool_2.fields[field_name])

    def test_training_env_evaluation_env(self):
        config = copy.deepcopy(CONFIG)
        config['environment_params']['evaluation'] = (
            config['environment_params']['training'])

        config['run_params']['checkpoint_replay_pool'] = True
        experiment_runner = ExperimentRunner(config=config)

        experiment_runner._build()

        self.assertIsNot(experiment_runner.training_environment,
                         experiment_runner.evaluation_environment)

        self.assertEqual(experiment_runner.algorithm._epoch, 0)
        self.assertEqual(experiment_runner.algorithm._timestep, 0)
        self.assertEqual(experiment_runner.algorithm._total_timestep, 0)
        self.assertFalse(experiment_runner.algorithm._training_started)

        self.assertEqual(experiment_runner.replay_pool.size, 0)
        self.assertEqual(experiment_runner.algorithm._alpha.numpy(), 1.0)

        for i in range(2):
            experiment_runner.train()

    def test_uses_training_env_as_evaluation_env(self):
        config = copy.deepcopy(CONFIG)

        self.assertNotIn('evaluation', config['environment_params'])

        config['run_params']['checkpoint_replay_pool'] = True
        experiment_runner = ExperimentRunner(config=config)

        experiment_runner._build()

        self.assertIs(experiment_runner.training_environment,
                      experiment_runner.evaluation_environment)

        self.assertEqual(experiment_runner.algorithm._epoch, 0)
        self.assertEqual(experiment_runner.algorithm._timestep, 0)
        self.assertEqual(experiment_runner.algorithm._total_timestep, 0)
        self.assertFalse(experiment_runner.algorithm._training_started)

        self.assertEqual(experiment_runner.replay_pool.size, 0)
        self.assertEqual(experiment_runner.algorithm._alpha.numpy(), 1.0)

        for i in range(2):
            experiment_runner.train()


if __name__ == "__main__":
    tf.test.main()
