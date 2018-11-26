import os
import pickle

import numpy as np
import tensorflow as tf
from ray import tune

from softlearning.environments.utils import get_environment_from_variant
from softlearning.algorithms.utils import get_algorithm_from_variant
from softlearning.policies.utils import get_policy_from_variant, get_policy
from softlearning.preprocessors.utils import get_preprocessor_from_variant
from softlearning.replay_pools.utils import get_replay_pool_from_variant
from softlearning.samplers.utils import get_sampler_from_variant
from softlearning.value_functions.utils import get_Q_function_from_variant

from softlearning.misc.utils import set_seed, initialize_tf_variables

from examples.utils import (
    parse_universe_domain_task,
    get_parser,
    launch_experiments_ray)
from examples.development.variants import (
    get_variant_spec,
    get_variant_spec_image)


class ExperimentRunner(tune.Trainable):
    def _setup(self, variant):
        if 'ray' in variant['mode']:
            set_seed(variant['run_params']['seed'])

        self._variant = variant

        self._session = tf.keras.backend.get_session()

        env = self.env = get_environment_from_variant(variant)
        replay_pool = self.replay_pool = (
            get_replay_pool_from_variant(variant, env))
        sampler = self.sampler = get_sampler_from_variant(variant)
        preprocessor = self.preprocessor = (
            get_preprocessor_from_variant(variant, env))
        Qs = self.Qs = get_Q_function_from_variant(variant, env)
        policy = self.policy = (
            get_policy_from_variant(variant, env, Qs, preprocessor))
        initial_exploration_policy = self.initial_exploration_policy = (
            get_policy('UniformPolicy', env))

        self.algorithm = get_algorithm_from_variant(
            variant=variant,
            env=env,
            policy=policy,
            initial_exploration_policy=initial_exploration_policy,
            Qs=Qs,
            pool=replay_pool,
            sampler=sampler,
            session=self._session,
        )

        self._session = tf.keras.backend.get_session()
        initialize_tf_variables(self._session, only_uninitialized=True)

        self.train_generator = None

    def _stop(self):
        pass

    def _train(self):
        if self.train_generator is None:
            self.train_generator = self.algorithm.train()

        diagnostics = next(self.train_generator)
        return diagnostics

    def _pickle_path(self, checkpoint_dir):
        return os.path.join(checkpoint_dir, 'checkpoint.pkl')

    def _tf_checkpoint_prefix(self, checkpoint_dir):
        return os.path.join(checkpoint_dir, 'checkpoint')

    def _get_tf_checkpoint(self):
        tf_checkpoint_items = {
            'Q_0': self.Qs[0],
            'Q_1': self.Qs[1],
            'initial_exploration_policy': self.initial_exploration_policy,
        }

        if self.preprocessor is not None:
            tf_checkpoint_items['preprocessor'] = self.preprocessor

        tf_checkpoint = tf.train.Checkpoint(**tf_checkpoint_items)

        return tf_checkpoint

    def _save(self, checkpoint_dir):
        """Implements the checkpoint logic.

        TODO(hartikainen): This implementation is currently very hacky. Things
        that need to be fixed:
          - Figure out how serialize/save tf.keras.Model subclassing. The
            current implementation just dumps the weights in a pickle, which
            is not optimal.
          - Try to unify all the saving and loading into easily
            extendable/maintainable interfaces. Currently we use
            `tf.train.Checkpoint` and `pickle.dump` in very unorganized way
            which makes things not so usable.
        """
        pickleable = {
            'variant': self._variant,
            'env': self.env,
            'replay_pool': self.replay_pool,
            'sampler': self.sampler,
            'algorithm': self.algorithm,
            'policy_weights': self.policy.get_weights()
        }

        pickle_path = self._pickle_path(checkpoint_dir)
        with open(pickle_path, 'wb') as f:
            pickle.dump(pickleable, f)

        tf_checkpoint = self._get_tf_checkpoint()
        tf_checkpoint.save(
            file_prefix=self._tf_checkpoint_prefix(checkpoint_dir),
            session=self._session)

        return os.path.join(checkpoint_dir, '')

    def _restore(self, checkpoint_dir):
        assert isinstance(checkpoint_dir, str), checkpoint_dir

        pickle_path = self._pickle_path(checkpoint_dir)
        with open(pickle_path, 'rb') as f:
            pickleable = pickle.load(f)

        np.testing.assert_equal(self._variant, pickleable['variant'])

        self.env = pickleable['env']
        self.replay_pool = pickleable['replay_pool']
        self.sampler = pickleable['sampler']

        tf_checkpoint = self._get_tf_checkpoint()
        status = tf_checkpoint.restore(tf.train.latest_checkpoint(
            self._tf_checkpoint_prefix(checkpoint_dir)))
        status.initialize_or_restore(self._session)

        self.algorithm.__setstate__(pickleable['algorithm'].__getstate__())
        self.policy.set_weights(pickleable['policy_weights'])


def main():
    args = get_parser().parse_args()

    universe, domain, task = parse_universe_domain_task(args)

    if ('image' in task.lower()
        or 'blind' in task.lower()
        or 'image' in domain.lower()):
        variant_spec = get_variant_spec_image(
            universe, domain, task, args.policy)
    else:
        variant_spec = get_variant_spec(universe, domain, task, args.policy)

    variant_spec['mode'] = args.mode

    local_dir = os.path.join('~/ray_results', universe, domain, task)

    launch_experiments_ray([variant_spec], args, local_dir, ExperimentRunner)


if __name__ == '__main__':
    main()
