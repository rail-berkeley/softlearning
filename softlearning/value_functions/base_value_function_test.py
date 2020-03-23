import pickle
from collections import OrderedDict

import numpy as np
import tensorflow as tf
import tree

from softlearning.value_functions.vanilla import feedforward_Q_function
from softlearning.environments.utils import get_environment


class ValueFunctionTest(tf.test.TestCase):
    def setUp(self):
        self.env = get_environment('gym', 'Swimmer', 'v3', {})
        self.hidden_layer_sizes = (8, 8)

        observation_shapes = OrderedDict((
            (key, value) for key, value in self.env.observation_shape.items()
        ))
        action_shape = self.env.action_shape
        input_shapes = (observation_shapes, action_shape)
        self.value_function = feedforward_Q_function(
            input_shapes=input_shapes,
            hidden_layer_sizes=self.hidden_layer_sizes,
        )

    def test_values(self):
        _ = self.env.reset()
        action1_np = self.env.action_space.sample()
        observation1_np = self.env.step(action1_np)[0]
        action2_np = self.env.action_space.sample()
        observation2_np = self.env.step(action2_np)[0]

        observations_np = type(observation1_np)((
            (key, np.stack((
                observation1_np[key], observation2_np[key]
            ), axis=0).astype(np.float32))
            for key in observation1_np.keys()
        ))

        actions_np = np.stack((
            action1_np, action2_np
        ), axis=0).astype(np.float32)

        observations_tf = tree.map_structure(
            lambda x: tf.constant(x, dtype=x.dtype), observations_np)
        actions_tf = tree.map_structure(
            lambda x: tf.constant(x, dtype=x.dtype), actions_np)

        for observations, actions in (
                (observations_np, actions_np),
                (observations_tf, actions_tf)):
            values = self.value_function.values(observations, actions)

            tf.debugging.assert_shapes(((values, (2, 1)),))

    def test_trainable_variables(self):
        self.assertEqual(
            len(self.value_function.trainable_variables),
            2 * (len(self.hidden_layer_sizes) + 1))

    def test_get_diagnostics(self):
        _ = self.env.reset()
        action1 = self.env.action_space.sample()
        observation1 = self.env.step(action1)[0]
        action2 = self.env.action_space.sample()
        observation2 = self.env.step(action2)[0]

        observations = type(observation1)((
            (key, np.stack((
                observation1[key], observation2[key]
            ), axis=0).astype(np.float32))
            for key in observation1.keys()
        ))

        actions = np.stack((
            action1, action2
        ), axis=0).astype(np.float32)

        diagnostics = self.value_function.get_diagnostics(
            observations, actions)

        self.assertTrue(isinstance(diagnostics, OrderedDict))
        self.assertEqual(tuple(diagnostics.keys()), ())

        for value in diagnostics.values():
            self.assertTrue(np.isscalar(value))

    def test_serialize_deserialize(self):
        _ = self.env.reset()
        action1_np = self.env.action_space.sample()
        observation1_np = self.env.step(action1_np)[0]
        action2_np = self.env.action_space.sample()
        observation2_np = self.env.step(action2_np)[0]

        observations = type(observation1_np)((
            (key, np.stack((
                observation1_np[key], observation2_np[key]
            ), axis=0).astype(np.float32))
            for key in observation1_np.keys()
        ))

        actions = np.stack((
            action1_np, action2_np
        ), axis=0).astype(np.float32)

        weights_1 = self.value_function.get_weights()

        values_1 = self.value_function.values(observations, actions).numpy()

        serialized = pickle.dumps(self.value_function)
        deserialized = pickle.loads(serialized)

        weights_2 = deserialized.get_weights()
        values_2 = deserialized.values(observations, actions).numpy()

        for weight_1, weight_2 in zip(weights_1, weights_2):
            np.testing.assert_array_equal(weight_1, weight_2)

        np.testing.assert_array_equal(values_1, values_2)


if __name__ == '__main__':
    tf.test.main()
