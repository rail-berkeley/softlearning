import numpy as np
import tensorflow as tf

from softlearning.models.feedforward import feedforward_model


class FeedforwardTest(tf.test.TestCase):

    def test_scoping_reuses_variables_on_single_instance(self):
        output_size = 5
        x1 = tf.random_uniform((3, 2))
        x2 = tf.random_uniform((3, 13))

        self.assertFalse(tf.trainable_variables())

        fn = feedforward_model(
            output_size=output_size,
            hidden_layer_sizes=(6, 4, 2),
            name='feedforward_function')

        self.assertEqual(len(tf.trainable_variables()), 0)

        _ = fn([x1, x2])
        num_trainable_variables_1 = len(tf.trainable_variables())

        self.assertGreater(num_trainable_variables_1, 0)

        _ = fn([x2, x1])
        num_trainable_variables_2 = len(tf.trainable_variables())

        self.assertEqual(num_trainable_variables_1, num_trainable_variables_2)

    def test_scoping_creates_new_variables_across_instances(self):
        output_size = 5
        x = tf.random_uniform((1, 13))

        self.assertFalse(tf.trainable_variables())

        fn1 = feedforward_model(
            output_size=output_size,
            hidden_layer_sizes=(6, 4, 2),
            name='feedforward_function_1')
        _ = fn1([x])

        num_trainable_variables_1 = len(tf.trainable_variables())

        fn2 = feedforward_model(
            output_size=output_size,
            hidden_layer_sizes=(6, 4, 2),
            name='feedforward_function_2')
        _ = fn2([x])

        num_trainable_variables_2 = len(tf.trainable_variables())

        self.assertGreater(num_trainable_variables_1, 0)
        self.assertEqual(
            num_trainable_variables_1 * 2, num_trainable_variables_2)

        num_trainable_variables_3 = len(tf.trainable_variables())
        self.assertEqual(
            num_trainable_variables_2, num_trainable_variables_3)

    def test_clone_model(self):
        """Make sure that cloning works and clones can predict."""
        output_size = 5
        x_np = np.random.uniform(0, 1, (1, 13)).astype(np.float32)
        x = tf.constant(x_np)

        fn1 = feedforward_model(
            output_size=output_size,
            hidden_layer_sizes=(6, 4, 2),
            name='feedforward_function')
        result_1 = fn1([x, x])

        tf.keras.backend.get_session().run(tf.global_variables_initializer())

        fn2 = tf.keras.models.clone_model(fn1)
        result_2 = fn2([x, x])

        variable_names = [x.name for x in fn1.variables]
        for variable_name, variable_1, variable_2 in zip(
                variable_names, fn1.get_weights(), fn2.get_weights()):
            self.assertEqual(variable_1.shape, variable_2.shape)

            if 'kernel' in variable_name:
                self.assertNotAllClose(variable_1, variable_2)

        self.assertEqual(
            len(set(fn1.trainable_variables)
                & set(fn2.trainable_variables)),
            0)

        with self.assertRaises(ValueError):
            # TODO(hartikainen): investigate why this fails
            result_1_predict = fn1.predict([x_np, x_np])
            result_1_eval = tf.keras.backend.eval(result_1)

            result_2_predict = fn2.predict([x_np, x_np])
            result_2_eval = tf.keras.backend.eval(result_2)

            self.assertEqual(fn1.name, fn2.name)
            self.assertEqual(result_1_predict.shape, result_2_predict.shape)

            self.assertAllEqual(result_1_predict, result_1_eval)
            self.assertAllEqual(result_2_predict, result_2_eval)

    def test_without_name(self):
        fn = feedforward_model(
            output_size=1,
            hidden_layer_sizes=(6, 4, 2))

        self.assertEqual(fn.name, 'feedforward_model')


if __name__ == '__main__':
    tf.test.main()
