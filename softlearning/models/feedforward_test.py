import numpy as np
import tensorflow as tf

from softlearning.models.feedforward import feedforward_model


class FeedforwardTest(tf.test.TestCase):

    def test_clone_model(self):
        """Make sure that cloning works and clones can predict."""
        output_shape = (5, )
        x_np = np.random.uniform(0, 1, (1, 13)).astype(np.float32)
        x = tf.constant(x_np)

        fn1 = feedforward_model(
            output_shape=output_shape,
            hidden_layer_sizes=(6, 4, 2),
            name='feedforward_function')
        result_1 = fn1([x, x]).numpy()

        fn2 = tf.keras.models.clone_model(fn1)
        result_2 = fn2([x, x]).numpy()

        variable_names = [x.name for x in fn1.variables]
        for variable_name, variable_1, variable_2 in zip(
                variable_names, fn1.get_weights(), fn2.get_weights()):
            self.assertEqual(variable_1.shape, variable_2.shape)

            if 'kernel' in variable_name:
                self.assertNotAllClose(variable_1, variable_2)

        self.assertEqual(
            len(set((v1.experimental_ref() for v1 in fn1.trainable_variables))
                &
                set((v2.experimental_ref() for v2 in fn2.trainable_variables))),
            0)

        result_1_predict = fn1.predict((x_np, x_np))
        result_2_predict = fn2.predict((x_np, x_np))

        self.assertEqual(fn1.name, fn2.name)
        self.assertEqual(result_1_predict.shape, result_2_predict.shape)

        self.assertAllEqual(result_1_predict, result_1)
        self.assertAllEqual(result_2_predict, result_2)

    def test_without_name(self):
        fn = feedforward_model(
            output_shape=(1, ),
            hidden_layer_sizes=(6, 4, 2))

        self.assertEqual(fn.name, 'feedforward_model')


if __name__ == '__main__':
    tf.test.main()
