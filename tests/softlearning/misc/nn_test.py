import tensorflow as tf

from softlearning.misc.nn import FeedforwardFunction


class FeedforwardFunctionTest(tf.test.TestCase):

    def setUp(self):
        tf.reset_default_graph()

    def testScopingReusesVariablesOnSingleInstance(self):
        output_size = 5
        x1 = tf.random_uniform((1, 13))
        x2 = tf.random_uniform((3, 13))

        fn = FeedforwardFunction(
            name='feedforward_function',
            hidden_layer_sizes=(6, 4, 2),
            output_size=output_size)

        self.assertTrue(not tf.trainable_variables())

        y1 = fn(x1)
        num_trainable_variables_1 = len(tf.trainable_variables())

        self.assertGreater(num_trainable_variables_1, 0)

        y2 = fn(x2)
        num_trainable_variables_2 = len(tf.trainable_variables())

        self.assertEqual(num_trainable_variables_1, num_trainable_variables_2)


    def testScopingCreatesNewVariablesOnAcrossInstances(self):
        output_size = 5
        x = tf.random_uniform((1, 13))

        fn1 = FeedforwardFunction(
            name='feedforward_function_1',
            hidden_layer_sizes=(6, 4, 2),
            output_size=output_size)

        fn2 = FeedforwardFunction(
            name='feedforward_function_2',
            hidden_layer_sizes=(6, 4, 2),
            output_size=output_size)

        self.assertTrue(not tf.trainable_variables())

        y1 = fn1(x)
        num_trainable_variables_1 = len(tf.trainable_variables())

        self.assertGreater(num_trainable_variables_1, 0)

        y2 = fn2(x)
        num_trainable_variables_2 = len(tf.trainable_variables())

        self.assertEqual(num_trainable_variables_1 * 2, num_trainable_variables_2)


if __name__ == '__main__':
    tf.test.main()
