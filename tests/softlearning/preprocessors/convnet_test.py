import pickle

import numpy as np
import tensorflow as tf

from softlearning.preprocessors.convnet import convnet_preprocessor


class TestConvnetPreprocessor(tf.test.TestCase):
    def test_serialize_deserialize(self):
        observation_shape = (200, )
        preprocessor = convnet_preprocessor(
            input_shapes=(observation_shape, ),
            image_shape=(8, 8, 3),
            output_size=5,
            conv_filters=(4, ),
            conv_kernel_sizes=((2, 2), ),
            pool_sizes=((2, 2), ),
            pool_strides=(2, ),
            dense_hidden_layer_sizes=(5, 5))

        session = tf.keras.backend.get_session()

        observations_np = np.random.uniform(
            -1, 1, (3, *observation_shape)).astype(np.float32)
        observations_tf = tf.constant(observations_np)

        preprocessed_tf = preprocessor([observations_tf])
        preprocessed_np = session.run(preprocessed_tf)
        preprocessed_predict_np = preprocessor.predict([observations_np])

        serialized = pickle.dumps(preprocessor)
        deserialized = pickle.loads(serialized)

        preprocessed_tf_2 = deserialized([observations_tf])
        preprocessed_np_2 = session.run(preprocessed_tf_2)
        preprocessed_predict_np_2 = deserialized.predict([observations_np])

        np.testing.assert_equal(
            preprocessed_predict_np, preprocessed_predict_np_2)
        np.testing.assert_equal(preprocessed_np, preprocessed_np_2)


if __name__ == '__main__':
    tf.test.main()
