"""Tests for real_nvp distribution."""

from tensorflow.python.platform import test
import tensorflow as tf
import numpy as np

from sac.distributions import real_nvp

def TRANSLATION_FN_WITHOUT_BIAS(inputs):
    return 5 * inputs ** 2

def SCALE_FN_WITHOUT_BIAS(inputs):
    return 3 * inputs

BIAS = 2
def TRANSLATION_FN_WITH_BIAS(inputs):
    return TRANSLATION_FN_WITHOUT_BIAS(inputs) - BIAS

def SCALE_FN_WITH_BIAS(inputs):
    return SCALE_FN_WITHOUT_BIAS(inputs) - BIAS

DEFAULT_2D_INPUTS = np.array([
    [ 0, 0],
    [ 0, 1],
    [ 1, 0],
    [ 1, 1]
], dtype=np.float32)

class CouplingLayerTest(test.TestCase):
    def test_forward_and_jacobian(self):
        odd_layer = real_nvp.CouplingLayer(
            parity="odd",
            name="coupling_odd",
            translation_fn=TRANSLATION_FN_WITH_BIAS,
            scale_fn=SCALE_FN_WITH_BIAS
        )

        even_layer = real_nvp.CouplingLayer(
            parity="even",
            name="coupling_even",
            translation_fn=TRANSLATION_FN_WITH_BIAS,
            scale_fn=SCALE_FN_WITH_BIAS
        )

        inputs = tf.constant(DEFAULT_2D_INPUTS)
        odd_forward_out, odd_log_det_jacobian = odd_layer.forward_and_jacobian(
            inputs)
        even_forward_out, even_log_det_jacobian = even_layer.forward_and_jacobian(
            odd_forward_out)

        # Verify that the true side of the mask comes out as identity
        with self.test_session() as session:
            (inputs_num,
             odd_forward_out_num,
             even_forward_out_num,
             odd_log_det_jacobian_num,
             even_log_det_jacobian_num) = session.run((
                 inputs,
                 odd_forward_out,
                 even_forward_out,
                 odd_log_det_jacobian,
                 even_log_det_jacobian
             ))
        self.assertAllEqual(odd_forward_out_num[:, 1], inputs_num[:, 1])
        self.assertAllEqual(even_forward_out_num[:, 0], odd_forward_out_num[:, 0])
        self.assertAllEqual(
            odd_log_det_jacobian_num,
            SCALE_FN_WITH_BIAS(inputs_num[:, 1]))
        self.assertAllEqual(
            even_log_det_jacobian_num,
            SCALE_FN_WITH_BIAS(odd_forward_out_num[:, 0]))

    def test_forward_and_jacobian_nonlinearity_without_bias(self):
        layer_without_bias = real_nvp.CouplingLayer(
            parity="odd",
            name="coupling_odd",
            translation_fn=TRANSLATION_FN_WITHOUT_BIAS,
            scale_fn=SCALE_FN_WITHOUT_BIAS
        )

        layer_with_bias = real_nvp.CouplingLayer(
            parity="odd",
            name="coupling_odd",
            translation_fn=TRANSLATION_FN_WITH_BIAS,
            scale_fn=SCALE_FN_WITH_BIAS
        )

        inputs = tf.constant(DEFAULT_2D_INPUTS)

        (with_bias_forward_out,
         with_bias_log_det_jacobian
        ) = layer_with_bias.forward_and_jacobian(inputs)

        (without_bias_forward_out,
         without_bias_log_det_jacobian
        ) = layer_without_bias.forward_and_jacobian(inputs)

        with self.test_session() as session:
            (inputs_num,
             with_bias_forward_out_num,
             without_bias_forward_out_num) = session.run(
                 (
                     inputs,
                     with_bias_forward_out,
                     without_bias_forward_out
                 )
             )

        # Should return identity for "odd" axis
        # Should return different results for "with" and "without" bias
        self.assertAllEqual(
            with_bias_forward_out_num,
            np.array([[-2.        ,  0.        ],
                      [ 3.        ,  1.        ],
                      [-1.86466467,  0.        ],
                      [ 5.71828175,  1.        ]], dtype=np.float32)
        )

        self.assertAllEqual(
            without_bias_forward_out_num,
            np.array([[  0.        ,   0.        ],
                      [  5.        ,   1.        ],
                      [  1.        ,   0.        ],
                      [ 25.08553696,   1.        ]], dtype=np.float32)
        )

    def test_forward_backward_returns_identity(self):
        layer1 = real_nvp.CouplingLayer(
            parity="odd",
            name="coupling_1",
            translation_fn=TRANSLATION_FN_WITH_BIAS,
            scale_fn=SCALE_FN_WITH_BIAS
        )
        layer2 = real_nvp.CouplingLayer(
            parity="even",
            name="coupling_2",
            translation_fn=TRANSLATION_FN_WITH_BIAS,
            scale_fn=SCALE_FN_WITH_BIAS
        )

        inputs = tf.constant(DEFAULT_2D_INPUTS)
        forward_out, log_det_jacobian = layer1.forward_and_jacobian(inputs)
        forward_out, log_det_jacobian = layer2.forward_and_jacobian(forward_out)
        backward_out = layer2.backward(forward_out)
        backward_out = layer1.backward(backward_out)

        with self.test_session():
            self.assertAllClose(inputs.eval(), backward_out.eval())

    def test_get_mask(self):
        inputs = tf.constant([[0,0], [0,1], [1,0], [1,1]], dtype=tf.float32)
        EXPECTED = {"odd": [0,1], "even": [1,0]}
        for parity, expected_mask in EXPECTED.items():
            layer = real_nvp.CouplingLayer(
                parity, "coupling_" + parity, lambda x: None, lambda x: None)
            mask = layer.get_mask(inputs, tf.float32)

            with self.test_session():
                self.assertAllEqual(
                    mask.eval(), tf.constant(expected_mask).eval())

if __name__ == '__main__':
  test.main()
