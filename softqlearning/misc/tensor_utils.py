import tensorflow as tf
import numpy as np


def flatten_tensor_variables(ts):
    return tf.concat([tf.reshape(x, [-1]) for x in ts], 0)


def unflatten_tensor_variables(flat_arr, shapes, symb_arrs):
    arrs = []
    n = 0
    for (shape, symb_arr) in zip(shapes, symb_arrs):
        size = np.prod(list(shape))
        arr = tf.reshape(flat_arr[n:n + size], shape)
        arrs.append(arr)
        n += size
    return arrs
