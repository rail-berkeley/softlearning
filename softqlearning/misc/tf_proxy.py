import sys


class _Wrap(object):
    """ Wrapper class that wraps tf.Tensor instances returned by tensorflow
    functions in a SerializableTensor instance that inherits both tf.Tensor and
    Serializable. This is useful if we need to serialize a tensorflow graph
    to save it to a file or pass it to a parallel process.
    """
    def __getattr__(self, attr):
        # Make these classes visible to everyone.
        if attr == 'TensorProxy':
            return TensorProxy
        if attr == 'SerializableTensor':
            return SerializableTensor

        tf_func = tf.__dict__[attr]

        def _wrap(*args, **kwargs):
            return TensorProxy(tf_func, *args, **kwargs)

        return _wrap

# Redirects all module/function look-ups to _Wrap.__getattr__().
sys.modules[__name__] = _Wrap()

# Need to do imports here since above line hides earlier imports in python2.
import tensorflow as tf
from future.utils import with_metaclass
from rllab.core.serializable import Serializable


class _MetaClass(type):
    """
    Modifies the arithmetic magic functions (__add__, __sub__, etc.) so that
    returned tensors are wrapped in SerializableTensor instances.
    """
    def __init__(cls, name, bases, dct):
        super(_MetaClass, cls).__init__(name, bases, dct)

        def make_tensor_proxy(a):
            def tensor_proxy(self, *args, **kwargs):
                wrapped_tensor = TensorProxy(getattr(self._wrapped_tensor, a),
                                             *args, **kwargs)
                return wrapped_tensor
            return tensor_proxy
#
        for attr in tf.Tensor.OVERLOADABLE_OPERATORS:
            setattr(cls, attr, make_tensor_proxy(attr))


class SerializableTensor(with_metaclass(_MetaClass, Serializable, tf.Tensor)):
    """ Proxy class that makes tf.Tensor Serializable. """
    def __init__(self, tensor_to_wrap):
        # Only inherited classes should be instantiated.
        assert type(self) is not SerializableTensor
        assert isinstance(tensor_to_wrap, tf.Tensor)

        self._wrapped_tensor = tensor_to_wrap
        self.__dict__.update(tensor_to_wrap.__dict__)


class TensorProxy(SerializableTensor):
    def __init__(self, tf_func, *args, **kwargs):
        Serializable.quick_init(self, locals())

        # TODO: does not work if tf_func returns a list of tensors.
        tensor_to_wrap = tf_func(*args, **kwargs)
        super(TensorProxy, self).__init__(tensor_to_wrap)
