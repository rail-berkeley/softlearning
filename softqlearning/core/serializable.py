import tensorflow as tf

from rllab.core.serializable import Serializable


class ScopedSerializable(Serializable):
    def __init__(self, locals_):
        super(Serializable, self).__init__(locals_)

    def quick_init(self, locals_):
        self.__scope = tf.get_variable_scope().name
        super(ScopedSerializable, self).quick_init(locals_)

    def __getstate__(self):
        d = super(ScopedSerializable, self).__getstate__()
        d['scope'] = self.__scope
        return d

    def __setstate__(self, d):
        with tf.variable_scope(d['scope']):
            super(ScopedSerializable, self).__setstate__(d)