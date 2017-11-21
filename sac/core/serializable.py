import tensorflow as tf

from rllab.core.serializable import Serializable


class ScopedSerializable(Serializable):
    def __init__(self, locals_):
        Serializable.__init__(self, locals_)
        self.__scope = None

    def quick_init(self, locals_):
        self.__scope = tf.get_variable_scope().name
        Serializable.quick_init(self, locals_)

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        d['scope'] = self.__scope
        return d

    def __setstate__(self, d):
        with tf.variable_scope(d['scope']):
            Serializable.__setstate__(self, d)


# TODO: this should go into rllab.core.serializable.Serializable.
def deep_clone(obj):
    assert isinstance(obj, Serializable)

    def maybe_deep_clone(o):
        if isinstance(o, Serializable):
            return deep_clone(o)
        else:
            return o

    d = obj.__getstate__()
    for key, val in d.items():
        d[key] = maybe_deep_clone(val)

    d['__args'] = list(d['__args'])  # Make args mutable.
    for i, val in enumerate(d['__args']):
        d['__args'][i] = maybe_deep_clone(val)

    for key, val in d['__kwargs']:
        d['__kwargs'][key] = maybe_deep_clone(val)

    out = type(obj).__new__(type(obj))
    # noinspection PyArgumentList
    out.__setstate__(d)

    return out
