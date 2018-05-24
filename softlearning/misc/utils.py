import datetime
import dateutil.tz
import os

from rllab.core.serializable import Serializable


def timestamp():
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    return now.strftime('%Y-%m-%d-%H-%M-%S-%f-%Z')


PROJECT_PATH = os.path.dirname(
    os.path.realpath(os.path.join(__file__, '..', '..')))


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
