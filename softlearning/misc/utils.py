import collections
import datetime
import dateutil.tz
import os

PROJECT_PATH = os.path.dirname(
    os.path.realpath(os.path.join(__file__, '..', '..')))


def timestamp():
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    return now.strftime('%Y-%m-%d-%H-%M-%S-%f-%Z')


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

    for key, val in d['__kwargs'].items():
        d['__kwargs'][key] = maybe_deep_clone(val)

    out = type(obj).__new__(type(obj))
    # noinspection PyArgumentList
    out.__setstate__(d)

    return out


def deep_update(d, u):
    for k, v in u.items():
        d[k] = (
            deep_update(d.get(k, {}), v)
            if isinstance(v, collections.Mapping)
            else v)

    return d


def get_git_rev():
    try:
        import git
        repo = git.Repo(os.getcwd())
        git_rev = repo.active_branch.commit.name_rev
    except:
        git_rev = None

    return git_rev


def flatten(unflattened, parent_key='', separator='.'):
    items = []
    for k, v in unflattened.items():
        if separator in k:
            raise ValueError(
                "Found separator ({}) from key ({})".format(separator, k))
        new_key = parent_key + separator + k if parent_key else k
        if isinstance(v, collections.MutableMapping) and v:
            items.extend(flatten(v, new_key, separator=separator).items())
        else:
            items.append((new_key, v))

    return dict(items)


def unflatten(flattened, separator='.'):
    result = {}
    for key, value in flattened.items():
        parts = key.split(separator)
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value

    return result
