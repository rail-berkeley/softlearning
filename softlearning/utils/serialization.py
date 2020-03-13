import inspect
import contextlib


_GLOBAL_CUSTOM_OBJECTS = {}
_GLOBAL_CUSTOM_NAMES = {}


OBJECT_UNDEFINED_CONFIG_KEY = 'object was saved without config'


class CustomObjectScope(object):
    """Provides a scope that changes to `_GLOBAL_CUSTOM_OBJECTS` cannot escape.

    Code within a `with` statement will be able to access custom objects
    by name. Changes to global custom objects persist
    within the enclosing `with` statement. At end of the `with` statement,
    global custom objects are reverted to state
    at beginning of the `with` statement.

    Example:

    Consider a custom object `MyObject` (e.g. a class):

    ```python
    ```
    """

    def __init__(self, *args):
        self.custom_objects = args
        self.backup = None

    def __enter__(self):
        self.backup = _GLOBAL_CUSTOM_OBJECTS.copy()
        for objects in self.custom_objects:
            _GLOBAL_CUSTOM_OBJECTS.update(objects)
        return self

    def __exit__(self, *args, **kwargs):
        _GLOBAL_CUSTOM_OBJECTS.clear()
        _GLOBAL_CUSTOM_OBJECTS.update(self.backup)


def custom_object_scope(*args):
    """Provides a scope that changes to `_GLOBAL_CUSTOM_OBJECTS` cannot escape.

    Convenience wrapper for `CustomObjectScope`.
    Code within a `with` statement will be able to access custom objects
    by name. Changes to global custom objects persist
    within the enclosing `with` statement. At end of the `with` statement,
    global custom objects are reverted to state
    at beginning of the `with` statement.

    Example:

    Consider a custom object `MyObject`

    ```python
      TODO(hartikainen): Add example.
    ```

    Arguments:
        *args: Variable length list of dictionaries of name, class pairs to add to
          custom objects.

    Returns:
        Object of type `CustomObjectScope`.
    """
    return CustomObjectScope(*args)


def get_custom_objects():
    """Retrieves a live reference to the global dictionary of custom objects.

    Updating and clearing custom objects using `custom_object_scope`
    is preferred, but `get_custom_objects` can
    be used to directly access `_GLOBAL_CUSTOM_OBJECTS`.

    Example:

    ```python
        get_custom_objects().clear()
        get_custom_objects()['MyObject'] = MyObject
    ```

    Returns:
        Global dictionary of names to classes (`_GLOBAL_CUSTOM_OBJECTS`).
    """
    return _GLOBAL_CUSTOM_OBJECTS


def serialize_softlearning_class_and_config(cls_name, cls_config):
    """Returns the serialization of the class with the given config."""
    return {'class_name': cls_name, 'config': cls_config}


def register_softlearning_serializable(package='Custom', name=None):
    """Registers an object with the Softlearning serialization framework.

    This decorator injects the decorated class or function into the Softlearning custom
    object dictionary, so that it can be serialized and deserialized without
    needing an entry in the user-provided custom object dict. It also injects a
    function that Softlearning will call to get the object's serializable string key.

    Note that to be serialized and deserialized, classes must implement the
    `get_config()` method. Functions do not have this requirement.

    The object will be registered under the key 'package>name' where `name`,
    defaults to the object name if not passed.

    Arguments:
      package: The package that this class belongs to.
      name: The name to serialize this class under in this package. If None,
        the class's name will be used.

    Returns:
      A decorator that registers the decorated class with the passed names.
    """

    def decorator(arg):
        """Registers a class with the Softlearning serialization framework."""
        class_name = name if name is not None else arg.__name__
        registered_name = package + '>' + class_name

        if inspect.isclass(arg) and not hasattr(arg, 'get_config'):
            raise ValueError(
                'Cannot register a class that does not have a get_config() method.')

        if registered_name in _GLOBAL_CUSTOM_OBJECTS:
            raise ValueError(
                '%s has already been registered to %s' %
                (registered_name, _GLOBAL_CUSTOM_OBJECTS[registered_name]))

        if arg in _GLOBAL_CUSTOM_NAMES:
            raise ValueError('%s has already been registered to %s' %
                             (arg, _GLOBAL_CUSTOM_NAMES[arg]))

        _GLOBAL_CUSTOM_OBJECTS[registered_name] = arg
        _GLOBAL_CUSTOM_NAMES[arg] = registered_name

        return arg

    return decorator


def get_registered_name(obj):
    """Returns the name registered to `obj` within the Softlearning framework.

    This function is part of the Softlearning serialization and deserialization
    framework. It maps objects to the string names associated with those objects
    for serialization/deserialization.

    Args:
      obj: The object to look up.

    Returns:
      The name associated with the object, or the default Python name if the
        object is not registered.
    """
    if obj in _GLOBAL_CUSTOM_NAMES:
        return _GLOBAL_CUSTOM_NAMES[obj]
    else:
        return obj.__name__


@contextlib.contextmanager
def skip_failed_serialization():
    global _SKIP_FAILED_SERIALIZATION
    prev = _SKIP_FAILED_SERIALIZATION
    try:
        _SKIP_FAILED_SERIALIZATION = True
        yield
    finally:
        _SKIP_FAILED_SERIALIZATION = prev


def get_registered_object(name, custom_objects=None, module_objects=None):
    """Returns the class with given `name` if it is registered with Softlearning.

    This function is part of the Softlearning serialization and deserialization
    framework. It maps strings to the objects associated with them for
    serialization/deserialization.

    Example:
    ```
      TODO(hartikainen): Add an example.
    ```

    Args:
      name: The name to look up.
      custom_objects: A dictionary of custom objects to look the name up in.
        Generally, custom_objects is provided by the user.
      module_objects: A dictionary of custom objects to look the name up in.
        Generally, module_objects is provided by midlevel library implementers.

    Returns:
      An instantiable class associated with 'name', or None if no such class
        exists.
    """
    if name in _GLOBAL_CUSTOM_OBJECTS:
        return _GLOBAL_CUSTOM_OBJECTS[name]
    elif custom_objects and name in custom_objects:
        return custom_objects[name]
    elif module_objects and name in module_objects:
        return module_objects[name]
    return None


def serialize_softlearning_object(instance):
    """Serialize softlearning object into python dict."""
    if instance is None:
        return None

    if hasattr(instance, 'get_config'):
        name = get_registered_name(instance.__class__)
        try:
            config = instance.get_config()
        except NotImplementedError as e:
            if _SKIP_FAILED_SERIALIZATION:
                return serialize_softlearning_class_and_config(
                    name, {OBJECT_UNDEFINED_CONFIG_KEY: True})
            raise e
        serialization_config = {}
        for key, item in config.items():
            if isinstance(item, str):
                serialization_config[key] = item
                continue

            # Any object of a different type needs to be converted to string or dict
            # for serialization (e.g. custom functions, custom classes)
            try:
                serialized_item = serialize_softlearning_object(item)
                if isinstance(serialized_item, dict) and not isinstance(item, dict):
                    serialized_item['__passive_serialization__'] = True
                serialization_config[key] = serialized_item
            except ValueError:
                serialization_config[key] = item

        return serialize_softlearning_class_and_config(
            name, serialization_config)

    if hasattr(instance, '__name__'):
        return get_registered_name(instance)

    raise ValueError('Cannot serialize', instance)


def get_custom_objects_by_name(item, custom_objects=None):
    """Returns the item if it is in either local or global custom objects."""
    if item in _GLOBAL_CUSTOM_OBJECTS:
        return _GLOBAL_CUSTOM_OBJECTS[item]
    elif custom_objects and item in custom_objects:
        return custom_objects[item]
    return None


def class_and_config_for_serialized_softlearning_object(
        config,
        module_objects=None,
        custom_objects=None,
        printable_module_name='object'):
    """Returns the class name and config for a serialized softlearning object."""
    if (not isinstance(config, dict) or 'class_name' not in config or
        'config' not in config):
        raise ValueError(f"Improper config format: {config}")

    class_name = config['class_name']
    cls = get_registered_object(class_name, custom_objects, module_objects)
    if cls is None:
        raise ValueError(f"Unknown {printable_module_name}: {class_name}")

    cls_config = config['config']
    deserialized_objects = {}
    for key, item in cls_config.items():
        if isinstance(item, dict) and '__passive_serialization__' in item:
            deserialized_objects[key] = deserialize_softlearning_object(
                item,
                module_objects=module_objects,
                custom_objects=custom_objects,
                printable_module_name='config_item')
        elif (isinstance(item, str) and
              inspect.isfunction(get_registered_object(item, custom_objects))):
            # Handle custom functions here. When saving functions, we only save the
            # function's name as a string. If we find a matching string in the custom
            # objects during deserialization, we convert the string back to the
            # original function.
            # Note that a potential issue is that a string field could have a naming
            # conflict with a custom function name, but this should be a rare case.
            # This issue does not occur if a string field has a naming conflict with
            # a custom object, since the config of an object will always be a dict.
            deserialized_objects[key] = get_registered_object(item, custom_objects)
    for key, item in deserialized_objects.items():
        cls_config[key] = deserialized_objects[key]

    return (cls, cls_config)


def deserialize_softlearning_object(identifier,
                                    module_objects=None,
                                    custom_objects=None,
                                    printable_module_name='object'):
    if identifier is None:
        return None

    if isinstance(identifier, dict):
        # In this case we are dealing with a Softlearning config dictionary.
        config = identifier
        (cls, cls_config) = (
            class_and_config_for_serialized_softlearning_object(
                config, module_objects, custom_objects, printable_module_name))

        if hasattr(cls, 'from_config'):
            arg_spec = inspect.getfullargspec(cls.from_config)
            custom_objects = custom_objects or {}

            if 'custom_objects' in arg_spec.args:
                return cls.from_config(
                    cls_config,
                    custom_objects=dict(
                        list(_GLOBAL_CUSTOM_OBJECTS.items()) +
                        list(custom_objects.items())))
            with CustomObjectScope(custom_objects):
                return cls.from_config(cls_config)
        else:
            # Then `cls` may be a function returning a class.
            # in this case by convention `config` holds
            # the kwargs of the function.
            custom_objects = custom_objects or {}
            with CustomObjectScope(custom_objects):
                return cls(**cls_config)
    elif isinstance(identifier, str):
        object_name = identifier
        if custom_objects and object_name in custom_objects:
            obj = custom_objects.get(object_name)
        elif object_name in _GLOBAL_CUSTOM_OBJECTS:
            obj = _GLOBAL_CUSTOM_OBJECTS[object_name]
        else:
            obj = module_objects.get(object_name)
            if obj is None:
                raise ValueError(
                    f"Unknown {printable_module_name}: {object_name}")
        # Classes passed by name are instantiated with no args, functions are
        # returned as-is.
        if inspect.isclass(obj):
            return obj()
        return obj
    elif inspect.isfunction(identifier):
        # If a function has already been deserialized, return as is.
        return identifier
    else:
        raise ValueError("Could not interpret serialized"
                         f" {printable_module_name}: {identifier}")
