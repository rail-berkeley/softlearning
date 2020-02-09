import tensorflow as tf
from tensorflow.python.platform import test

from softlearning.utils import serialization
from softlearning import policies


class SerializeSoftlearningObjectTest(test.TestCase):

    def test_serialize_none(self):
        serialized = serialization.serialize_softlearning_object(None)
        self.assertEqual(serialized, None)
        deserialized = serialization.deserialize_softlearning_object(
            serialized)
        self.assertEqual(deserialized, None)

    def test_serialize_custom_class_with_default_name(self):

        @serialization.register_softlearning_serializable()
        class TestClass(object):

            def __init__(self, value):
                self._value = value

            def get_config(self):
                return {'value': self._value}

        serialized_name = 'Custom>TestClass'
        inst = TestClass(value=10)
        class_name = serialization._GLOBAL_CUSTOM_NAMES[TestClass]
        self.assertEqual(serialized_name, class_name)
        config = serialization.serialize_softlearning_object(inst)
        self.assertEqual(class_name, config['class_name'])
        new_inst = serialization.deserialize_softlearning_object(config)
        self.assertIsNot(inst, new_inst)
        self.assertIsInstance(new_inst, TestClass)
        self.assertEqual(10, new_inst._value)

        # Make sure registering a new class with same name will fail.
        with self.assertRaisesRegex(
                ValueError, ".*has already been registered.*"):
            @serialization.register_softlearning_serializable()  # pylint: disable=function-redefined
            class TestClass(object):

                def __init__(self, value):
                    self._value = value

                def get_config(self):
                    return {'value': self._value}

    def test_serialize_custom_class_with_custom_name(self):

        @serialization.register_softlearning_serializable(
            'TestPackage', 'CustomName')
        class OtherTestClass(object):

            def __init__(self, val):
                self._val = val

            def get_config(self):
                return {'val': self._val}

        serialized_name = 'TestPackage>CustomName'
        inst = OtherTestClass(val=5)
        class_name = serialization._GLOBAL_CUSTOM_NAMES[OtherTestClass]
        self.assertEqual(serialized_name, class_name)
        fn_class_name = serialization.get_registered_name(
            OtherTestClass)
        self.assertEqual(fn_class_name, class_name)

        cls = serialization.get_registered_object(fn_class_name)
        self.assertEqual(OtherTestClass, cls)

        config = serialization.serialize_softlearning_object(inst)
        self.assertEqual(class_name, config['class_name'])
        new_inst = serialization.deserialize_softlearning_object(config)
        self.assertIsNot(inst, new_inst)
        self.assertIsInstance(new_inst, OtherTestClass)
        self.assertEqual(5, new_inst._val)

    def test_serialize_custom_function(self):

        @serialization.register_softlearning_serializable()
        def my_fn():
            return 42

        serialized_name = 'Custom>my_fn'
        class_name = serialization._GLOBAL_CUSTOM_NAMES[my_fn]
        self.assertEqual(serialized_name, class_name)
        fn_class_name = serialization.get_registered_name(my_fn)
        self.assertEqual(fn_class_name, class_name)

        config = serialization.serialize_softlearning_object(my_fn)
        self.assertEqual(class_name, config)
        fn = serialization.deserialize_softlearning_object(config)
        self.assertEqual(42, fn())

        fn_2 = serialization.get_registered_object(fn_class_name)
        self.assertEqual(42, fn_2())

    def test_serialize_custom_class_without_get_config_fails(self):

        with self.assertRaisesRegex(
                ValueError,
                "Cannot register a class that does not have a get_config.*"):

            @serialization.register_softlearning_serializable(  # pylint: disable=unused-variable
                'TestPackage', 'TestClass')
            class TestClass(object):

                def __init__(self, value):
                    self._value = value

    def test_serializable_object(self):

        class SerializableInt(int):
            """A serializable object to pass out of a test layer's config."""

            def __new__(cls, value):
                return int.__new__(cls, value)

            def get_config(self):
                return {'value': int(self)}

            @classmethod
            def from_config(cls, config):
                return cls(**config)

        policy = policies.ContinuousUniformPolicy(
            action_range=([SerializableInt(-1)], [SerializableInt(1)]),
            input_shapes={'what': tf.TensorShape((3, ))},
            output_shape=(1, ),
            observation_keys=None,
            name='SerializableNestedInt')

        config = policies.serialize(policy)

        new_policy = policies.deserialize(
            config,
            custom_objects={
                'SerializableInt': SerializableInt,
            })

        self.assertEqual(new_policy._action_range, policy._action_range)
        self.assertEqual(new_policy._input_shapes, policy._input_shapes)
        self.assertIsInstance(new_policy._input_shapes['what'], tf.TensorShape)
        self.assertEqual(new_policy._output_shape, policy._output_shape)
        self.assertEqual(
            new_policy._observation_keys, policy._observation_keys)

        for action_bound in new_policy._action_range:
            for element in action_bound:
                self.assertIsInstance(element, SerializableInt)

    def test_nested_serializable_object(self):
        class SerializableInt(int):
            """A serializable object to pass out of a test layer's config."""

            def __new__(cls, value):
                return int.__new__(cls, value)

            def get_config(self):
                return {'value': int(self)}

            @classmethod
            def from_config(cls, config):
                return cls(**config)

        class SerializableNestedInt(int):
            """A serializable object containing another serializable object."""

            def __new__(cls, value, int_obj):
                obj = int.__new__(cls, value)
                obj.int_obj = int_obj
                return obj

            def get_config(self):
                return {'value': int(self), 'int_obj': self.int_obj}

            @classmethod
            def from_config(cls, config):
                return cls(**config)

        nested_int = SerializableInt(4)
        policy = policies.ContinuousUniformPolicy(
            action_range=(
                [SerializableNestedInt(-1, nested_int)],
                [SerializableNestedInt(1, nested_int)],
            ),
            input_shapes={'what': tf.TensorShape((3, ))},
            output_shape=(1, ),
            observation_keys=None,
            name='SerializableNestedInt')

        config = policies.serialize(policy)

        new_policy = policies.deserialize(
            config,
            custom_objects={
                'SerializableInt': SerializableInt,
                'SerializableNestedInt': SerializableNestedInt
            })

        # Make sure the string field doesn't get convert to custom object, even
        # they have same value.
        self.assertEqual(new_policy.name, 'SerializableNestedInt')

        self.assertEqual(new_policy._action_range, policy._action_range)
        self.assertEqual(new_policy._input_shapes, policy._input_shapes)
        self.assertIsInstance(new_policy._input_shapes['what'], tf.TensorShape)
        self.assertEqual(new_policy._output_shape, policy._output_shape)
        self.assertEqual(
            new_policy._observation_keys, policy._observation_keys)

        for action_bound in new_policy._action_range:
            for element in action_bound:
                self.assertIsInstance(element, SerializableNestedInt)
                self.assertIsInstance(element.int_obj, SerializableInt)
                self.assertEqual(element.int_obj, 4)

    def test_nested_serializable_fn(self):

        def serializable_fn(x):
            """A serializable function to pass out of a test layer's config."""
            return x

        class SerializableNestedInt(int):
            """A serializable object containing a serializable function."""

            def __new__(cls, value, fn):
                obj = int.__new__(cls, value)
                obj.fn = fn
                return obj

            def get_config(self):
                return {'value': int(self), 'fn': self.fn}

            @classmethod
            def from_config(cls, config):
                return cls(**config)

        policy = policies.ContinuousUniformPolicy(
            action_range=(
                [SerializableNestedInt(-1, serializable_fn)],
                [SerializableNestedInt(1, serializable_fn)],
            ),
            input_shapes={'what': tf.TensorShape((3, ))},
            output_shape=(1, ),
            observation_keys=None)

        config = policies.serialize(policy)

        new_policy = policies.deserialize(
            config,
            custom_objects={
                'serializable_fn': serializable_fn,
                'SerializableNestedInt': SerializableNestedInt
            })

        self.assertEqual(new_policy._action_range, policy._action_range)
        self.assertEqual(new_policy._input_shapes, policy._input_shapes)
        self.assertIsInstance(new_policy._input_shapes['what'], tf.TensorShape)
        self.assertEqual(new_policy._output_shape, policy._output_shape)
        self.assertEqual(
            new_policy._observation_keys, policy._observation_keys)

        for action_bound in new_policy._action_range:
            for element in action_bound:
                self.assertIsInstance(element, SerializableNestedInt)
                self.assertIs(element.fn, serializable_fn)
