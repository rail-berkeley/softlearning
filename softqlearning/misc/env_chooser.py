import importlib


class EnvChooser(object):
    @staticmethod
    def choose_env(env_name, **kwargs):
        module_name, class_name = env_name.rsplit('.', 1)
        env_module = importlib.import_module(module_name)
        return env_module.__getattribute__(class_name)(**kwargs)
