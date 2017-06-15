import abc
from railrl.qfunctions.nn_qfunction import NNQFunction
from railrl.qfunctions.optimizable_q_function import OptimizableQFunction
from railrl.qfunctions.separable_q_function import SeparableQFunction


class NAFQFunction(NNQFunction,
                   OptimizableQFunction,
                   SeparableQFunction,
                   metaclass=abc.ABCMeta):
    pass
