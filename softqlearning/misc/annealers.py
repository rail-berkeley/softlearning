from rllab.misc.overrides import overrides


class BaseAnnealer:
    def apply(self, val, itr):
        raise NotImplementedError


class StepAnnealer(BaseAnnealer):
    def __init__(self, step_coeff, step_interval):
        super(StepAnnealer, self).__init__()
        self._step_coeff = step_coeff
        self._step_interval = step_interval

    @overrides
    def apply(self, val, itr):
        return val * self._step_coeff**(itr // self._step_interval)
