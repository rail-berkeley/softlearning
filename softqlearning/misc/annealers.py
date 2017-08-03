def step_annealer(step_fraction, step_interval):
    def anneal(val, itr):
        return val * step_fraction**(itr // step_interval)

    return anneal
