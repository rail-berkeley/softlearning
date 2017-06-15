from collections import OrderedDict

import numpy as np


def create_stats_ordered_dict(name, data):
    if len(data) == 0:
        return OrderedDict()

    if isinstance(data, tuple):
        ordered_dict = OrderedDict()
        for number, d in enumerate(data):
            sub_dict = create_stats_ordered_dict(
                "{0}_{1}".format(name, number),
                d,
            )
            ordered_dict.update(sub_dict)
        return ordered_dict
    else:
        return OrderedDict([
            (name + ' Mean', np.mean(data)),
            (name + ' Std', np.std(data)),
            (name + ' Max', np.max(data)),
            (name + ' Min', np.min(data)),
        ])
