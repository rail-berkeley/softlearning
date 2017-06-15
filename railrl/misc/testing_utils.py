import math
import numpy as np


def is_binomial_trial_likely(n, p, num_success):
    mean = n * p
    std = math.sqrt(n * p * (1 - p))
    margin = 3 * std
    return mean - margin < num_success < mean + margin


def are_np_array_iterables_equal(np_itr1, np_itr2, threshold=1e-5):
    # in case generators were passed in
    np_list1 = list(np_itr1)
    np_list2 = list(np_itr2)
    return (
        len(np_list1) == len(np_list2) and
        all(are_np_arrays_equal(arr1, arr2, threshold=threshold)
            for arr1, arr2 in zip(np_list1, np_list2))
    )


def are_np_arrays_equal(arr1, arr2, threshold=1e-5):
    if arr1.shape != arr2.shape:
        return False
    return (np.abs(arr1 - arr2) <= threshold).all()


def is_list_subset(list1, list2):
    for a in list1:
        if a not in list2:
            return False
    return True


def are_dict_lists_equal(list1, list2):
    return is_list_subset(list1, list2) and is_list_subset(list2, list1)

