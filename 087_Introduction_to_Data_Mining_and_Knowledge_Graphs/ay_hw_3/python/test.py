__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '3/20/2020 12:51 AM'

import random
import sys
from pprint import pprint


def genHashFuncs(num_of_func, baskets):
    """
    generate a list of hash funcs
    :param num_of_func: the number of hash func you need
    :param baskets: the number of baskets a hash func should use
    :return: a list of func object
    """
    func_list = list()

    def build_func(param_a, param_b, param_m):
        def apply_funcs(input_x):
            return ((param_a * input_x + param_b) % 233333333333) % param_m

        return apply_funcs

    param_as = random.sample(range(1, sys.maxsize - 1), num_of_func)
    param_bs = random.sample(range(0, sys.maxsize - 1), num_of_func)
    for a, b in zip(param_as, param_bs):
        func_list.append(build_func(a, b, baskets))

    return func_list


if __name__ == '__main__':
    hash_funcs = genHashFuncs(5, 10)
    array = [random.randint(1, 1000) for _ in range(1000)]

    res = dict()
    for val in array[:3]:
        res[val] = list(map(lambda func: func(val), hash_funcs))

    pprint(res)
