__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '4/14/2020 4:16 PM'

import csv
import json
import random
import sys
import time
from binascii import hexlify

from pyspark import SparkContext, SparkConf

CITY = 'city'
BUSINESS_ID = 'business_id'
NUM_OF_HASH_FUNC = 7
LENGTH_OF_BIT_ARRAY = 7000


def export2File(result, export_file_path):
    """
    export list content to a file
    :param result: result data
    :param export_file_path: output file path
    :return: nothing, but a file
    """
    with open(export_file_path, "w+", newline="") as output_file:
        writer = csv.writer(output_file, delimiter=' ')
        writer.writerow(result)


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


def makePredication(str_item, bit_array):
    if str_item is not None and str_item != "":
        str_index = int(hexlify(str_item.encode('utf8')), 16)
        signature = set([func(str_index) for func in hash_funcs])
        yield 1 if signature.issubset(bit_array) else 0
    else:
        yield 0


if __name__ == '__main__':
    start = time.time()
    # define input variables
    # input_first_json_path = "../data/business_first.json"
    # input_second_json_path = "../data/business_second.json"
    # output_file_path = "../out/task1.csv"

    input_first_json_path = sys.argv[1]
    input_second_json_path = sys.argv[2]
    output_file_path = sys.argv[3]

    # spark settings
    conf = SparkConf().setMaster("local[*]") \
        .setAppName("ay_hw_6_task1") \
        .set("spark.executor.memory", "4g") \
        .set("spark.driver.memory", "4g")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")

    # ======================== Preprocessing Data ==========================
    # read the training json file
    training_data_rdd = sc.textFile(input_first_json_path) \
        .map(lambda row: json.loads(row)).map(lambda kv: kv[CITY]).distinct() \
        .filter(lambda city_str: city_str != "") \
        .map(lambda city_str: int(hexlify(city_str.encode('utf8')), 16))

    # ======================== Initialize HASH Func ==========================
    hash_funcs = genHashFuncs(NUM_OF_HASH_FUNC, LENGTH_OF_BIT_ARRAY)

    city_hashed_indexes = training_data_rdd \
        .flatMap(lambda index: [func(index) for func in hash_funcs]).collect()

    existing_bit_array = set(city_hashed_indexes)

    # ======================== Preprocessing Data ==========================
    # read the test data json file
    result_rdd = sc.textFile(input_second_json_path) \
        .map(lambda row: json.loads(row)).map(lambda kv: kv[CITY]) \
        .flatMap(lambda city_str: makePredication(city_str, existing_bit_array))

    # ======================== Export Result File ==========================
    export2File(result_rdd.collect(), output_file_path)
    print("Duration: %d s." % (time.time() - start))
