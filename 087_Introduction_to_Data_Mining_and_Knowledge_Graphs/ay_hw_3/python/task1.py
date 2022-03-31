__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '3/4/2020 9:26 PM'

import itertools
import json
import math
import random
import sys
import time

from pyspark import SparkContext, SparkConf

USER_ID = 'user_id'
BUSINESS_ID = 'business_id'
NUM_OF_HASH_FUNC = 30
BANDS = 30
BUS_ID_1 = 'b1'
BUS_ID_2 = 'b2'


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


def getMinValue(list1, list2):
    """
    get min value in each element in two list
    :param list1: e.g. [3,6,2,6,8]
    :param list2: e.g. [1,4,5,6,2]
    :return: a list which contain the min value in each column
        e.g.  =======> [1,4,2,6,2]
    """
    return [min(val1, val2) for val1, val2 in zip(list1, list2)]


def splitList(value_list, chunk_num):
    """
    split a list in to several chunks
    :param value_list: a list whose shape is [N]
    :param chunk_num: the number of chunk you want to split
    :return: a list of list
    e.g. return [[1,a], [2,b], [3,c], [4,d]] and a + b + c + d = N
    """
    chunk_lists = list()
    size = int(math.ceil(len(value_list) / int(chunk_num)))
    for index, start in enumerate(range(0, len(value_list), size)):
        chunk_lists.append((index, hash(tuple(value_list[start:start + size]))))
    return chunk_lists


def computeJaccard(set1, set2):
    """
    compute Jaccard Similarity
    :param set1:
    :param set2:
    :return: a float number
    """
    return float(float(len(set(set1) & set(set2))) / float(len(set(set1) | set(set2))))


def verifySimilarity(candidate_pairs, index_data_dict,
                     reversed_index_dict, threshold):
    """
    iterate these candidate pairs,
            and compute the jaccard similarity from original data
    :param candidate_pairs: tuple(bidx1, bidx2)
    :param index_data_dict: dict(bidx: [uidx1, uidx2,...])
    :param reversed_index_dict: dict(bidx: bid_str)
    :param threshold: jaccard similarity threshold
    :return: a list of dict which contain truly similar
            bidx pair and theirs similarity
    """
    result = list()
    temp_set = set()
    for pair in candidate_pairs:
        if pair not in temp_set:
            temp_set.add(pair)
            similarity = computeJaccard(index_data_dict.get(pair[0], set()),
                                        index_data_dict.get(pair[1], set()))
            if similarity >= threshold:
                result.append({"b1": reversed_index_dict[pair[0]],
                               "b2": reversed_index_dict[pair[1]],
                               "sim": similarity})
    return result

def export2File(json_array, file_path):
    """
    export json content to a file
    :param json_array: a list of dict
    :param file_path: output file path
    :return: nothing, but a file
    """
    with open(file_path, 'w+') as output_file:
        for item in json_array:
            output_file.writelines(json.dumps(item) + "\n")
        output_file.close()


if __name__ == '__main__':
    start = time.time()
    # define input variables
    input_json_path = "../data/train_review2.json"
    output_file_path = "../out/task1_2.res"

    # input_json_path = sys.argv[1]
    # output_file_path = sys.argv[2]

    # spark settings
    conf = SparkConf().setMaster("local") \
        .setAppName("ay_hw_3_task1") \
        .set("spark.executor.memory", "4g") \
        .set("spark.driver.memory", "4g")
    sc = SparkContext(conf=conf)

    # ======================== Preprocessing Data ==========================
    # read the original json file
    input_lines = sc.textFile(input_json_path).map(lambda row: json.loads(row)) \
        .map(lambda kv: (kv[USER_ID], kv[BUSINESS_ID]))

    # collect (sorted & distinct) user and tokenize them
    # => generate dict(distinct user id: index(uidx))
    # => e.g. {'-2QGc6Lb0R027lz0DpWN1A': 1, 'xxx': int, ...} user count: 26184
    user_index_rdd = input_lines.map(lambda kv: kv[0]).distinct() \
        .sortBy(lambda item: item).zipWithIndex().map(lambda kv: {kv[0]: kv[1]}) \
        .flatMap(lambda kv_items: kv_items.items())

    user_index_dict = user_index_rdd.collectAsMap()
    reversed_index_user_dict = {v: k for k, v in user_index_dict.items()}

    # collect (sorted & distinct) business index and tokenize them
    # => generate dict(distinct business_id: index(bidx))
    # => e.g. {'--9e1ONYQuAa-CB_Rrw7Tw': 0, 'xxx': int, ....} business count: 10253
    bus_index_dict = input_lines.map(lambda kv: kv[1]).distinct() \
        .sortBy(lambda item: item).zipWithIndex().map(lambda kv: {kv[0]: kv[1]}) \
        .flatMap(lambda kv_items: kv_items.items()).collectAsMap()
    reversed_index_bus_dict = {v: k for k, v in bus_index_dict.items()}

    # generate a list of hash func obj
    # => an elegant way to call a hash func is [func(val) for func in hash_funcs]
    # => func(val) will return a hashed val based on the func and val you gave
    hash_funcs = genHashFuncs(NUM_OF_HASH_FUNC, len(user_index_dict) * 2)

    # hash all user idx (row_id) into hashed values
    # tuple(distinct user_id, [ list of user_idx(row_id) hashed indexes ])
    # => e.g. [('--9e1ONYQuAa-CB_Rrw7Tw', [3, 74, 53, 11, 31, 49, ...]),()]
    # => e.g. [(2, [3, 74, 53, 11, 31, 49, ...]),()]
    user_hashed_indexes_rdd = user_index_rdd \
        .map(lambda kv: (user_index_dict[kv[0]], [func(kv[1]) for func in hash_funcs]))

    # ======================== Algorithm Implement ==========================
    # find the list of businesses those who were rated by this user
    # tuple(user_index, [ list of business indexes ])
    # tuple(uidx, [ list of business indexes ])
    # => e.g. [(5646, [418, 4619, 4698, 310, 4343, 3642]), (4169, [20, ...])]
    uidx_bidxs_rdd = input_lines \
        .map(lambda kv: (user_index_dict[kv[0]], bus_index_dict[kv[1]])) \
        .groupByKey().map(lambda uidx_bidxs: (uidx_bidxs[0], list(set(uidx_bidxs[1]))))

    # find the list of users who comment this business before
    # tuple( bidx, [ list of user index ])
    # => e.g. [(2, {836, 918, 1607, 1807, 2715....})]
    bidx_uidxs_dict = input_lines.map(lambda kv: (bus_index_dict[kv[1]],
                                                  user_index_dict[kv[0]])) \
        .groupByKey().map(lambda bidx_uidxs: {bidx_uidxs[0]: list(set(bidx_uidxs[1]))}) \
        .flatMap(lambda kv_items: kv_items.items()).collectAsMap()

    # generate min-hash signature (tips: use rdd join)
    # tuple(bidx, list of user_hashed_indexes)
    # => e.g. [(1734, {34, 17, 26, 25, 4, 8, 2, ....}), (4936, {5, 3, 15, 26, 2...})]
    signature_matrix_rdd = uidx_bidxs_rdd.leftOuterJoin(user_hashed_indexes_rdd) \
        .map(lambda kvv: kvv[1]) \
        .flatMap(lambda bidx_uhidxs: [(bidx, bidx_uhidxs[1]) for bidx in bidx_uhidxs[0]]) \
        .reduceByKey(getMinValue).coalesce(2)

    # apply LSH algorithm
    # 1. split your signature matrix into several bands
    # 2. group by; using (band index, piece of signature) as key, original bidx as value
    # 3. generate candidate pairs
    # tuple(bidx1, bidx2)
    # => e.g. [(2056, 5306), (2056, 5034), (5306, 5034), (211, 1035), ...]
    candidate_pairs = signature_matrix_rdd \
        .flatMap(lambda kv: [(tuple(chunk), kv[0]) for chunk in splitList(kv[1], BANDS)]) \
        .groupByKey().map(lambda kv: list(kv[1])).filter(lambda val: len(val) > 1) \
        .flatMap(lambda bid_list: [pair for pair in itertools.combinations(bid_list, 2)])

    # iterate candidate pair and check theirs similarity from original data
    # to put it simply, find a set of users who were both rated these two businesses.
    # dict("b1": bid_str, "b2": bid_str, "sim": float)
    # => e.g. {"b1": "eo49Xbwss1EK...", "b2": "IT9ckwTnBWx...", "sim": 0.5}
    result_list = verifySimilarity(candidate_pairs=set(candidate_pairs.collect()),
                                   index_data_dict=bidx_uidxs_dict,
                                   reversed_index_dict=reversed_index_bus_dict,
                                   threshold=0.05)

    # export your finding
    export2File(result_list, output_file_path)

    print("Duration: %d s." % (time.time() - start))
