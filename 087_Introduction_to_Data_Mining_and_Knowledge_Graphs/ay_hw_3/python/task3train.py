__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '3/12/2020 4:17 PM'

import collections
import itertools
import json
import math
import sys
import time
import random
from pyspark import SparkConf, SparkContext

USER_ID = 'user_id'
BUSINESS_ID = 'business_id'
SCORE = "stars"
CO_RATED_THRESHOLD = 3
ITEM_BASED_MODEL = "item_based"
USER_BASED_MODEL = "user_based"
NUM_OF_HASH_FUNC = 30
BANDS = 30
JACCARD_SIMILARITY_THRESHOLD = 0.01


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
            return ((param_a * input_x + param_b) % 23333333333) % param_m

        return apply_funcs

    param_as = random.sample(range(1, sys.maxsize - 1), num_of_func)
    param_bs = random.sample(range(0, sys.maxsize - 1), num_of_func)
    for a, b in zip(param_as, param_bs):
        func_list.append(build_func(a, b, baskets))

    return func_list


def computeSimilarity(dict1, dict2):
    """
    compute Pearson Correlation Similarity
    :param dict1:
    :param dict2:
    :return: a float number
    """
    co_rated_user = list(set(dict1.keys()) & (set(dict2.keys())))
    val1_list, val2_list = list(), list()
    [(val1_list.append(dict1[user_id]),
      val2_list.append(dict2[user_id])) for user_id in co_rated_user]

    avg1 = sum(val1_list) / len(val1_list)
    avg2 = sum(val2_list) / len(val2_list)

    numerator = sum(map(lambda pair: (pair[0] - avg1) * (pair[1] - avg2), zip(val1_list, val2_list)))

    if numerator == 0:
        return 0
    denominator = math.sqrt(sum(map(lambda val: (val - avg1) ** 2, val1_list))) * \
                  math.sqrt(sum(map(lambda val: (val - avg2) ** 2, val2_list)))
    if denominator == 0:
        return 0

    return numerator / denominator


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


def flatMixedList(dict_list):
    """
    flat the dict_list into a big dict
    :param dict_list: [{a: 1}, {b: 2}, {c: 3}, ...]
    :return: a dict {a:!, b:2, c:3,...}
    """
    result = collections.defaultdict(list)
    for item in dict_list:
        result[list(item.keys())[0]] = list(item.values())[0]
    return result


def existNRecords(dict1, dict2):
    """
    check if whether these two set contain N number of same item or not
    :param dict1:
    :param dict2:
    :return: Boolean Value
    """
    if dict1 is not None and dict2 is not None:
        return True if len(set(dict1.keys()) & set(dict2.keys())) >= CO_RATED_THRESHOLD else False
    return False


def applyHashFuncs(hash_funcs, index):
    return list(map(lambda func: func(index), hash_funcs))


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


def computeJaccard(dict1, dict2):
    """
    compute Jaccard Similarity
    :param dict1:
    :param dict2:
    :return: a float number
    """
    if dict1 is not None and dict2 is not None:
        users1 = set(dict1.keys())
        users2 = set(dict2.keys())
        if len(users1 & users2) >= CO_RATED_THRESHOLD:
            if float(float(len(users1 & users2)) / float(len(users1 | users2))) >= JACCARD_SIMILARITY_THRESHOLD:
                return True

    return False


if __name__ == '__main__':
    start = time.time()
    train_file_path = "../data/train_review2.json"
    export_model_file_path = "../out/task3user.model"  # task3user.model
    model_type = "user_based"  # either "item_based" or "user_based"

    # train_file_path = sys.argv[1]
    # export_model_file_path = sys.argv[2]
    # model_type = sys.argv[3]

    conf = SparkConf().setMaster("local") \
        .setAppName("ay_hw_3_task3_train") \
        .set("spark.executor.memory", "4g") \
        .set("spark.driver.memory", "4g")
    sc = SparkContext(conf=conf)

    input_lines = sc.textFile(train_file_path).map(lambda row: json.loads(row)) \
        .map(lambda kv: (kv[USER_ID], kv[BUSINESS_ID], kv[SCORE]))

    # collect (sorted & distinct) user and tokenize them
    # => generate dict(distinct user id: index(uidx))
    # => e.g. {'-2QGc6Lb0R027lz0DpWN1A': 1, 'xxx': int, ...} user count: 26184
    user_index_dict = input_lines.map(lambda kvv: kvv[0]).distinct() \
        .sortBy(lambda item: item).zipWithIndex().map(lambda kv: {kv[0]: kv[1]}) \
        .flatMap(lambda kv_items: kv_items.items()).collectAsMap()
    reversed_index_user_dict = {v: k for k, v in user_index_dict.items()}

    # collect (sorted & distinct) business index and tokenize them
    # => generate dict(distinct business_id: index(bidx))
    # => e.g. {'--9e1ONYQuAa-CB_Rrw7Tw': 0, 'xxx': int, ....} business count: 10253
    bus_index_dict = input_lines.map(lambda kvv: kvv[1]).distinct() \
        .sortBy(lambda item: item).zipWithIndex().map(lambda kv: {kv[0]: kv[1]}) \
        .flatMap(lambda kv_items: kv_items.items()).collectAsMap()
    reversed_index_bus_dict = {v: k for k, v in bus_index_dict.items()}
    candidate_pair = None

    if model_type == ITEM_BASED_MODEL:
        # group original data by bidx, and remove those unpopular business (rated time < 3)
        # tuple(bidx, (uidx, score))
        # [(5306, [(3662, 5.0), (3218, 5.0), (300, 5.0),..]), ()
        shrunk_bid_uids_rdd = input_lines \
            .map(lambda kv: (bus_index_dict[kv[1]], (user_index_dict[kv[0]], kv[2]))) \
            .groupByKey().mapValues(lambda uid_score: list(uid_score)) \
            .filter(lambda bid_uid_score: len(bid_uid_score[1]) >= CO_RATED_THRESHOLD) \
            .mapValues(lambda vals: [{uid_score[0]: uid_score[1]} for uid_score in vals]) \
            .mapValues(lambda val: flatMixedList(val))

        candidate_bids = shrunk_bid_uids_rdd.map(lambda bid_uids: bid_uids[0]).coalesce(2)

        # convert shrunk_bid_uids_rdd into dict form
        # dict(bidx: dict(uidx: score))
        # => e.g. {5306: defaultdict(<class 'list'>, {3662: 5.0, 3218: 5.0, 300: 5.0...}),
        bid_uid_dict = shrunk_bid_uids_rdd \
            .map(lambda bid_uid_score: {bid_uid_score[0]: bid_uid_score[1]}) \
            .flatMap(lambda kv_items: kv_items.items()).collectAsMap()

        # generate all possible pair between candidate bidx
        # and compute the pearson similarity
        candidate_pair = candidate_bids.cartesian(candidate_bids) \
            .filter(lambda id_pair: id_pair[0] < id_pair[1]) \
            .filter(lambda id_pair: existNRecords(bid_uid_dict[id_pair[0]],
                                                  bid_uid_dict[id_pair[1]])) \
            .map(lambda id_pair: (id_pair,
                                  computeSimilarity(bid_uid_dict[id_pair[0]],
                                                    bid_uid_dict[id_pair[1]]))) \
            .filter(lambda kv: kv[1] > 0) \
            .map(lambda kv: {"b1": reversed_index_bus_dict[kv[0][0]],
                             "b2": reversed_index_bus_dict[kv[0][1]],
                             "sim": kv[1]})

    elif model_type == USER_BASED_MODEL:
        hash_funcs = genHashFuncs(NUM_OF_HASH_FUNC, len(bus_index_dict) * 2)

        # group original data by bidx, and remove those unpopular business (rated time < 3)
        # tuple(bidx, (uidx, score))
        # [(5306, [(3662, 5.0), (3218, 5.0), (300, 5.0),..]), ()
        shrunk_bid_uids_rdd = input_lines \
            .map(lambda kv: (bus_index_dict[kv[1]], (user_index_dict[kv[0]], kv[2]))) \
            .groupByKey().mapValues(lambda uid_score: list(uid_score)) \
            .filter(lambda bid_uid_score: len(bid_uid_score[1]) >= CO_RATED_THRESHOLD) \
            .persist()

        # build min hash signature for every business index (bus_idx)
        # and generate user_index pair
        # tuple(uidx1, uidx2)
        # => [(24752, [0, 0, 0, 8, 0, 0,...]), (1666, [0, 0, 8, 3,...
        # => [(3218, 4128), (300, 3662), (300, 4128), (3218, 3662), (30 ...
        uidx_pair = shrunk_bid_uids_rdd \
            .flatMap(lambda bid_uid_score: [(uid_score[0],
                                             applyHashFuncs(hash_funcs, bid_uid_score[0]))
                                            for uid_score in bid_uid_score[1]]) \
            .reduceByKey(getMinValue) \
            .flatMap(lambda kv: [(tuple(chunk), kv[0]) for chunk in splitList(kv[1], BANDS)]) \
            .groupByKey().map(lambda kv: sorted(set(kv[1]))).filter(lambda val: len(val) > 1) \
            .flatMap(lambda uid_list: [pair for pair in itertools.combinations(uid_list, 2)])\
            .distinct()

        # convert shrunk_bid_uids_rdd into dict form
        # dict(uidx: dict{bidx:score,....})
        # => {2275: defaultdict(<class 'list'>, {4978: 4.0, 1025: 4.0, 545: 5.0,...
        uid_bids_dict = shrunk_bid_uids_rdd \
            .flatMap(lambda bid_uid_score: [(item[0], (bid_uid_score[0], item[1]))
                                            for item in bid_uid_score[1]]) \
            .groupByKey().mapValues(lambda val: list(set(val))) \
            .filter(lambda uidx_mixed: len(uidx_mixed[1]) >= CO_RATED_THRESHOLD) \
            .mapValues(lambda vals: [{bid_score[0]: bid_score[1]} for bid_score in vals]) \
            .mapValues(lambda val: flatMixedList(val)) \
            .map(lambda uid_bid_score: {uid_bid_score[0]: uid_bid_score[1]}) \
            .flatMap(lambda kv_items: kv_items.items()).collectAsMap()

        # generate all possible pair between candidate uidx
        # and compute the pearson similarity
        candidate_pair = uidx_pair.filter(lambda pair: computeJaccard(uid_bids_dict.get(pair[0], None),
                                                                      uid_bids_dict.get(pair[1], None))) \
            .map(lambda id_pair: (id_pair, computeSimilarity(uid_bids_dict[id_pair[0]],
                                                             uid_bids_dict[id_pair[1]]))) \
            .filter(lambda kv: kv[1] > 0) \
            .map(lambda kv: {"u1": reversed_index_user_dict[kv[0][0]],
                             "u2": reversed_index_user_dict[kv[0][1]],
                             "sim": kv[1]})

    export2File(candidate_pair.collect(), export_model_file_path)
    print("Duration: %d s." % (time.time() - start))
