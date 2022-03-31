__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '3/6/2020 4:30 PM'

import collections
import itertools
import json
import math
import random
import time

from pyspark import SparkContext, SparkConf

USER_ID = 'user_id'
BUSINESS_ID = 'business_id'


def flatMixedList(dict_list):
    """

    :param dict_list:
    :return:
    """
    result = collections.defaultdict(list)
    for item in dict_list:
        result[list(item.keys())[0]] = list(item.values())[0]
    return result


def computeJaccard(set1, set2):
    return float(float(len(set(set1) & set(set2))) / float(len(set(set1) | set(set2))))


def export_2_file(json_array, file_path):
    str_result = ""
    with open(file_path, 'w+') as output_file:
        for item in json_array:
            str_result += json.dumps(item) + "\n"
        output_file.write(str_result)
        output_file.close()


def removeDuplicateIds(uid_bid_list):
    uid_set = set()
    bid_set = set()
    for item in uid_bid_list:
        uid_set.add(item[0])
        bid_set.add(item[1])

    return sorted(uid_set), sorted(bid_set)


if __name__ == '__main__':
    start = time.time()
    input_json_path = "../data/train_review.json"
    output_file_path = "../out/ground_truth_pair2.res"

    conf = SparkConf().setMaster("local") \
        .setAppName("ay_hw_3_truth") \
        .set("spark.executor.memory", "16g") \
        .set("spark.driver.memory", "16g")

    sc = SparkContext(conf=conf)

    input_lines = sc.textFile(input_json_path).map(lambda row: json.loads(row))

    # collect distinct user and distinct business
    # => generate dict(distinct uid: index) & dict(distinct business_id: index)
    # => e.g. {'-2QGc6Lb0R027lz0DpWN1A': 1, 'xxx': int} user count: 26184
    # => e.g. {'--9e1ONYQuAa-CB_Rrw7Tw': 0, 'xxx': int} business count: 10253
    uid_bid_list = input_lines.map(lambda kv: (kv[USER_ID], kv[BUSINESS_ID])).collect()
    user_ids, bus_ids = removeDuplicateIds(uid_bid_list)
    user_index_dict = dict(zip(user_ids, range(0, len(user_ids))))
    bus_index_dict = dict(zip(bus_ids, range(0, len(bus_ids))))

    # tuple(distinct user_id, [ list of business indexes ])
    # => e.g. [(2, [836, 918, 1607, 1807, 2715....])]
    bidx_uidx_rdd = input_lines.map(lambda kv: (bus_index_dict[kv[BUSINESS_ID]], user_index_dict[kv[USER_ID]])) \
        .groupByKey().map(lambda bidx_uidx: (bidx_uidx[0], sorted(set(bidx_uidx[1])))).sortBy(
        lambda pair: pair[0])

    candidate_pair = list(itertools.combinations(bidx_uidx_rdd.map(lambda kv: kv[0]).collect(), 2))

    bid_uidxs_dict = bidx_uidx_rdd.map(lambda kv: {kv[0]: set(kv[1])}).collect()
    bid_uidxs_flat_dict = flatMixedList(bid_uidxs_dict)

    reversed_bid_bidx_dict = {v: k for k, v in bus_index_dict.items()}

    # truth_rdd = sc.parallelize(candidate_pair) \
    #     .map(lambda pair: (pair, computeJaccard(bid_uidxs_flat_dict[pair[0]],
    #                                             bid_uidxs_flat_dict[pair[1]]))) \
    #     .filter(lambda kv: kv[1] >= 0.05) \
    #     .map(lambda kv: {"b1": reversed_bid_bidx_dict[kv[0][0]],
    #                      "b2": reversed_bid_bidx_dict[kv[0][1]],
    #                      "sim": kv[1]})
    truth_rdd = sc.parallelize(candidate_pair) \
        .map(lambda pair: (pair, computeJaccard(bid_uidxs_flat_dict[pair[0]],
                                                bid_uidxs_flat_dict[pair[1]]))) \
        .filter(lambda kv: kv[1] >= 0.05) \
        .map(lambda kv: kv[0])

    f = open("../out/truth_pair", "w")
    f.write(str(truth_rdd.collect()))
    f.close()

    # export_2_file(truth_rdd.collect(), output_file_path)

    print("Duration: %d s." % (time.time() - start))
