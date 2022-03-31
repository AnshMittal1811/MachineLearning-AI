__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '3/12/2020 4:17 PM'

import collections
import json
import sys
import time

from pyspark import SparkConf, SparkContext

USER_ID = 'user_id'
USER_ID_1 = 'u1'
USER_ID_2 = 'u2'
BUSINESS_ID = 'business_id'
BUS_ID_1 = 'b1'
BUS_ID_2 = 'b2'
SIMILARITY = 'sim'
SCORE = 'stars'
N_NEIGHBORS = 3
AVG_BUSINESS_STAR = 3.823989
AVG_USER_STAR = 3.823989
ITEM_BASED_MODEL = "item_based"
USER_BASED_MODEL = "user_based"


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


def makePrediction(mixed, data_dict, model_type, avg_score_dict=None,
                   reversed_index_dict=None, N_NEIGHBORS=None):
    """

    :param mixed: tuple(bidx, [tuple(bidx, score), ...])  item_based
                or tuple(uidx, [tuple(uidx, score), ...])  user_based
    :param data_dict:
    :param model_type:
    :param avg_score_dict:
    :param reversed_index_dict:
    :param N_NEIGHBORS:
    :return: tuple(bidx, [(score, similarity),(),()])
    """
    if model_type == ITEM_BASED_MODEL:
        target_bid = mixed[0]  # bidx
        target_bid_str = reversed_index_dict.get(target_bid, "UNK")
        mixed_bids_score_list = list(mixed[1])  # list of tuple(bidx, score)
        result = list()
        for bids_score in mixed_bids_score_list:
            if target_bid < bids_score[0]:
                key = tuple((target_bid, bids_score[0]))
            else:
                key = tuple((bids_score[0], target_bid))

            result.append(tuple((bids_score[1], data_dict.get(key, 0))))

        score_sim_list = sorted(result, key=lambda item: item[1], reverse=True)[:N_NEIGHBORS]
        numerator = sum(map(lambda item: item[0] * item[1], score_sim_list))
        if numerator == 0:
            return tuple((target_bid, avg_score_dict.get(target_bid_str, AVG_BUSINESS_STAR)))
        denominator = sum(map(lambda item: abs(item[1]), score_sim_list))
        if denominator == 0:
            return tuple((target_bid, avg_score_dict.get(target_bid_str, AVG_BUSINESS_STAR)))

        return tuple((target_bid, numerator / denominator))

    else:
        target_uid = mixed[0]  # uidx
        target_uid_str = reversed_index_dict.get(target_uid, "UNK")
        mixed_uids_score_list = list(mixed[1])  # list of tuple(uidx, score)
        result = list()
        for uids_score in mixed_uids_score_list:
            if target_uid < uids_score[0]:
                key = tuple((target_uid, uids_score[0]))
            else:
                key = tuple((uids_score[0], target_uid))

            other_uid_str = reversed_index_dict.get(uids_score[0], "UNK")
            avg_score = avg_score_dict.get(other_uid_str, AVG_BUSINESS_STAR)
            # score, acg_score, similarity between users
            result.append(tuple((uids_score[1], avg_score, data_dict.get(key, 0))))

        numerator = sum(map(lambda item: (item[0] - item[1]) * item[2], result))
        if numerator == 0:
            return tuple((target_uid, avg_score_dict.get(target_uid_str, AVG_BUSINESS_STAR)))
        denominator = sum(map(lambda item: abs(item[2]), result))
        if denominator == 0:
            return tuple((target_uid, avg_score_dict.get(target_uid_str, AVG_BUSINESS_STAR)))

        return tuple((target_uid,
                      avg_score_dict.get(target_uid_str, AVG_USER_STAR) + (numerator / denominator)))


if __name__ == '__main__':
    start = time.time()
    train_file_path = "../data/train_review.json"
    test_file_path = "../data/test_review.json"
    export_model_file_path = "../out/task3user.model"
    output_file_path = "../out/task3user.predict"
    model_type = "user_based"  # either "item_based" or "user_based"
    bus_avg_file_path = "../data/business_avg.json"
    user_avg_file_path = "../data/user_avg.json"

    # train_file_path = sys.argv[1]
    # test_file_path = sys.argv[2]
    # export_model_file_path = sys.argv[3]
    # output_file_path = sys.argv[4]
    # model_type = sys.argv[5]
    # bus_avg_file_path = "../resource/asnlib/publicdata/business_avg.json"
    # user_avg_file_path = "../resource/asnlib/publicdata/user_avg.json"

    conf = SparkConf().setMaster("local") \
        .setAppName("ay_hw_3_task3_predict") \
        .set("spark.executor.memory", "4g") \
        .set("spark.driver.memory", "4g")
    sc = SparkContext(conf=conf)

    train_input_lines = sc.textFile(train_file_path).map(lambda row: json.loads(row)) \
        .map(lambda kv: (kv[USER_ID], kv[BUSINESS_ID], kv[SCORE])).persist()

    user_index_dict = train_input_lines.map(lambda kvv: kvv[0]).distinct() \
        .sortBy(lambda item: item).zipWithIndex().map(lambda kv: {kv[0]: kv[1]}) \
        .flatMap(lambda kv_items: kv_items.items()).collectAsMap()
    reversed_index_user_dict = {v: k for k, v in user_index_dict.items()}

    bus_index_dict = train_input_lines.map(lambda kvv: kvv[1]).distinct() \
        .sortBy(lambda item: item).zipWithIndex().map(lambda kv: {kv[0]: kv[1]}) \
        .flatMap(lambda kv_items: kv_items.items()).collectAsMap()
    reversed_index_bus_dict = {v: k for k, v in bus_index_dict.items()}

    output_pair = None
    if model_type == ITEM_BASED_MODEL:

        #  dict((bidx_pair): similarity)
        # => {(7415, 7567): 0.11736313170325506, (7415, 9653): 0.5222329678670935 ...}
        bid_pair_sim_dict = sc.textFile(export_model_file_path) \
            .map(lambda row: json.loads(row)) \
            .map(lambda kvv: {(bus_index_dict[kvv[BUS_ID_1]], bus_index_dict[kvv[BUS_ID_2]]):
                                  kvv[SIMILARITY]}) \
            .flatMap(lambda kv_items: kv_items.items()).collectAsMap()

        # tuple(uidx, [(bidx, score)])
        # => [(0, [(7819, 3.0), (10185, 3.0), (5437, 3.0) ...]), (), ()...]
        train_uid_bidx_score_rdd = train_input_lines.map(lambda kvv: (kvv[0], (kvv[1], kvv[2]))) \
            .groupByKey() \
            .map(lambda uid_bids: (user_index_dict[uid_bids[0]],
                                   [(bus_index_dict[bid_score[0]], bid_score[1]) for bid_score in
                                    list(set(uid_bids[1]))]))

        # tokenized uid and bidx from test file
        # tuple(uidx, bidx)
        # => [(20979, 1682), (13257, 8957), (20822, 932), (15374, 3977), (9611, 5172)]
        test_uid_bidx_rdd = sc.textFile(test_file_path).map(lambda row: json.loads(row)) \
            .map(lambda kv: (user_index_dict.get(kv[USER_ID], -1),
                             bus_index_dict.get(kv[BUSINESS_ID], -1))) \
            .filter(lambda uid_bid: uid_bid[0] != -1 and uid_bid[1] != -1)

        # read avg info from json file and convert it into dict
        # dict(bid_str: avg_score)
        # => {"MHiKdBFx4McRQONnuMbByw": 3.857142857142857, ...}
        bus_avg_dict = sc.textFile(bus_avg_file_path).map(lambda row: json.loads(row)) \
            .map(lambda kv: dict(kv)).flatMap(lambda kv_items: kv_items.items()) \
            .collectAsMap()

        # find the neighborhood set N of items rated by user u that are similar to item i
        # tuple(uidx, tuple(bidx, [tuple(score, similarity), ...]))
        # => [(20520, (8347, [(7318, 5.0), (4851, 5.0), (6188, 5.0), (2670, 2.0)
        output_pair = test_uid_bidx_rdd.leftOuterJoin(train_uid_bidx_score_rdd) \
            .mapValues(lambda mixed: makePrediction(mixed=tuple(mixed),
                                                    N_NEIGHBORS=N_NEIGHBORS,
                                                    avg_score_dict=bus_avg_dict,
                                                    data_dict=bid_pair_sim_dict,
                                                    reversed_index_dict=reversed_index_bus_dict,
                                                    model_type=ITEM_BASED_MODEL)) \
            .map(lambda kvv: {"user_id": reversed_index_user_dict[kvv[0]],
                              "business_id": reversed_index_bus_dict[kvv[1][0]],
                              "stars": kvv[1][1]})

    elif model_type == USER_BASED_MODEL:
        #  dict((uidx_pair): similarity)
        # => {(7415, 7567): 0.11736313170325506, (7415, 9653): 0.5222329678670935 ...}
        uid_pair_sim_dict = sc.textFile(export_model_file_path) \
            .map(lambda row: json.loads(row)) \
            .map(lambda kvv: {(user_index_dict[kvv[USER_ID_1]], user_index_dict[kvv[USER_ID_2]]):
                                  kvv[SIMILARITY]}) \
            .flatMap(lambda kv_items: kv_items.items()).collectAsMap()

        # find the set of user who rated item i before
        # tuple(bidx, [tuple(uidx, score)])
        # => [(6372, [(5807, 5.0), (13378, 1.0), (14411, 5.0)...]), ()]
        train_bid_uidx_score_rdd = train_input_lines \
            .map(lambda kvv: (bus_index_dict[kvv[1]], (user_index_dict[kvv[0]], kvv[2]))) \
            .groupByKey() \
            .map(lambda bid_uidxs: (bid_uidxs[0], [(uid_score[0], uid_score[1])
                                                   for uid_score in list(set(bid_uidxs[1]))]))

        # read avg info from json file and convert it into dict
        # dict(uid_str: avg_score)
        # => {"MHiKdBFx4McRQONnuMbByw": 3.857142857142857, ...}
        user_avg_dict = sc.textFile(user_avg_file_path).map(lambda row: json.loads(row)) \
            .ma.p(lambda kv: dict(kv)).flatMap(lambda kv_items: kv_items.items()) \
            .collectAsMap()

        # tokenized uid and bidx from test file
        # tuple(bidx, uidx)
        # => [(4871, 24954), (7557, 17243), (4593, 16426), (6791, 23814), (4383, 377)]
        test_bid_uidx_rdd = sc.textFile(test_file_path).map(lambda row: json.loads(row)) \
            .map(lambda kv: (bus_index_dict.get(kv[BUSINESS_ID], -1),
                             user_index_dict.get(kv[USER_ID], -1))) \
            .filter(lambda bid_uid: bid_uid[0] != -1 and bid_uid[1] != -1)

        output_pair = test_bid_uidx_rdd.leftOuterJoin(train_bid_uidx_score_rdd) \
            .mapValues(lambda mixed: makePrediction(mixed=tuple(mixed),
                                                    data_dict=uid_pair_sim_dict,
                                                    avg_score_dict=user_avg_dict,
                                                    reversed_index_dict=reversed_index_user_dict,
                                                    model_type=USER_BASED_MODEL)) \
            .map(lambda kvv: {"user_id": reversed_index_user_dict[kvv[1][0]],
                              "business_id": reversed_index_bus_dict[kvv[0]],
                              "stars": kvv[1][1]})

    export2File(output_pair.collect(), output_file_path)
    print("Duration: %d s." % (time.time() - start))
