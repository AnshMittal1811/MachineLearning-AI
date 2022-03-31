__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '3/11/2020 4:47 PM'

import collections
import json
import math
import sys
import time

from pyspark import SparkConf, SparkContext

USER_ID = 'user_id'
USER_PROFILE = 'user_profile'
USER_INDEX = "user_index"
BUSINESS_ID = 'business_id'
BUSINESS_PROFILE = 'business_profile'
BUSINESS_INDEX = "business_index"
TYPE = "type"


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


def computeCosineSimilarity(profile1, profile2):
    """
    computer cosine similarity by two given words set
    :param profile1:
    :param profile2:
    :return: a float number
    """
    if len(profile1) != 0 and len(profile2) != 0:
        profile1 = set(profile1)
        profile2 = set(profile2)
        numerator = len(profile1.intersection(profile2))
        denominator = math.sqrt(len(profile1)) * math.sqrt(len(profile2))
        return numerator / denominator
    else:
        return 0.0


if __name__ == '__main__':
    start = time.time()
    # define input variables
    test_file_path = "../data/test_review2.json"
    export_model_file_path = "../out/task4.model"
    output_file_path = "../out/task2.predict"

    # test_file_path = sys.argv[1]
    # export_model_file_path = sys.argv[2]
    # output_file_path = sys.argv[3]

    # spark settings
    conf = SparkConf().setMaster("local") \
        .setAppName("ay_hw_3_task2_predict") \
        .set("spark.executor.memory", "4g") \
        .set("spark.driver.memory", "4g")
    sc = SparkContext(conf=conf)

    # ======================== Preprocessing Data ==========================
    # read the original model file
    model_file_lines = sc.textFile(export_model_file_path) \
        .map(lambda row: json.loads(row))

    # generate a dict which enable to map user_id_str to user_index
    # => dict(user id str: user index (uidx))
    # => e.g. {'---1lKK3aKOuomHnwAkAow': 0, 'xxx': int, ...} user count: 26184
    USER_INDEX_DICT = model_file_lines.filter(lambda kv: kv[TYPE] == USER_INDEX) \
        .map(lambda kv: {kv[USER_ID]: kv[USER_INDEX]}) \
        .flatMap(lambda kv_items: kv_items.items()).collectAsMap()
    REVERSED_USER_INDEX_DICT = {v: k for k, v in USER_INDEX_DICT.items()}

    # generate a dict which enable to map business_id_str to business_index
    # => dict(business id str: business index (bidx))
    # => e.g. {'--9e1ONYQuAa-CB_Rrw7Tw': 0, 'xxx': int, ...} user count: 10235
    BUSINESS_INDEX_DICT = model_file_lines.filter(lambda kv: kv[TYPE] == BUSINESS_INDEX) \
        .map(lambda kv: {kv[BUSINESS_ID]: kv[BUSINESS_INDEX]}) \
        .flatMap(lambda kv_items: kv_items.items()).collectAsMap()
    REVERSED_BUS_INDEX_DICT = {v: k for k, v in BUSINESS_INDEX_DICT.items()}

    # get user profile from reading model file
    # => dict(uidx: keywords index)
    # => {1363: [20480, 20483, 6, 1024...], ...}
    USER_PROFILE_DICT = model_file_lines.filter(lambda kv: kv[TYPE] == USER_PROFILE) \
        .map(lambda kv: {kv[USER_INDEX]: kv[USER_PROFILE]}) \
        .flatMap(lambda kv_items: kv_items.items()).collectAsMap()

    # get business profile from reading model file
    # => dict(bidx: keywords index)
    # => {1363: [20480, 20483, 6, 1024...], ...}
    BUSINESS_PROFILE_DICT = model_file_lines.filter(lambda kv: kv[TYPE] == BUSINESS_PROFILE) \
        .map(lambda kv: {kv[BUSINESS_INDEX]: kv[BUSINESS_PROFILE]}) \
        .flatMap(lambda kv_items: kv_items.items()).collectAsMap()

    # ======================== Make Predicationf ==========================
    predict_result = sc.textFile(test_file_path).map(lambda row: json.loads(row)) \
        .map(lambda kv: (kv[USER_ID], kv[BUSINESS_ID])) \
        .map(lambda kv: (USER_INDEX_DICT.get(kv[0], -1), BUSINESS_INDEX_DICT.get(kv[1], -1))) \
        .filter(lambda uid_bid: uid_bid[0] != -1 and uid_bid[1] != -1) \
        .map(lambda kv: ((kv), computeCosineSimilarity(USER_PROFILE_DICT.get(kv[0], set()),
                                                       BUSINESS_PROFILE_DICT.get(kv[1], set())))) \
        .filter(lambda kv: kv[1] > 0.01) \
        .map(lambda kv: {"user_id": REVERSED_USER_INDEX_DICT[kv[0][0]],
                         "business_id": REVERSED_BUS_INDEX_DICT[kv[0][1]],
                         "sim": kv[1]})

    export2File(predict_result.collect(), output_file_path)
    print("Duration: %d s." % (time.time() - start))
