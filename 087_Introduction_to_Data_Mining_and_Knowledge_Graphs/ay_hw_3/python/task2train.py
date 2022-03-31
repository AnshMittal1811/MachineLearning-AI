__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '3/10/2020 9:56 AM'

import collections
import json
import math
import re
import string
import sys
import time
from functools import reduce

USER_ID = 'user_id'
USER_PROFILE = 'user_profile'
USER_INDEX = "user_index"
BUSINESS_ID = 'business_id'
BUSINESS_PROFILE = 'business_profile'
BUSINESS_INDEX = "business_index"
REVIEW_TEXT = "text"
TOP_200 = 200
RARE_THRESHOLD = 3

from pyspark import SparkConf, SparkContext


def splitTextAndRemove(texts, stop_words):
    """
    split the text into words and remove those stopwords
    :param texts: unprocessed text
    :param stop_words: stopwords set
    :return:
    """
    word_list = list()
    for text in texts:
        text = text.translate(str.maketrans('', '', string.digits + string.punctuation))
        word_list.extend(
            list(filter(lambda word: word not in stop_words
                                     and word != ''
                                     and word not in string.ascii_lowercase,
                        re.split(r"[~\s\r\n]+", text))))

    return word_list


def countWords(words_list):
    """
    count the appearance of word in given words list
    :param words_list:
    :return: a sorted list which contain words and theirs appearance
    """
    counter_dict = collections.defaultdict(list)
    max_appearance_times = 0
    for word in words_list:
        if word in counter_dict.keys():
            counter_dict[word].append(1)
        else:
            counter_dict[word] = [1]

    max_appearance_times = max(
        len(reduce(lambda a, b: a if a > b else b,
                   counter_dict.values())), max_appearance_times)
    counter_dict = dict(filter(lambda kv: len(kv[1]) > RARE_THRESHOLD, counter_dict.items()))
    return sorted([(key, len(val), max_appearance_times) for key, val in counter_dict.items()],
                  key=lambda kv: kv[1], reverse=True)


def removeDuplicateIds(uid_bid_mixed_list):
    """
    find distinct user id str set and business id str set
    :param uid_bid_mixed_list: a list of tuple(uid_str, bid_str)
    :return: a set of user_id_str and a set of business_id_str
    """
    uid_set = set()
    bid_set = set()
    for item in uid_bid_mixed_list:
        uid_set.add(item[0])
        bid_set.add(item[1])

    return sorted(uid_set), sorted(bid_set)


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


def extendValue(list1, list2):
    """
    extend the list, keep the size same
    :param list1:
    :param list2:
    :return:
    """
    result = list(list1)
    result.extend(list2)
    return result


def wrapper(data, type, keys):
    """
    transform the data into json type
    :param data: different type of data
    :param type: profile's type
    :param keys: keys using for entitle the json element
    :return:
    """
    result = list()
    if isinstance(data, dict):
        for key, val in data.items():
            result.append({
                "type": type,
                keys[0]: key,
                keys[1]: val
            })
    elif isinstance(data, list):
        for kv in data:
            for key, val in kv.items():
                result.append({
                    "type": type,
                    keys[0]: key,
                    keys[1]: val
                })
    return result


if __name__ == '__main__':
    start = time.time()
    # define input variables
    train_file_path = "../data/train_review2.json"
    export_model_file_path = "../out/task5.model"
    stop_words_file_path = "../data/stopwords"

    # train_file_path = sys.argv[1]
    # export_model_file_path = sys.argv[2]
    # stop_words_file_path = sys.argv[3]

    # generate stopwords set
    stop_words_set = set(word.strip() for word in open(stop_words_file_path))

    # spark settings
    conf = SparkConf().setMaster("local") \
        .setAppName("ay_hw_3_task2_train") \
        .set("spark.executor.memory", "4g") \
        .set("spark.driver.memory", "4g")
    sc = SparkContext(conf=conf)
    model_content = list()

    # ======================== Preprocessing Data ==========================
    # read the original json file
    input_lines = sc.textFile(train_file_path).map(lambda row: json.loads(row))

    # collect (sorted & distinct) user and tokenize them
    # => generate dict(distinct user id: index(uidx))
    # => e.g. {'-2QGc6Lb0R027lz0DpWN1A': 1, 'xxx': int, ...} user count: 26184
    user_index_rdd = input_lines.map(lambda kv: kv[USER_ID]).distinct() \
        .sortBy(lambda item: item).zipWithIndex().map(lambda kv: {kv[0]: kv[1]}) \
        .flatMap(lambda kv_items: kv_items.items())

    user_index_dict = user_index_rdd.collectAsMap()

    # collect (sorted & distinct) business index and tokenize them
    # => generate dict(distinct business_id: index(bidx))
    # => e.g. {'--9e1ONYQuAa-CB_Rrw7Tw': 0, 'xxx': int, ....} business count: 10253
    bus_index_dict = input_lines.map(lambda kv: kv[BUSINESS_ID]).distinct() \
        .sortBy(lambda item: item).zipWithIndex().map(lambda kv: {kv[0]: kv[1]}) \
        .flatMap(lambda kv_items: kv_items.items()).collectAsMap()

    model_content.extend(wrapper(user_index_dict, USER_INDEX,
                                 keys=[USER_ID, USER_INDEX]))

    model_content.extend(wrapper(bus_index_dict, BUSINESS_INDEX,
                                 keys=[BUSINESS_ID, BUSINESS_INDEX]))

    # calculate the tf for each document (business)
    # 1. tokenize the business id
    # 2. split review text in to words
    # 3. count the word's appearance
    # tuple(bid, word, tf_value)
    # => e.g. [((5306, 'fries'), 1.0), ((5306, 'flippin'), 0.875),....]
    bid_words_tf_rdd = input_lines \
        .map(lambda kv: (bus_index_dict[kv[BUSINESS_ID]],
                         str(kv[REVIEW_TEXT].encode('utf-8')).lower())) \
        .groupByKey() \
        .mapValues(lambda texts: splitTextAndRemove(list(texts), stop_words_set)) \
        .map(lambda bid_text: (bid_text[0], countWords(bid_text[1]))) \
        .flatMap(lambda bid_words_vvs: [((bid_words_vvs[0], words_vv[0]),
                                         words_vv[1] / words_vv[2])
                                        for words_vv in bid_words_vvs[1]]).persist()

    # calculate the idf for each document (business)
    # 1. split review text in to words
    # 2. count the word's appearance
    # tuple(bid, word, idf_value)
    # => [((4, 'fries'), 3.4623849360266448), ((2057, 'fries'), 3.4623849360266448), ...]
    bid_words_idf_rdd = bid_words_tf_rdd \
        .map(lambda bid_words_tf: (bid_words_tf[0][1], bid_words_tf[0][0])) \
        .groupByKey().mapValues(lambda bids: list(set(bids))) \
        .flatMap(lambda word_bids: [((bid, word_bids[0]),
                                     math.log(len(bus_index_dict) / len(word_bids[1]), 2))
                                    for bid in word_bids[1]])

    # 1. group by tf idf score for each word
    # 2. select top 200 highest TF-IDF word which will serve as a business profile
    # tuple(bid, [list of top words]
    # => [(5306, ['flippin', 'fries', 'burger', 'specialty', 'heat',...
    bid_word_tf_idf_rdd = bid_words_tf_rdd.leftOuterJoin(bid_words_idf_rdd) \
        .mapValues(lambda tf_idf: tf_idf[0] * tf_idf[1]) \
        .map(lambda bid_word_val: (bid_word_val[0][0],
                                   (bid_word_val[0][1], bid_word_val[1]))) \
        .groupByKey() \
        .mapValues(lambda val: sorted(list(val), reverse=True,
                                      key=lambda item: item[1])[:TOP_200]) \
        .mapValues(lambda word_vals: [item[0] for item in word_vals])

    # tokenize the word into index
    # dict(word: int)
    # => e.g. {'fries': 0, 'good': 1, 'salad': 2, 'came': 3, 'us': 4, 'like': 5
    word_index_dict = bid_word_tf_idf_rdd \
        .flatMap(lambda kv: [(word, 1) for word in kv[1]]) \
        .groupByKey().map(lambda kv: kv[0]).zipWithIndex().map(lambda kv: {kv[0]: kv[1]}) \
        .flatMap(lambda kv_items: kv_items.items()).collectAsMap()

    print(word_index_dict)

    # tokenize the word into index for each document (business)
    # dict("type": "business_profile", "business_index": int, "business_profile": [])
    # => e.g. {"type": "business_profile",
    #          "business_index": 4552,
    #          "business_profile": [4882, 15346, ...]}
    bus_profile = bid_word_tf_idf_rdd.mapValues(lambda words: [word_index_dict[word] for word in words]) \
        .map(lambda bid_word_idxs: {bid_word_idxs[0]: bid_word_idxs[1]}).persist()

    bus_profile_data = bus_profile.collect()
    bus_profile_dict = flatMixedList(bus_profile_data)
    model_content.extend(wrapper(bus_profile_data, BUSINESS_PROFILE,
                                 keys=[BUSINESS_INDEX, BUSINESS_PROFILE]))

    # generate user profile
    # combine all the business profile which this user rated before
    # dict("type": "user_profile", "user_index": int, "user_profile": [])
    # => e.g. {"type": "user_profile", "user_index": 12991, "user_profile": [8195, 122
    user_profile = input_lines.map(lambda kv: (kv[USER_ID], kv[BUSINESS_ID])) \
        .groupByKey().map(lambda kv: (user_index_dict[kv[0]], list(set(kv[1])))) \
        .mapValues(lambda bids: [bus_index_dict[bid] for bid in bids]) \
        .flatMapValues(lambda bids: [bus_profile_dict[bid] for bid in bids]) \
        .reduceByKey(extendValue).filter(lambda uid_bids: len(uid_bids[1]) > 1) \
        .map(lambda uid_bids: {uid_bids[0]: list(set(uid_bids[1]))})

    model_content.extend(wrapper(user_profile.collect(), USER_PROFILE,
                                 keys=[USER_INDEX, USER_PROFILE]))

    export2File(model_content, export_model_file_path)

    print("Duration: %d s." % (time.time() - start))
