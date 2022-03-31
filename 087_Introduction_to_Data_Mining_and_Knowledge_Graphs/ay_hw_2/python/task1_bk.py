__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '2/10/2020 10:05 PM'

import collections
import copy
import math
import sys
import time
from functools import reduce
from itertools import combinations
from operator import add

from pyspark import SparkContext

# TODO change the number of bucket will have different result
BUCKET_NUMBER = 99


def hash_func(combination):
    result = sum(map(lambda x: int(x), list(combination)))
    return result % BUCKET_NUMBER


def check_proper_subset(combination, candidate_list):
    """
    check if this combination's proper subset in candidate_list
    :param combination: assume its shape is n
    :param candidate_list: then item's shape in this list need to be n - 1
    :return: True of False
    """
    length = len(combination)
    for item in combinations(combination, length - 1):
        if len(item) == 1: item = item[0]
        if item not in candidate_list:
            return False
    return True


def check_bitmap(combination, bitmap):
    """
    check if its hash result in bitmap
    :param combination:
    :param bitmap:
    :return:
    """
    return bitmap[hash_func(combination)]


def wrapper(singleton_list):
    """
    reformat str item into tuple
    :param singleton_list:
    :return:
    """
    return [tuple(item.split(",")) for item in singleton_list]


def shrink_basket(basket, frequent_singleton):
    """
    we dont need to compute basket_item which is not frequent_single,
    basically we can return the interaction about this two set
    :param basket:
    :param frequent_singleton:
    :return:
    """
    return sorted(list(set(basket).intersection(set(frequent_singleton))))


def find_candidate_itemset(data_baskets, original_support, whole_length):
    """
    using PCY to find frequent itemset from subset basket
    :param data_baskets: subset baskets
    :param original_support:
    :param whole_length:
    :return: all candidate itemsets list
    """

    # compute support threshold in subset baskets
    support, data_baskets = gen_ps_threshold(data_baskets, original_support, whole_length)
    baskets_list = list(data_baskets)
    # print("baskets_list -> ", baskets_list)
    all_candidate_dict = collections.defaultdict(list)
    # first phrase of PCY algorithm, acquiring frequent_singleton and bitmap
    frequent_singleton, bitmap = init_singleton_and_bitmap(baskets_list, support)
    index = 1
    candidate_list = frequent_singleton
    all_candidate_dict[str(index)] = wrapper(frequent_singleton)

    # the second phrase, third phrase .... until the candidate list is empty
    while None is not candidate_list and len(candidate_list) > 0:
        index += 1
        temp_counter = collections.defaultdict(list)
        temp_bitmap = [0 for _ in range(BUCKET_NUMBER)]
        for basket in baskets_list:
            # we dont need to compute basket_item which is not frequent_single
            basket = shrink_basket(basket, frequent_singleton)
            if len(basket) >= index:
                for pair in combinations(basket, index):
                    if check_bitmap(pair, bitmap) and \
                            check_proper_subset(pair, candidate_list):
                        temp_counter[pair].append(1)

                for triplet in combinations(basket, index + 1):
                    key = hash_func(triplet)
                    temp_bitmap[key] = (temp_bitmap[key] + 1)

        # filter the temp_counter
        filtered_dict = dict(filter(lambda elem: len(elem[1]) >= support, temp_counter.items()))
        print("index ->", index)
        print("filtered_dict -> ", filtered_dict)
        # generate new candidate list
        candidate_list = sorted(list(filtered_dict.keys()))
        if len(candidate_list) == 0:
            break
        all_candidate_dict[str(index)] = candidate_list
        # generate new bitmap
        bitmap = list(map(lambda value: True if value >= support else False, temp_bitmap))

    # convert 2d-array into 1d array
    yield reduce(lambda val1, val2: val1 + val2, all_candidate_dict.values())


def init_singleton_and_bitmap(baskets, support):
    """
    first phrase of PCY algorithm
    :param baskets:
    :param support:
    :return: frequent_singleton: a list of sorted frequent singleton =>  ['100', '101', '102'...
                bitmap: a boolean list => [True, False, True ...
    """
    bitmap = [0 for _ in range(BUCKET_NUMBER)]
    temp_counter = collections.defaultdict(list)
    for basket in baskets:
        # find frequent singleton
        for item in basket:
            temp_counter[item].append(1)

        # find frequent bucket
        for pair in combinations(basket, 2):
            key = hash_func(pair)
            bitmap[key] = (bitmap[key] + 1)

    filtered_dict = dict(filter(lambda elem: len(elem[1]) >= support, temp_counter.items()))
    frequent_singleton = sorted(list(filtered_dict.keys()))
    bitmap = list(map(lambda value: True if value >= support else False, bitmap))

    return frequent_singleton, bitmap


def count_frequent_itemset(data_baskets, candidate_pairs):
    """
    count how many time each candidate item occurred in the sub baskets
    :param data_baskets: sub baskets
    :param candidate_pairs: all candidate pairs
    :return: C, v
    """
    baskets_list = list(data_baskets)
    temp_counter = collections.defaultdict(list)

    for basket in baskets_list:
        max_length = len(basket)
        for index in range(1, max_length + 1):
            for pairs in combinations(basket, index):
                if pairs in set(candidate_pairs):
                    temp_counter[pairs].append(1)

    yield [tuple((key, sum(value))) for key, value in temp_counter.items()]


def gen_ps_threshold(partition, support, whole_length):
    """
    generate each partition's support threshold
    :param partition:
    :param support:
    :param whole_length: the original rdd's size
    :return: support threshold in this partition
    """
    partition = copy.deepcopy(list(partition))
    return math.ceil(support * len(list(partition)) / whole_length), partition


def reformat(itemset_data):
    """
    reformat pairs which length is 1 ('100',), -> ('100'),
    and a line break after each subset who has a same length
    :param itemset_data: a list of paris (singletons, pairs, triples, etc.)
    :return: a formatted str
    """
    temp_index = 1
    result_str = ""
    for pair in itemset_data:
        if len(pair) == 1:
            result_str += str("(" + str(pair)[1:-2] + "),")

        elif len(pair) != temp_index:
            result_str = result_str[:-1]
            result_str += "\n\n"
            temp_index = len(pair)
            result_str += (str(pair) + ",")
        else:
            result_str += (str(pair) + ",")

    return result_str[:-1]


def export_2_file(candidate_data, frequent_data, file_path):
    with open(file_path, 'w+') as output_file:
        str_result = 'Candidates:\n' + reformat(candidate_data) + '\n\n' \
                     + 'Frequent Itemsets:\n' + reformat(frequent_data)
        output_file.write(str_result)
        output_file.close()


if __name__ == '__main__':
    start = time.time()
    case_number = "2"  # 1 for Case 1 and 2 for Case 2
    support_threshold = "9"
    input_csv_path = "../data/small2.csv"
    output_file_path = "../out/output6.txt"

    # case_number = sys.argv[1]  # 1 for Case 1 and 2 for Case 2
    # support_threshold = sys.argv[2]
    # input_csv_path = sys.argv[3]
    # output_file_path = sys.argv[4]
    partition_number = 2

    sc = SparkContext.getOrCreate()

    raw_rdd = sc.textFile(input_csv_path, partition_number)
    # skip the first row => csv header
    header = raw_rdd.first()
    data_rdd = raw_rdd.filter(lambda line: line != header)
    whole_data_size = None
    basket_rdd = None

    if 1 == int(case_number):
        # frequent businesses market-basket model
        basket_rdd = data_rdd.map(lambda line: (line.split(',')[0], line.split(',')[1])) \
            .groupByKey().map(lambda user_items: (user_items[0], sorted(list(set(list(user_items[1])))))) \
            .map(lambda item_users: item_users[1])

    elif 2 == int(case_number):
        # frequent user market-basket model
        basket_rdd = data_rdd.map(lambda line: (line.split(',')[1], line.split(',')[0])) \
            .groupByKey().map(lambda item_users: (item_users[0], sorted(list(set(list(item_users[1])))))) \
            .map(lambda item_users: item_users[1])

    # implement SON Algorithm
    # phrase 1 subset of data -> (F,1) -> distinct -> sort
    whole_data_size = basket_rdd.count()

    candidate_itemset = basket_rdd.mapPartitions(
        lambda partition: find_candidate_itemset(
            data_baskets=partition, original_support=int(support_threshold), whole_length=whole_data_size)) \
        .flatMap(lambda pairs: pairs).distinct() \
        .sortBy(lambda pairs: (len(pairs), pairs)).collect()

    print(candidate_itemset)
    # phrase 2 subset of data + candidate_pairs -> (C, v) -> reduceByKey(add) -> filter
    frequent_itemset = basket_rdd.mapPartitions(
        lambda partition: count_frequent_itemset(data_baskets=partition,
                                                 candidate_pairs=candidate_itemset)) \
        .flatMap(lambda pairs: pairs).reduceByKey(add) \
        .filter(lambda pair_count: pair_count[1] >= int(support_threshold)) \
        .map(lambda pair_count: pair_count[0]) \
        .sortBy(lambda pairs: (len(pairs), pairs)).collect()

    print(frequent_itemset)
    export_2_file(candidate_data=candidate_itemset,
                  frequent_data=frequent_itemset,
                  file_path=output_file_path)

    print("Duration: %d s." % (time.time() - start))
