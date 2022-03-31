__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '4/14/2020 10:32 PM'

import copy
import csv
import datetime
import json
import random
import sys
import time
from binascii import hexlify

from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext

CITY = 'city'
HOST = "localhost"
BATCH_DURATION = 5
LENGTH_OF_WINDOW = 30
SLIDING_INTERVAL = 10

NUM_OF_HASH_FUNC = 12
NUM_OF_BASKET = 10000
NUM_OF_BIT = 20


class CheckPoint:

    def __init__(self, export_file_path):
        self.export_file_path = export_file_path
        self.intermediate_records = dict()
        self.intermediate_records["header"] = ("Time", "Ground Truth", "Estimation")
        self.index = 1
        with open(self.export_file_path, "w", encoding="utf-8") as output_file:
            output_file.write("Time,Ground Truth,Estimation\n")

    def save(self, time_str, nof_truth, nof_estimation):
        self.intermediate_records[self.index] = (time_str, nof_truth, nof_estimation)
        with open(self.export_file_path, "a", encoding="utf-8") as output_file:
            output_file.write(time_str + "," + str(nof_truth) + "," + str(nof_estimation) + "\n")
        self.index += 1


class KMeans:

    def __init__(self, k: int, max_iterations: int):
        self.n_cluster = k
        self.max_iteration = max_iterations

    def fit(self, list_data: list, seed=666):
        """

        :param list_data:
        :param seed:
        :return:
        """
        self.list_data = list_data
        self._check_data_size()
        self._init_centroid(seed)
        epochs = 1
        while True:
            for item in self.list_data:
                temp_dict = dict()
                for centroid in self.centroid_info.keys():
                    temp_dict[(centroid, item)] = abs(self.centroid_info[centroid] - item)
                assigned_info = list(sorted(temp_dict.items(), key=lambda kv: kv[1]))[:1]
                self.cluster_result[assigned_info[0][0][0]].append(assigned_info[0][0][1])
                # print(self.cluster_result)

            previous_info, current_info = self._update_centroid_location()
            if not self._is_changed(previous_info, current_info) \
                    or epochs >= self.max_iteration:
                break
            # print(self.centroid_stable_flag)
            self._clear_cluster_result()
            epochs += 1

        return self.centroid_info, self.cluster_result

    def _init_centroid(self, seed: int):
        """
        randomly choose some data point as centroid
        :param seed: random seed
        :return:
        """
        random.seed(seed)
        self.centroid_info = dict()
        self.cluster_result = dict()
        self.centroid_stable_flag = dict()
        for key_index, chosen_value in enumerate(
                random.sample(self.list_data, self.n_cluster)):
            self.centroid_info.setdefault("c" + str(key_index), float(chosen_value))
            self.cluster_result.setdefault("c" + str(key_index), list())
            self.centroid_stable_flag.setdefault("c" + str(key_index), False)

    def _update_centroid_location(self):
        """
        compute the location of centroid based on the data points belong to that cluster
        :return: previous_centroid_info, current_centroid_info
        """
        previous_centroid_info = copy.deepcopy(self.centroid_info)
        for centroid, belongings in self.cluster_result.items():
            if not self.centroid_stable_flag.get(centroid):
                temp_list = list()
                temp_list.append(self.centroid_info.get(centroid))
                temp_list.extend(belongings)

                self.centroid_info[centroid] = float(sum(temp_list) / len(temp_list))

        return previous_centroid_info, self.centroid_info

    def _clear_cluster_result(self):
        """
        if the location of centroid change, we need to clear the previous result
        :return: Nothing, just reset the cluster result
        """
        for key in self.cluster_result.keys():
            self.cluster_result[key] = list()

    def _is_changed(self, dictA: dict, dictB: dict):
        for key in dictA.keys():
            if round(dictA.get(key), 1) != round(dictB.get(key), 1):
                self.centroid_stable_flag[key] = False
                return True
            else:
                self.centroid_stable_flag[key] = True

        return False

    def _check_data_size(self):
        """
        check the data size. modify the number of cluster
        if the data size less than the previous setting
        :return:
        """
        if len(self.list_data) < self.n_cluster:
            self.n_cluster = len(self.list_data)


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
            return format((param_a * input_x + param_b) % param_m, 'b') \
                .zfill(NUM_OF_BIT)

        return apply_funcs

    param_as = random.sample(range(1, sys.maxsize - 1), num_of_func)
    param_bs = random.sample(range(0, sys.maxsize - 1), num_of_func)
    for a, b in zip(param_as, param_bs):
        func_list.append(build_func(a, b, baskets))

    return func_list


def findLongestTrailing(list_obj):
    max_length = 0
    for str_value in list_obj:
        temp_count = 0
        for char in list(''.join(reversed(str_value))):
            if char == '0':
                temp_count += 1
            else:
                break
        max_length = max(max_length, temp_count)
    return max_length


def findCluster(sorted_list_obj, k=3):
    cluster_size = int(len(sorted_list_obj) / k)
    result = list()
    temp_list = list()
    for i in range(0, len(sorted_list_obj)):
        temp_list.append(sorted_list_obj[i])
        if i + 1 < len(sorted_list_obj):
            if sorted_list_obj[i + 1] - sorted_list_obj[i] >= 1 \
                    and len(temp_list) >= cluster_size:
                result.append(temp_list)
                temp_list = list()

    if len(temp_list) > 0:
        result.append(temp_list)
    return result


def findMedian(sorted_list_obj):
    list_size = len(sorted_list_obj)
    if list_size % 2 == 0:
        median = (sorted_list_obj[list_size // 2] + sorted_list_obj[list_size // 2 - 1]) / 2
        sorted_list_obj[0] = median
    if list_size % 2 == 1:
        median = sorted_list_obj[(list_size - 1) // 2]
        sorted_list_obj[0] = median
    return sorted_list_obj[0]


def applyFM(rdd):
    # save current time
    current_time = datetime.datetime.fromtimestamp(time.time()) \
        .strftime('%Y-%m-%d %H:%M:%S')
    # save ground truth number
    ground_truth = list(set(rdd.collect()))

    # [22632825789245816, 18965941658415737, 1335474462939231252334, ....]
    distinct_city_ids = list(map(lambda val: int(hexlify(val.encode('utf8')), 16), ground_truth))
    city_hashed_idx_2d_array = list()

    # apply each hash func on city_id
    # => [['00000010010100110001', '00000001010111010110', '0000....],
    #     [ ...]]
    # shape = (nof_ground_truth, nof_hash_func) => (*, 12)
    for city_id in distinct_city_ids:
        city_hashed_idx_2d_array.append([func(city_id) for func in hash_funcs])

    zero_length_list = list()
    # get all hash result by column index
    # e.g. get all result by which using 1 st hash func
    transposed_2d_array = list(map(list, zip(*city_hashed_idx_2d_array)))
    print("transposed_2d_array -> ", transposed_2d_array)
    for hidxes in transposed_2d_array:
        zero_length_list.append(findLongestTrailing(hidxes))

    print("zero_length_list -> ", zero_length_list)
    estimation_result_list = list(map(lambda val: 2 ** val, sorted(zero_length_list)))

    print("estimation_result_list -> ", estimation_result_list)
    _, clusters = KMeans(k=3, max_iterations=5).fit(estimation_result_list)

    print("clusters -> ", clusters)
    avg_list = list()
    for cluster in list(clusters.values()):
        if len(cluster) != 0:
            avg_list.append(sum(cluster) / len(cluster))
        else:
            avg_list.append(0)

    print("avg_list {} -> median {}".format(avg_list, int(sorted(avg_list)[1])))
    checkpoint.save(current_time, len(ground_truth), int(sorted(avg_list)[1]))


if __name__ == '__main__':
    start = time.time()
    # define input data stream port and output file path
    port = int('9999')
    output_file_path = "../out/task2_2.csv"

    # port = int(sys.argv[1])
    # output_file_path = sys.argv[2]

    # initialize streaming context
    conf = SparkConf().setMaster("local[*]") \
        .setAppName("ay_hw_6_task2") \
        .set("spark.executor.memory", "4g") \
        .set("spark.driver.memory", "4g")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("OFF")
    ssc = StreamingContext(sc, BATCH_DURATION)

    # connect to the stream server
    input_streaming = ssc.socketTextStream(HOST, port)

    # declare result obj -> checkpoint
    checkpoint = CheckPoint(export_file_path=output_file_path)

    # ======================== Initialize HASH Func ==========================
    hash_funcs = genHashFuncs(NUM_OF_HASH_FUNC, NUM_OF_BASKET)

    # ======================== Preprocessing Data ==========================
    # read the data from streaming
    # get ground truth
    # get original distinct city str
    # => [22632825789245816, 18965941658415737, 1335474462939231252334, ....]
    data_stream = input_streaming.window(LENGTH_OF_WINDOW, SLIDING_INTERVAL) \
        .map(lambda row: json.loads(row)).map(lambda kv: kv[CITY]) \
        .filter(lambda city_str: city_str != "") \
        .foreachRDD(applyFM)

    ssc.start()
    ssc.awaitTermination()
