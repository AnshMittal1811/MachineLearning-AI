__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '4/6/2020 8:25 PM'

import collections
import csv
import copy
import itertools
import json
import os
import random
import sys
import time
from math import sqrt
from pyspark import SparkConf, SparkContext


# os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
# os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'


class KMeans:

    def __init__(self, k: int, max_iterations: int):
        self.n_cluster = k
        self.max_iteration = max_iterations

    def fit(self, data_dict: dict, seed=666):
        """

        :param data_dict:
        :param seed:
        :return:
        """
        self.data_dict = data_dict
        self._check_data_size()  # this func might change the value of self.n_cluster
        self._init_centroid(seed)
        epochs = 1
        while True:
            # print("epochs -> ", epochs)
            for key in list(self.data_dict.keys()):
                # iterate every data point in the data,
                # and compute the distance between centroid and data point
                # finally, we get the min distance, whose corresponding data point
                # will be assigned to that cluster.
                # [(centroid, key), distance]
                # => [((925, 19), 7.159782595825231)]
                temp_dict = dict()
                for centroid in self.centroid_info.keys():
                    temp_dict[(centroid, key)] = computeDistance(self.centroid_info[centroid],
                                                                 self.data_dict[key])
                assigned_info = list(sorted(temp_dict.items(), key=lambda kv: kv[1]))[:1]
                # print("assigned_info ->", assigned_info)
                # {467: [18, 30, 32, 92, 97, 98], 925: [3, 4, 7, 11, ...
                self.cluster_result[assigned_info[0][0][0]].append(assigned_info[0][0][1])
                # print(self.cluster_result)

            previous_info, current_info = self._update_centroid_location()
            # print("previous_info -> ", previous_info)
            # print("current_info -> ", current_info)
            if not self._is_changed(previous_info, current_info) \
                    or epochs >= self.max_iteration:
                break
            # print(self.centroid_stable_flag)
            self._clear_cluster_result()
            epochs += 1

        return self.centroid_info, self.centroid_stat_info, self.cluster_result

    def _init_centroid(self, seed: int):
        """
        randomly choose some data point as centroid
        :param seed: random seed
        :return:
        """
        random.seed(seed)
        self.centroid_info = dict()
        self.centroid_stat_info = dict()
        self.cluster_result = dict()
        self.centroid_stable_flag = dict()
        for key_index, chosen_key in enumerate(
                random.sample(self.data_dict.keys(), self.n_cluster)):
            self.centroid_info.setdefault("c" + str(key_index), self.data_dict.get(chosen_key))
            self.centroid_stat_info.setdefault("c" + str(key_index), self.data_dict.get(chosen_key))
            self.cluster_result.setdefault("c" + str(key_index), list())
            self.centroid_stable_flag.setdefault("c" + str(key_index), False)

    def _update_centroid_location(self):
        """
        compute the location of centroid based on the data points belong to that cluster
        :return: previous_centroid_info, current_centroid_info
        """
        previous_centroid_info = copy.deepcopy(self.centroid_info)
        # print("update cluster_result ->", self.cluster_result)
        for centroid, belongings in self.cluster_result.items():
            if not self.centroid_stable_flag.get(centroid):
                temp_list = list()
                temp_list.append(self.centroid_info.get(centroid))
                for belonging in belongings:
                    temp_list.append(self.data_dict.get(belonging))

                self.centroid_info[centroid] = [sum(i) / len(i) for i in zip(*temp_list)]  # SUM/N
                self.centroid_stat_info[centroid] = [sum([v ** 2 for v in i]) / len(i) for i in
                                                     zip(*temp_list)]  # SUMSQ / N

        return previous_centroid_info, self.centroid_info

    def _clear_cluster_result(self):
        """
        if the location of centroid change, we need to clear the previous result
        :return: Nothing, just reset the cluster result
        """
        for key in self.cluster_result.keys():
            self.cluster_result[key] = list()
        # print("after clear -> ",self.cluster_result)

    def _is_changed(self, dictA: dict, dictB: dict):
        for key in dictA.keys():
            valueA = set(map(lambda value: round(value, 0), dictA.get(key)))
            valueB = set(map(lambda value: round(value, 0), dictB.get(key)))
            if len(valueA.difference(valueB)) == 0:
                self.centroid_stable_flag[key] = True
            else:
                self.centroid_stable_flag[key] = False
                return True

        return False

    def _check_data_size(self):
        """
        check the data size. modify the number of cluster
        if the data size less than the previous setting
        :return:
        """
        if len(self.data_dict.keys()) < self.n_cluster:
            self.n_cluster = len(self.data_dict.keys())


class Cluster:

    def __init__(self):
        self.centroid_info = None
        self.cluster_result = None
        self.signature = None

    def init(self, info: dict, statistics: dict, result: dict):
        self.centroid_info = info
        self.SUM_N = info
        self.SUMSQ_N = statistics
        self.cluster_result = result
        self.total_nof_point = 0
        self.dimension_of_data = len(list(info.values())[0])
        self.STD = dict()
        self.setSTD()

    def getClusterResult(self):
        return self.cluster_result

    def getClusterResultByKey(self, key):
        return list(self.cluster_result.get(key))

    def getSignature(self):
        return self.signature

    def getCentroidInfo(self):
        return self.centroid_info

    def getCentroidLocationByKey(self, key: str):
        return list(self.centroid_info.get(key))

    def getDimension(self):
        return self.dimension_of_data

    def getClusterNum(self):
        return len(self.centroid_info.keys())

    def getSUMSQ_NByKey(self, key: str):
        return list(self.SUMSQ_N.get(key))

    def setSTD(self):
        # print("previous STD -> ", self.STD)
        self.STD = dict()
        for key in self.SUM_N.keys():
            self.STD[key] = [sqrt(sq_n - sum_n ** 2) for (sq_n, sum_n)
                             in zip(self.SUMSQ_N.get(key), self.SUM_N.get(key))]
        # print("after setSTD func we get -> ", self.STD)

    def getSTD(self):
        return self.STD

    def getSTDByKey(self, key: str):
        return list(self.STD.get(key))

    def getPointNum(self):
        self.total_nof_point = 0
        for key, value in self.cluster_result.items():
            if type(value) == list:
                self.total_nof_point += len(value)
        return self.total_nof_point

    def updateCentroidInfo(self, temp_cluster_result, temp_data_dict):
        """
        avg(2,3,4,5,6) = 4  avg(2,3,4,5,6,6,9,13,17) = 65 / 9 = 7.2
        sum(4,6,9,13,17)) / 5 =  9.8    9.8 != 7.2
        compute the location of centroid based on these new data points
        assigned to this cluster
        :param temp_cluster_result: # => {'c1': [21318,234,2334,....]}
        :param temp_data_dict: # => {21318: [x,a,s,,f,f,...], 123123: [x,a,s,,f,f,...]}
        """

        if len(temp_cluster_result.keys()) > 0:
            previous_centroid_info = copy.deepcopy(self.centroid_info)
            previous_cluster_result = copy.deepcopy(self.cluster_result)
            previous_sumsq = copy.deepcopy(self.SUMSQ_N)
            for centroid, belongings in temp_cluster_result.items():
                temp_list = list()

                old_location1 = previous_centroid_info.get(centroid)
                length_of_old_result = len(previous_cluster_result.get(centroid))
                old_location1 = list(map(lambda v: v * length_of_old_result, old_location1))

                for belonging in belongings:
                    temp_list.append(temp_data_dict.get(belonging))

                # update SUM/N
                old_location2 = [sum(i) for i in zip(*temp_list)]
                total_count = len(temp_list) + length_of_old_result
                self.SUM_N[centroid] = self.centroid_info[centroid] = \
                    computeAVGByColumn(old_location1, old_location2,
                                       fixed_denominator=total_count)

                old_location3 = previous_sumsq.get(centroid)
                old_location3 = list(map(lambda v: v * length_of_old_result, old_location3))
                old_location4 = [sum([v ** 2 for v in i]) for i in zip(*temp_list)]
                # update SUMSQ / N
                self.SUMSQ_N[centroid] = computeAVGByColumn(old_location3, old_location4,
                                                            fixed_denominator=total_count)

            self.setSTD()  # refresh STD
            self.updateClusterResult(temp_cluster_result)  # refresh cluster belongings
        #     print("previous_info -> ", previous_centroid_info)
        #     print("current_info -> ", self.centroid_info)
        #     print("info -> updated centroid info from {} object".format(self.getSignature()))
        # else:
        #     print("info -> data's size is 0. don't need to update {} 's centroid info".format(self.getSignature()))

    def updateClusterResult(self, temp_cluster_result):
        """
        combine two dict
        :param temp_cluster_result:
        :return:
        """
        if len(temp_cluster_result.keys()) > 0:
            combined_result = collections.defaultdict(list)
            for key, val in itertools.chain(self.cluster_result.items(),
                                            temp_cluster_result.items()):
                combined_result[key] += val

            self.cluster_result = combined_result
        #     print("info -> updated cluster result from {} object".format(self.getSignature()))
        # else:
        #     print("info -> data's size is 0. don't need to update {} 's cluster result".format(self.getSignature()))


def computeAVGByColumn(list1, list2, fixed_denominator=None):
    temp_list = list()
    temp_list.append(list1)
    temp_list.append(list2)
    if fixed_denominator is None:
        return [sum(i) / len(i) for i in zip(*temp_list)]
    else:
        return [sum(i) / fixed_denominator for i in zip(*temp_list)]


class DS(Cluster):

    def __init__(self):
        Cluster.__init__(self)
        self.signature = "DS"

    # => {'c1': [21318,234,2334,....]} cs_cluster_result
    # => {'c1': [x,a,s,,f,f,...]} cs_cluster_centroid_info
    def mergeToOneCluster(self, ds_cluster_key: str,
                          cs_cluster_sumsq_info: list,
                          cs_cluster_centroid_info: list,
                          cs_cluster_result: list):
        old_location1 = self.getCentroidLocationByKey(ds_cluster_key)
        length_of_old_result = len(self.getClusterResultByKey(ds_cluster_key))
        old_location1 = list(map(lambda v: v * length_of_old_result, old_location1))

        length_of_new_assigned_result = len(cs_cluster_result)
        old_location2 = list(map(lambda v: v * length_of_new_assigned_result, cs_cluster_centroid_info))

        new_location = computeAVGByColumn(old_location1, old_location2,
                                          length_of_new_assigned_result + length_of_old_result)

        old_sumsq1 = list(map(lambda v: v * length_of_old_result, self.getSUMSQ_NByKey(ds_cluster_key)))
        old_sumsq2 = list(map(lambda v: v * length_of_new_assigned_result, cs_cluster_sumsq_info))
        new_sumsq = computeAVGByColumn(old_sumsq1, old_sumsq2,
                                       length_of_new_assigned_result + length_of_old_result)

        # update centroid info and SUM_N
        self.centroid_info.update({ds_cluster_key: new_location})
        # update cluster result
        self.cluster_result[ds_cluster_key].extend(cs_cluster_result)
        # update SUMSQ
        self.SUMSQ_N.update({ds_cluster_key: new_sumsq})

        # refresh total number
        self.getPointNum()
        # refresh STD
        self.setSTD()


class CS(Cluster):

    def __init__(self):
        Cluster.__init__(self)
        self.signature = "CS"
        self.r2c_index = 0
        self.merge_index = 0

    def removeCluster(self, key):
        # remove centroid info and SUM_N
        self.centroid_info.pop(key)

        # remove SUMSQ_N
        self.SUMSQ_N.pop(key)

        # remove cluster result
        self.cluster_result.pop(key)

        # remove STD info
        self.STD.pop(key)

        self.getPointNum()

    def delta_update(self, info: dict, statistics: dict, result: dict):
        """
        add more clusters
        :param info:
        :param statistics:
        :param result:
        :return:
        """
        if len(info.keys()) != 0:
            for r2c_key in list(info.keys()):
                self.centroid_info.update({"r2c" + str(self.r2c_index): info.get(r2c_key)})
                self.SUMSQ_N.update({"r2c" + str(self.r2c_index): statistics.get(r2c_key)})
                self.cluster_result.update({"r2c" + str(self.r2c_index): result.get(r2c_key)})
                self.setSTD()  # refresh STD
                self.r2c_index += 1
        #     print("info -> delta_update CS centroid info and cluster result")
        # else:
        #     print("info -> data's size is 0. don't need to delta_update CS object")

    def mergeCluster(self, cluster1: str, cluster2: str):
        new_location = computeAVGByColumn(list(self.centroid_info[cluster1]),
                                          list(self.centroid_info[cluster2]))
        new_sumsq = computeAVGByColumn(list(self.SUMSQ_N[cluster1]),
                                       list(self.SUMSQ_N[cluster2]))
        new_cluster_result = list(self.cluster_result[cluster1])
        new_cluster_result.extend(list(self.cluster_result[cluster2]))

        new_cluster_key = "m" + str(self.merge_index)
        # update centroid info and SUM_N
        self.centroid_info.pop(cluster1)
        self.centroid_info.pop(cluster2)
        self.centroid_info.update({new_cluster_key: new_location})

        # update SUMSQ_N
        self.SUMSQ_N.pop(cluster1)
        self.SUMSQ_N.pop(cluster2)
        self.SUMSQ_N.update({new_cluster_key: new_sumsq})

        # update cluster result
        self.cluster_result.pop(cluster1)
        self.cluster_result.pop(cluster2)
        self.cluster_result.update({new_cluster_key: new_cluster_result})

        # refresh STD
        self.setSTD()
        self.merge_index += 1

    def getClusterResultSortedInfo(self):
        result = collections.defaultdict(list)
        for key in self.cluster_result.keys():
            result[key] = sorted(self.cluster_result[key])

        return result


class RS:

    def __init__(self):
        self.remaining_set = dict()

    def add(self, data: dict):
        self.remaining_set.update(data)

    def count(self):
        return len(self.remaining_set.keys())

    def getRemainingData(self):
        return self.remaining_set

    @classmethod
    def getSignature(cls):
        return "RS"

    def gatherUp(self, left_over: dict):
        self.remaining_set = left_over


class IntermediateRecords:

    def __init__(self):
        self.intermediate_records = dict()
        self.intermediate_records["header"] = (
            "round_id", "nof_cluster_discard", "nof_point_discard",
            "nof_cluster_compression", "nof_point_compression",
            "nof_point_retained"
        )

    def save_check_point(self, round_id, ds, cs, rs):
        self.intermediate_records[round_id] = (
            round_id, ds.getClusterNum(), ds.getPointNum(),
            cs.getClusterNum(), cs.getPointNum(), rs.count()
        )
        print("{} -> DS_INFO: C:{} NUM:{} | CS_INFO: C:{} NUM:{} | RS_INFO: NUM:{}".format(
            round_id, ds.getClusterNum(), ds.getPointNum(), cs.getClusterNum(), cs.getPointNum(), rs.count()
        ))

    def export(self, output_file_path: str):
        export2File(self.intermediate_records, output_file_path, file_type="csv")


def export2File(result_dict, output_file_path, export_type="w+", file_type="json"):
    """
    export list content to a file
    :param export_type:
    :param file_type:
    :param result_dict: a list of dict
    :param output_file_path: output file path
    :return: nothing, but a file
    """
    if file_type == "json":
        with open(output_file_path, export_type) as output_file:
            output_file.writelines(json.dumps(result_dict))
            output_file.close()
    elif file_type == "csv":
        with open(output_file_path, export_type, newline="") as output_file:
            writer = csv.writer(output_file)
            for key, value in result_dict.items():
                writer.writerow(value)


def checkBelongings(cluster_centroid: dict, cluster_statistics: dict, cluster_result: dict):
    """

    :param cluster_statistics:
    :param cluster_centroid:
    :param cluster_result:
    :return:
    """
    remaining_data_points = dict()
    temp_cluster_result = copy.deepcopy(cluster_result)
    for centroid, belongings in temp_cluster_result.items():
        if len(belongings) <= 1:
            if len(belongings) != 0:
                remaining_data_points.update({belongings[0]: cluster_centroid.get(centroid)})
            cluster_result.pop(centroid)
            cluster_centroid.pop(centroid)
            cluster_statistics.pop(centroid)

    return cluster_centroid, cluster_statistics, cluster_result, remaining_data_points


def computeDistance(arrA, arrB, std=None, distance_type="euclidean"):
    """

    :param arrA:
    :param arrB:
    :param std:
    :param distance_type:
    :return:
    """
    if distance_type == "euclidean":
        return float(sqrt(sum([(a - b) ** 2 for (a, b) in zip(arrA, arrB)])))
    elif distance_type == "mahalanobis":
        return float(sqrt(sum([((a - b) / sd) ** 2 for (a, b, sd) in zip(arrA, arrB, std)])))


def assign2NS(data, alpha_value, DS_obj=None, CS_obj=None, cluster_type=""):
    """
    :param data: (21318, [a,b,c,d,e,f,g,h ...])
    :param alpha_value:
    :param DS_obj:
    :param CS_obj:
    :param cluster_type:
    :return:
    """
    if DS_obj is not None and cluster_type == DS_obj.getSignature():
        ds_dimensions = DS_obj.getDimension()
        distance_to_ds = float('inf')
        min_key_to_ds = None
        for key, location in DS_obj.getCentroidInfo().items():
            temp_to_ds_distance = computeDistance(data[1], location, DS_obj.getSTD().get(key),
                                                  distance_type="mahalanobis")
            if temp_to_ds_distance < alpha_value * sqrt(ds_dimensions) \
                    and temp_to_ds_distance < distance_to_ds:
                distance_to_ds = temp_to_ds_distance
                min_key_to_ds = (key, data[0])

        if min_key_to_ds is not None:
            yield tuple((min_key_to_ds, data[1], False))
        else:
            yield tuple((("-1", data[0]), data[1], True))

    elif CS_obj is not None and cluster_type == CS_obj.getSignature():
        cs_dimensions = CS_obj.getDimension()
        distance_to_cs = float('inf')
        min_key_to_cs = None
        for key, location in CS_obj.getCentroidInfo().items():
            temp_to_cs_distance = computeDistance(data[1], location, CS_obj.getSTD().get(key),
                                                  distance_type="mahalanobis")
            if temp_to_cs_distance < alpha_value * sqrt(cs_dimensions) \
                    and temp_to_cs_distance < distance_to_cs:
                distance_to_cs = temp_to_cs_distance
                min_key_to_cs = (key, data[0])

        if min_key_to_cs is not None:
            yield tuple((min_key_to_cs, data[1], False))
        else:
            yield tuple((("-1", data[0]), data[1], True))


def merge2CS(alpha_value, CS_obj: CS):
    cs_dimensions = CS_obj.getDimension()
    previous_cs_obj = copy.deepcopy(CS_obj)
    available_set = set(list(previous_cs_obj.getCentroidInfo().keys()))
    for key_pair in itertools.combinations(list(previous_cs_obj.getCentroidInfo().keys()), 2):
        if key_pair[0] in available_set and key_pair[1] in available_set:
            cs_2_cs_distance = computeDistance(arrA=previous_cs_obj.getCentroidLocationByKey(key_pair[0]),
                                               arrB=previous_cs_obj.getCentroidLocationByKey(key_pair[1]),
                                               std=previous_cs_obj.getSTDByKey(key_pair[0]),
                                               distance_type="mahalanobis")
            if cs_2_cs_distance < alpha_value * sqrt(cs_dimensions):
                # do merge
                CS_obj.mergeCluster(key_pair[0], key_pair[1])
                available_set.discard(key_pair[0])
                available_set.discard(key_pair[1])


def assignCS2DS(alpha_value, DS_obj: DS, CS_obj: CS):
    """

    :param alpha_value:
    :param DS_obj:
    :param CS_obj:
    :return:
    """
    ds_dimensions = DS_obj.getDimension()
    previous_ds_obj = copy.deepcopy(DS_obj)
    previous_cs_obj = copy.deepcopy(CS_obj)
    for cs_centroid in previous_cs_obj.getCentroidInfo().keys():

        for ds_centroid in previous_ds_obj.getCentroidInfo().keys():
            cs_2_ds_distance = computeDistance(
                arrA=previous_cs_obj.getCentroidLocationByKey(cs_centroid),
                arrB=previous_ds_obj.getCentroidLocationByKey(ds_centroid),
                std=previous_ds_obj.getSTDByKey(ds_centroid),
                distance_type="mahalanobis"
            )
            if cs_2_ds_distance < alpha_value * sqrt(ds_dimensions):
                # do merge
                DS_obj.mergeToOneCluster(ds_cluster_key=ds_centroid,
                                         cs_cluster_sumsq_info=previous_cs_obj.getSUMSQ_NByKey(cs_centroid),
                                         cs_cluster_centroid_info=previous_cs_obj.getCentroidLocationByKey(cs_centroid),
                                         cs_cluster_result=previous_cs_obj.getClusterResultByKey(cs_centroid))
                CS_obj.removeCluster(cs_centroid)
                break


def exportClusterResult(DS_obj: DS, CS_obj: CS, RS_obj: RS, output_file_path):
    result = collections.defaultdict()
    for ds_key in list(DS_obj.getClusterResult().keys()):
        [result.setdefault(str(data_id), int(ds_key[1:])) for data_id in DS_obj.getClusterResultByKey(ds_key)]

    for cs_key in list(CS_obj.getClusterResult().keys()):
        [result.setdefault(str(data_id), -1) for data_id in CS_obj.getClusterResultByKey(cs_key)]

    for rs_key in list(RS_obj.getRemainingData().keys()):
        result.setdefault(str(rs_key), -1)

    export2File(result, output_file_path, "w+", file_type="json")


if __name__ == '__main__':
    start = time.time()
    # define input variables
    input_csv_dir_path = "../data/test5"
    num_of_cluster = int("15")
    output_cluster_path = "../out/cluster5.json"
    output_intermediate_file_path = "../out/intermediate5.csv"

    # input_csv_dir_path = sys.argv[1]
    # num_of_cluster = int(sys.argv[2])
    # output_cluster_path = sys.argv[3]
    # output_intermediate_file_path = sys.argv[4]

    conf = SparkConf().setMaster("local[*]") \
        .setAppName("ay_hw_5_bfr") \
        .set("spark.executor.memory", "4g") \
        .set("spark.driver.memory", "4g")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")

    alpha = 3
    discard_set = DS()
    compression_set = CS()
    retained_set = RS()
    intermediate_records = IntermediateRecords()

    for index, file_path in enumerate(sorted(os.listdir(input_csv_dir_path))):
        data_file_path = ''.join(input_csv_dir_path + "/" + file_path)
        # a. Load the data points from one file
        row_data_rdd = sc.textFile(data_file_path).map(lambda row: row.split(",")) \
            .map(lambda kvs: (int(kvs[0]), list(map(eval, kvs[1:]))))

        if index == 0:
            total_length = row_data_rdd.count()
            first_N = 10000 if total_length > 10000 else int(total_length * 0.1)
            # run k-means by a sample dataset
            sample_data = row_data_rdd.filter(lambda kv: kv[0] < first_N).collectAsMap()
            
            # b. Run K-Means on a small random sample of the data points to initialize
            # the K centroids using the Euclidean distance as the similarity measurement
            ds_center, ds_stat, ds_cluster = KMeans(k=num_of_cluster, max_iterations=5).fit(sample_data)

            # c. Use the K-Means result from b to generate the DS clusters
            # (i.e., discard points and generate statistics)
            discard_set.init(ds_center, ds_stat, ds_cluster)

            # d. The initialization of DS has finished, so far, you have K clusters in DS.
            # e. Run K-Means on the rest of the data points with a large number of clusters
            # (e.g., 5 times of K) to generate CS (clusters with more than one points)
            rest_data = row_data_rdd.filter(lambda kv: kv[0] >= first_N).collectAsMap()
            center, stat, cluster = KMeans(k=num_of_cluster * 3, max_iterations=3).fit(rest_data)

            # and generate RS (clusters with only one point).
            cs_center, cs_stat, cs_cluster, remaining2 = checkBelongings(center, stat, cluster)
            compression_set.init(cs_center, cs_stat, cs_cluster)
            retained_set.add(remaining2)

        else:
            # f. Load the data points from next file.
            # => (('c1', 21318), [x,a,s,,f,f,...], False)
            step1_rdd = row_data_rdd \
                .flatMap(lambda data_point: assign2NS(data_point, alpha, DS_obj=discard_set,
                                                      cluster_type=discard_set.getSignature()))

            # g. For the new points, compare them to the clusters in DS using
            # the Mahalanobis Distance and assign them to the nearest DS cluster
            # if the distance is < 𝛼√𝑑.
            # => input ('c1', 21318), [x,a,s,,f,f,...], False)
            # => output (('c1', 21318), [x,a,s,,f,f,...])
            temp_ds_rdd = step1_rdd.filter(lambda pair_loc_flag: pair_loc_flag[2] is False) \
                .map(lambda pair_loc_flag: (pair_loc_flag[0], pair_loc_flag[1]))

            # g.i update ds cluster result based on these new data point
            # => (('c1', 21318), [x,a,s,,f,f,...])
            # => {'c1': [21318,234,2334,....]}
            temp_ds_cluster_result = temp_ds_rdd.map(lambda key_pair_loc: key_pair_loc[0]) \
                .groupByKey().mapValues(list).collectAsMap()

            # g.ii update ds centroid info based on these new data point
            # => (('c1', 21318), [x,a,s,,f,f,...])
            # => {21318: [x,a,s,,f,f,...], 123123: [x,a,s,,f,f,...]}
            temp_ds_data_dict = temp_ds_rdd \
                .map(lambda key_pair_loc: (key_pair_loc[0][1], list(key_pair_loc[1]))) \
                .collectAsMap()
            discard_set.updateCentroidInfo(temp_ds_cluster_result, temp_ds_data_dict)

            # h. For the new points that are not assigned to DS clusters, using
            # the Mahalanobis Distance and assign the points to the nearest CS
            # cluster if the distance is < 𝛼√𝑑.
            # => input ('-1', 21318), [x,a,s,,f,f,...], True)
            # => output (('c1', 23452), [x,a,s,,f,f,...], False)
            step2_rdd = step1_rdd.filter(lambda pair_loc_flag: pair_loc_flag[2] is True) \
                .map(lambda pair_loc_flag: (pair_loc_flag[0][1], pair_loc_flag[1])) \
                .flatMap(lambda data_point: assign2NS(data_point, alpha, CS_obj=compression_set,
                                                      cluster_type=compression_set.getSignature()))

            # => (('c1', 23452), [x,a,s,,f,f,...], False)
            # => (('c1', 23452), [x,a,s,,f,f,...])
            temp_cs_rdd = step2_rdd.filter(lambda pair_loc_flag: pair_loc_flag[2] is False) \
                .map(lambda pair_loc_flag: (pair_loc_flag[0], pair_loc_flag[1]))

            # h.i update ds cluster result based on these new data point
            # => (('c1', 21318), [x,a,s,,f,f,...])
            # => {'c1': [21318,234,2334,....]}
            temp_cs_cluster_result = temp_cs_rdd.map(lambda key_pair_loc: key_pair_loc[0]) \
                .groupByKey().mapValues(list).collectAsMap()
            # h.ii update ds centroid info based on these new data point
            # => (('c1', 21318), [x,a,s,,f,f,...])
            # => {21318: [x,a,s,,f,f,...], 123123: [x,a,s,,f,f,...]}
            temp_cs_data_dict = temp_cs_rdd \
                .map(lambda key_pair_loc: (key_pair_loc[0][1], list(key_pair_loc[1]))) \
                .collectAsMap()
            compression_set.updateCentroidInfo(temp_cs_cluster_result, temp_cs_data_dict)

            # i. For the new points that are not assigned to any clusters in DS or CS,
            # assign them to RS.
            # => (('-1', 23452), [x,a,s,,f,f,...], True)
            # => {23452: [x,a,s,,f,f,...]}
            remaining_data_dict = step2_rdd.filter(lambda pair_loc_flag: pair_loc_flag[2] is True) \
                .map(lambda pair_loc_flag: (pair_loc_flag[0][1], pair_loc_flag[1])) \
                .collectAsMap()
            retained_set.add(remaining_data_dict)

            # j. Merge the data points in RS by running K-Means with a large number
            # of clusters (e.g., 5 times of K) to generate CS (clusters with more
            # than one points) and RS (clusters with only one point)
            center, stat, cluster = KMeans(k=num_of_cluster * 3, max_iterations=5) \
                .fit(retained_set.getRemainingData())
            cs_center, cs_stat, cs_cluster, remaining3 = checkBelongings(center, stat, cluster)
            compression_set.delta_update(cs_center, cs_stat, cs_cluster)
            retained_set.gatherUp(remaining3)

            # k. Merge clusters in CS that have a Mahalanobis Distance < 𝛼√𝑑
            merge2CS(alpha, CS_obj=compression_set)

        # m. If this is the last round (after processing the last chunk of data points),
        # merge clusters in CS with the clusters in DS that have a Mahalanobis Distance < 𝛼√𝑑.
        if index + 1 == len(os.listdir(input_csv_dir_path)):
            assignCS2DS(alpha, DS_obj=discard_set, CS_obj=compression_set)

        intermediate_records.save_check_point(index + 1, discard_set, compression_set, retained_set)

    # export your findings
    intermediate_records.export(output_intermediate_file_path)
    exportClusterResult(DS_obj=discard_set,
                        CS_obj=compression_set,
                        RS_obj=retained_set,
                        output_file_path=output_cluster_path)
    print("Duration: %d s." % (time.time() - start))
