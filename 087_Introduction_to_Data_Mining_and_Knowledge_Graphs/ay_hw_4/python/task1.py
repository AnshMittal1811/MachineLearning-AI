__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '3/24/2020 10:59 AM'

import itertools
import os
import sys
import time

from graphframes import GraphFrame
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

# os.environ["PYSPARK_SUBMIT_ARGS"] = (
#     "--packages graphframes:graphframes:0.7.0-spark2.4-s_2.11")

os.environ["PYSPARK_SUBMIT_ARGS"] = (
    "--packages graphframes:graphframes:0.6.0-spark2.3-s_2.11")


def export2File(result_array, file_path):
    """
    export list content to a file
    :param result_array: a list of dict
    :param file_path: output file path
    :return: nothing, but a file
    """
    with open(file_path, 'w+') as output_file:
        for id_array in result_array:
            output_file.writelines(str(id_array)[1:-1] + "\n")
        output_file.close()


if __name__ == '__main__':
    start = time.time()
    # define input variables
    # filter_threshold = "7"
    # input_csv_path = "../data/ub_sample_data.csv"
    # output_file_path = "../out/task1.txt"

    filter_threshold = sys.argv[1]
    input_csv_path = sys.argv[2]
    output_file_path = sys.argv[3]

    conf = SparkConf().setMaster("local") \
        .setAppName("ay_hw_4_task1") \
        .set("spark.executor.memory", "4g") \
        .set("spark.driver.memory", "4g")
    sc = SparkContext(conf=conf)
    sparkSession = SparkSession(sc)
    sc.setLogLevel("WARN")

    # read the original json file and remove the header
    raw_data_rdd = sc.textFile(input_csv_path)
    header = raw_data_rdd.first()
    uid_bidxes_dict = raw_data_rdd.filter(lambda line: line != header) \
        .map(lambda line: (line.split(',')[0], line.split(',')[1])) \
        .groupByKey().mapValues(lambda bids: sorted(list(bids))) \
        .collectAsMap()

    uid_pairs = list(itertools.combinations(list(uid_bidxes_dict.keys()), 2))

    edge_list = list()
    vertex_set = set()
    for pair in uid_pairs:
        if len(set(uid_bidxes_dict[pair[0]]).intersection(
                set(uid_bidxes_dict[pair[1]]))) >= int(filter_threshold):
            edge_list.append(tuple(pair))
            edge_list.append(tuple((pair[1], pair[0])))
            vertex_set.add(pair[0])
            vertex_set.add(pair[1])

    # vertex_df = vertex_rdd.toDF(["id"]).write.csv('vertex.csv')
    # edge_df = edge_rdd.toDF(["src", "dst"]).write.csv('edge.csv')
    vertex_df = sc.parallelize(list(vertex_set)).map(lambda uid: (uid,)).toDF(['id'])
    edge_df = sc.parallelize(edge_list).toDF(["src", "dst"])

    graph_frame = GraphFrame(vertex_df, edge_df)

    communities = graph_frame.labelPropagation(maxIter=5)

    communities_rdd = communities.rdd.coalesce(1) \
        .map(lambda idx_label: (idx_label[1], idx_label[0])) \
        .groupByKey().map(lambda label_idxes: sorted(list(label_idxes[1]))) \
        .sortBy(lambda idxes: (len(idxes), idxes))

    # export your finding
    export2File(communities_rdd.collect(), output_file_path)

    print("Duration: %d s." % (time.time() - start))
