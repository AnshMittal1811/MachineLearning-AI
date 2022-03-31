__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '3/24/2020 10:59 AM'

import sys
import os
import time
from collections import UserDict

from graphframes import *
from pyspark import SparkContext, SparkConf
from pyspark.python.pyspark.shell import sqlContext
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

os.environ["PYSPARK_SUBMIT_ARGS"] = (
    "--packages graphframes:graphframes:0.7.0-spark2.4-s_2.11")


# os.environ["PYSPARK_SUBMIT_ARGS"] = (
#     "--packages graphframes:graphframes:0.6.0-spark2.3-s_2.11")

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


class Node(UserDict):
    def __setitem__(self, key, value):
        super().__setitem__(key, value)


if __name__ == '__main__':
    start = time.time()

    # conf = SparkConf().setMaster("local") \
    #     .setAppName("ay_hw_4_learn") \
    #     .set("spark.executor.memory", "4g") \
    #     .set("spark.driver.memory", "4g")
    # sc = SparkContext(conf=conf)

    vertices_schema = StructType([StructField('id', StringType())])
    vertices = sqlContext.read.csv("../data/5.csv", schema=vertices_schema)

    edges_schema = StructType([StructField('src', StringType()),
                               StructField('dst', StringType())])
    edges = sqlContext.read.csv("../data/6.csv", schema=edges_schema)

    g = GraphFrame(vertices, edges)

    # => [Row(id=451, user_idx_str='79yaBDbLASfIdB-C2c8DzA', label=468),
    communities = g.labelPropagation(maxIter=5).coalesce(5)

    communities_rdd = communities.rdd.coalesce(1) \
        .map(lambda idx_label: (idx_label[1], idx_label[0])) \
        .groupByKey().map(lambda label_idxes: sorted(list(label_idxes[1]))) \
        .sortBy(lambda idxes: (len(idxes), idxes))

    # export your finding
    export2File(communities_rdd.collect(), "ww4.txt")

    print("Duration: %d s." % (time.time() - start))
