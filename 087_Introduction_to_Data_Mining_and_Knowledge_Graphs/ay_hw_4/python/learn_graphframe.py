__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '3/24/2020 10:59 AM'

import sys
import os
import time

from graphframes import *
from pyspark import SparkContext, SparkConf
from pyspark.python.pyspark.shell import sqlContext

os.environ["PYSPARK_SUBMIT_ARGS"] = (
    "--packages graphframes:graphframes:0.7.0-spark2.4-s_2.11")

# os.environ["PYSPARK_SUBMIT_ARGS"] = (
#     "--packages graphframes:graphframes:0.6.0-spark2.3-s_2.11")

if __name__ == '__main__':
    start = time.time()

    # conf = SparkConf().setMaster("local") \
    #     .setAppName("ay_hw_4_learn") \
    #     .set("spark.executor.memory", "4g") \
    #     .set("spark.driver.memory", "4g")
    # sc = SparkContext(conf=conf)

    vertices = sqlContext.createDataFrame([
        ("a", "Alice", 34),
        ("b", "Bob", 36),
        ("c", "Charlie", 30),
        ("d", "David", 29),
        ("e", "Esther", 32),
        ("f", "Fanny", 36),
        ("g", "Gabby", 60)], ["id", "name", "age"])

    edges = sqlContext.createDataFrame([
        ("a", "b", "friend"),
        ("b", "c", "follow"),
        ("c", "b", "follow"),
        ("f", "c", "follow"),
        ("e", "f", "follow"),
        ("e", "d", "friend"),
        ("d", "a", "friend"),
        ("a", "e", "friend")
    ], ["src", "dst", "relationship"])

    g = GraphFrame(vertices, edges)
    print("=>",g)

    g.vertices.show()
    g.edges.show()
    g.vertices.groupBy().min("age").show()

    result = g.labelPropagation(maxIter=5)
    result.select("id", "label").show()
