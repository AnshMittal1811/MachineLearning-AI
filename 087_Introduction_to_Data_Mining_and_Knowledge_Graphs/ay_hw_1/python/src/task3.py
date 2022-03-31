__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '1/29/2020 10:23 PM'

import json
import sys
from operator import add

from pyspark import SparkContext


def review_partitioner(key):
    return ord(key[:1])


if __name__ == '__main__':

    # review_json_path = "../../asnlib/publicdata/review.json"
    # output_file_path = "../out/output3.json"
    # partition_type = "customized"  # either "default" or "customized"
    # num_partition = '21'
    # num_n = '10'
    review_json_path = sys.argv[1]
    output_file_path = sys.argv[2]
    partition_type = sys.argv[3]
    num_partition = sys.argv[4]
    num_n = sys.argv[5]

    result_dict = dict()

    sc = SparkContext.getOrCreate()

    review_lines = sc.textFile(review_json_path).map(lambda row: json.loads(row))
    business_ids_rdd = review_lines.map(lambda kv: (kv['business_id'], 1))

    if partition_type != "default":
        business_ids_rdd = business_ids_rdd.partitionBy(int(num_partition), review_partitioner)

    result_dict['n_partitions'] = business_ids_rdd.getNumPartitions()
    result_dict['n_items'] = business_ids_rdd.glom().map(len).collect()
    result_dict['result'] = business_ids_rdd.reduceByKey(add) \
        .filter(lambda kv: kv[1] > int(num_n)).collect()

    with open(output_file_path, 'w+') as output_file:
        json.dump(result_dict, output_file)
    output_file.close()
