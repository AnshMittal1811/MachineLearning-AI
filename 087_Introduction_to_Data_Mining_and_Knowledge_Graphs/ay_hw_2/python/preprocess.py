__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '2/10/2020 10:05 PM'

import csv
import json

from pyspark import SparkContext


def export2csv(output_file_path, data):
    with open(output_file_path, 'w', newline='') as output_file:
        writer = csv.writer(output_file, quoting=csv.QUOTE_NONE)
        writer.writerow(["user_id", "business_id"])
        for row in data:
            writer.writerow(row)


if __name__ == '__main__':
    input_review_json_path = "../data/review.json"
    input_business_json_path = "../data/business.json"
    output_csv_path = "../out/user_business.csv"
    state = "NV"  # Nevada

    sc = SparkContext.getOrCreate()

    input_business_lines = sc.textFile(input_business_json_path) \
        .map(lambda lines: json.loads(lines))

    business_ids = input_business_lines \
        .map(lambda kv: (kv['business_id'], kv['state'])) \
        .filter(lambda kv: kv[1] == state).map(lambda kv: kv[0]).collect()

    input_review_lines = sc.textFile(input_review_json_path) \
        .map(lambda lines: json.loads(lines))

    rew_ids_bus_ids = input_review_lines \
        .map(lambda kv: (kv['user_id'], kv['business_id'])) \
        .filter(lambda kv: kv[1] in business_ids).collect()

    export2csv(output_csv_path, rew_ids_bus_ids)
