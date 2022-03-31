__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '1/28/2020 9:11 PM'

import sys
import json
from pyspark import SparkContext

BUSINESS_ID = 'business_id'
CATEGORY = 'categories'
START_SCORE = 'stars'
COUNT = 'count'


# explore the two datasets together (i.e., review and business) and write a program to
# compute the average stars for each business category and output top n categories
# with the highest average stars (1pts)
def get_avg_star(business_start_rdd, business_category_rdd, top_n):
    score_rdd = business_start_rdd.groupByKey() \
        .mapValues(lambda values: [float(value) for value in values]) \
        .map(lambda kvv: (kvv[0], (sum(kvv[1]), len(kvv[1]))))

    category_rdd = business_category_rdd \
        .filter(lambda kv: (kv[1] is not None) and (kv[1] is not "")) \
        .mapValues(lambda values: [value.strip() for value in values.split(',')])

    joined_rdd = category_rdd.leftOuterJoin(score_rdd)

    sorted_rdd = joined_rdd \
        .map(lambda kvv: kvv[1]) \
        .filter(lambda kv: kv[1] is not None) \
        .flatMap(lambda kv: [(category, kv[1]) for category in kv[0]]) \
        .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])) \
        .mapValues(lambda value: float(value[0] / value[1])) \
        .takeOrdered(top_n, key=lambda kv: (-kv[1], kv[0]))

    return sorted_rdd


# need to implement a version without Spark and compare to a version with Spark
# for reducing the time duration of execution (1pts)
def load_json_file_2_dict_list(file_path, key_list):
    lines = open(file_path, encoding="utf8").readlines()
    # create a dict item {'A': json['A']}
    return list(map(lambda line: {key_list[0]: json.loads(line)[key_list[0]],
                                  key_list[1]: json.loads(line)[key_list[1]]}, lines))


def build_id_set(dict_list):
    id_set = set()
    [id_set.add(dict_item[BUSINESS_ID]) for dict_item in dict_list]
    return id_set


def group_by_business_id(dict_list):
    grouped_dict = dict()
    for item in dict_list:
        if item[BUSINESS_ID] not in grouped_dict.keys():
            grouped_dict[item[BUSINESS_ID]] = (float(item[START_SCORE]), 1)
        else:
            new_value = grouped_dict.get(item[BUSINESS_ID])[0] + item[START_SCORE]
            new_count = grouped_dict.get(item[BUSINESS_ID])[1] + 1
            grouped_dict.update({item[BUSINESS_ID]: (new_value, new_count)})

    return grouped_dict


def gen_category_list(dict_list):
    grouped_dict = dict()
    for item in dict_list:
        if (item[CATEGORY] is not None) and (item[CATEGORY] is not ""):
            grouped_dict[item[BUSINESS_ID]] = [category.strip()
                                               for category in item[CATEGORY].split(',')]

    return grouped_dict


def left_join(id_score_dict, id_category_dict):
    joined_dict = dict()
    for id, score_count in id_score_dict.items():
        # there is a wired situation where the business's category is null, but still have a review post
        # so this if condition help to prevent null type error
        if None is not id_category_dict.get(id):
            for category in id_category_dict.get(id):
                # before we add this category into dict, we need to check if it exists
                if category not in joined_dict.keys():
                    joined_dict[category] = score_count
                else:
                    new_value = joined_dict.get(category)[0] + score_count[0]
                    new_count = joined_dict.get(category)[1] + score_count[1]
                    joined_dict.update({category: (new_value, new_count)})

    return {k: float(v[0] / v[1]) for k, v in joined_dict.items()}


if __name__ == '__main__':
    # review_json_path = "../../asnlib/publicdata/review.json"
    # business_json_path = "../../asnlib/publicdata/business.json"
    # output_file_path = "../out/output2.json"
    # if_spark = "no_spark"  # either "spark" or "no_spark"
    # top_n = '10'
    review_json_path = sys.argv[1]
    business_json_path = sys.argv[2]
    output_file_path = sys.argv[3]
    if_spark = sys.argv[4]
    top_n = sys.argv[5]

    result_dict = dict()

    if if_spark == "spark":
        sc = SparkContext.getOrCreate()

        review_lines = sc.textFile(review_json_path).map(lambda row: json.loads(row))
        business_lines = sc.textFile(business_json_path).map(lambda row: json.loads(row))

        business_start_rdd = review_lines.map(lambda kv: (kv[BUSINESS_ID], kv[START_SCORE]))
        business_category_rdd = business_lines.map(lambda kv: (kv[BUSINESS_ID], kv[CATEGORY]))

        result_dict['result'] = get_avg_star(business_start_rdd, business_category_rdd, int(top_n))
    else:
        # read data in a crazy way
        business_start_list = load_json_file_2_dict_list(review_json_path,
                                                         [BUSINESS_ID, START_SCORE])

        business_category_list = load_json_file_2_dict_list(business_json_path,
                                                            [BUSINESS_ID, CATEGORY])

        id_score_dict = group_by_business_id(business_start_list)
        id_category_dict = gen_category_list(business_category_list)
        print(id_category_dict)
        joined_dict = left_join(id_score_dict, id_category_dict)

        result_dict['result'] = sorted(joined_dict.items(), key=lambda kv: (-kv[1], kv[0]))[:int(top_n)]

    # print(result_dict)

    with open(output_file_path, 'w+') as output_file:
        json.dump(result_dict, output_file)
    output_file.close()
