__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '1/27/2020 1:19 PM'

import sys
from operator import add

from pyspark import SparkContext
import json
from datetime import datetime


# A. The total number of reviews (0.5pts)
def get_total_num_review(review_ids_rdd):
    return review_ids_rdd.count()


# B. The number of reviews in a given year, y (0.5pts)
def get_num_review_in_given_year(review_date_rdd, year):
    return review_date_rdd.filter(lambda kv: datetime.strptime(
        kv[1], '%Y-%m-%d %H:%M:%S').year == year).count()


# C. The number of distinct users who have written the reviews (0.5pts)
def get_distinct_user(user_ids_rdd):
    return user_ids_rdd.distinct().count()


# D. Top m users who have the largest number of reviews and its count (0.5pts)
def get_most_active_user_info(business_user_rdd, top_m):
    return business_user_rdd.map(lambda kv: (kv[1], 1)).reduceByKey(add). \
        takeOrdered(top_m, key=lambda kv: (-kv[1], kv[0]))


# E. Top n frequent words in the review text. The words should be in lower cases.
# The following punctuations i.e., “(”, “[”, “,”, “.”, “!”, “?”, “:”, “;”, “]”, “)”,
# and the given stopwords are excluded (1pts)
def trim_and_ignore(word):
    if word not in stop_words_set:
        return ''.join(ch for ch in word if ch not in exclude_char_set)


def get_most_frequent_words(review_text_rdd, top_n):
    # TODO If two users/words have the same count, please sort them in the alphabetical order
    words_dict = review_text_rdd.map(lambda kv: kv[1]).flatMap(lambda text: text.lower().split(' ')) \
        .map(lambda word: (trim_and_ignore(word), 1)).filter(lambda kv: (kv[0] is not None) and (kv[0] is not "")) \
        .reduceByKey(add).takeOrdered(top_n, key=lambda kv: (-kv[1], kv[0]))

    return list(map(lambda kv: kv[0], words_dict))


if __name__ == '__main__':
    # input_json_path = "../../asnlib/publicdata/review_sample.json"
    # output_file_path = "../out/output.json"
    # stop_words_path = "../../asnlib/publicdata/stopwords"
    # year = '2018'
    # top_m = '4'
    # top_n = '100'
    input_json_path = sys.argv[1]
    output_file_path = sys.argv[2]
    stop_words_path = sys.argv[3]
    year = sys.argv[4]
    top_m = sys.argv[5]
    top_n = sys.argv[6]
    exclude_char_set = set(r"""()[],.!?:;""")
    stop_words_set = set(word.strip() for word in open(stop_words_path))
    sc = SparkContext.getOrCreate()

    result_dict = dict()
    input_lines = sc.textFile(input_json_path).map(lambda row: json.loads(row))
    review_ids_rdd = input_lines.map(lambda kv: kv['review_id'])
    result_dict['A'] = get_total_num_review(review_ids_rdd)

    review_date_rdd = input_lines.map(lambda kv: (kv['review_id'], kv['date']))
    result_dict['B'] = get_num_review_in_given_year(review_date_rdd, int(year))

    user_ids_rdd = input_lines.map(lambda kv: kv['user_id'])
    result_dict['C'] = get_distinct_user(user_ids_rdd)

    business_user_rdd = input_lines.map(lambda kv: (kv['business_id'], kv['user_id']))
    result_dict['D'] = get_most_active_user_info(business_user_rdd, int(top_m))
    # TODO 那你说 一个人在相同的business_id下评论多条  这是算一条  还是多条
    review_text_rdd = input_lines.map(lambda kv: (kv['review_id'], kv['text']))
    result_dict['E'] = get_most_frequent_words(review_text_rdd, int(top_n))

    print(result_dict)

    with open(output_file_path, 'w+') as output_file:
        json.dump(result_dict, output_file)
    output_file.close()
