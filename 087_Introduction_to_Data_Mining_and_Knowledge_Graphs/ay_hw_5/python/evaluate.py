from sklearn.metrics.cluster import normalized_mutual_info_score
from pyspark import SparkContext
import json

sc = SparkContext(appName="INF553HW5", master="local[*]")
sc.setLogLevel("WARN")
sc.setLogLevel("ERROR")

clustering_file_path = "D:\\Projects\\INF_553\\ay_hw_5\\out\\cluster5.json"
label_path = "D:\\Projects\\INF_553\\ay_hw_5\\data\\cluster5.json"

ground_truth = sc.textFile(label_path).map(lambda line: json.loads(line)). \
    flatMap(lambda line: [(index, label) for index, label in line.items()]). \
    map(lambda pair: (int(pair[0]), pair[1])). \
    collect()
ground_truth.sort()
ground_truth = [cid for _, cid in ground_truth]

ground_truth_cluster_size = sc.textFile(label_path).map(lambda line: json.loads(line)). \
    flatMap(lambda line: [(index, label) for index, label in line.items()]). \
    map(lambda pair: (pair[1], pair[0])). \
    groupByKey(). \
    mapValues(len).collect()
ground_truth_cluster_size.sort(key=lambda pair: pair[1])

prediction_cluster_size = sc.textFile(clustering_file_path).map(lambda line: json.loads(line)). \
    flatMap(lambda line: [(index, label) for index, label in line.items()]). \
    map(lambda pair: (pair[1], pair[0])). \
    groupByKey(). \
    mapValues(len).collect()
prediction_cluster_size.sort(key=lambda pair: pair[1])
prediction = sc.textFile(clustering_file_path).map(lambda line: json.loads(line)). \
    flatMap(lambda line: [(index, label) for index, label in line.items()]). \
    map(lambda pair: (int(pair[0]), pair[1])). \
    collect()
prediction.sort()
prediction = [cid for _, cid in prediction]

print(normalized_mutual_info_score(ground_truth, prediction))
print(ground_truth_cluster_size)
print(prediction_cluster_size)