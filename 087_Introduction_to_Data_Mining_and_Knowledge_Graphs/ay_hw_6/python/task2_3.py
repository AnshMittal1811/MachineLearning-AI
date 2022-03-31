from pyspark import SparkContext, SparkConf, StorageLevel
from pyspark.streaming import StreamingContext
import sys
import json
import time
import random
import binascii
import datetime


# port = 9999
# output_file = "../out/task2_4.csv"

port = int(sys.argv[1])
output_file = sys.argv[2]

conf = (
    SparkConf()
    .setAppName("inf553_hw6_task2")
    .set("spark.executor.memory", "4g")
    .set("spark.driver.memory", "4g")
)
sc = SparkContext(conf=conf)
sc.setLogLevel("OFF")


def Fajoet_Martion(data):
    global times
    stream = data.collect()
    truth = len(set(stream))
    if times == 0:
        output = open(output_file, 'w')
        output.write("Time,Ground Truth,Estimation\n")
        times = 1
    else:
        output = open(output_file, 'a')

    ts = time.time()
    start = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    random.seed(9)
    a_l = [random.randint(0, (2 ** 16) - 1) for i in range(hash_number)]
    b_l = [random.randint(0, (2 ** 16) - 1) for i in range(hash_number)]
    R_2 = []
    for i in range(hash_number):
        max_zero = -1
        for data in stream:
            code_data = int(binascii.hexlify(data.encode("utf8")), 16)
            hash_value = (a_l[i] * code_data + b_l[i]) % n
            binary_val = format(hash_value, '032b')

            if hash_value == 0:
                num_zero = 0
            else:
                num_zero = len(str(binary_val)) - len(str(binary_val).rstrip("0"))
            if num_zero > max_zero:
                max_zero = num_zero
        R_2.append(2**max_zero)

    avg1 = sum(R_2[:4])/len(R_2[:4])
    avg2 = sum(R_2[4:8])/len(R_2[4:8])
    avg3 = sum(R_2[8:12])/len(R_2[8:12])
    ls = [avg1,avg2,avg3]
    ls.sort()
    estimate = int(ls[1])
    output.write(str(start)+","+str(truth)+","+str(estimate)+"\n")
    output.close()

ssc = StreamingContext(sc, 5)
lines = ssc.socketTextStream("localhost", port)
times = 0
hash_number = 12
n = 2**10-1
rdd = (
    lines.window(30, 10)
    .map(json.loads)
    .map(lambda x: x["city"])
    .foreachRDD(Fajoet_Martion)
)

ssc.start()
ssc.awaitTermination()


