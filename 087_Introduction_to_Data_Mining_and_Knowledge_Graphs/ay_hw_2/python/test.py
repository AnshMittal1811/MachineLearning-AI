import sys
from pyspark import SparkContext, SparkConf, StorageLevel
import itertools
from operator import add
import time


def Union(lst1, lst2):
    # combination of two list
    return sorted(list(set(lst1) | set(lst2)))


def intersection(lst1, lst2):
    return sorted(list(set(lst1) & set(lst2)))


def Size3Candidate(preFreq, k):  # preFreq> list of lists
    candidates = []
    # do merge 2 k-1 frequent which produce a size k candidate > not following apriori rule
    for index, f1 in enumerate(preFreq[:-1]):
        for f2 in preFreq[index + 1:]:
            if f1[:-1] == f2[:-1]:
                comb = tuple(Union(f1, f2))
                subs = [sub for sub in itertools.combinations(comb, k - 1)]
                if set(subs).issubset(set(preFreq)):
                    candidates.append(comb)
            if f1[:-1] != f2[:-1]:
                break
    return candidates


def freqinSubset(iterator, distinctItems, support, numofwhole):
    baskets = list(iterator)
    numofsub = len(baskets)
    sub_s = int(support * numofsub / numofwhole)

    freqSub = {}
    candies = [tuple([x]) for x in distinctItems]
    # single candidate > actually not used!!!! larger than the length of one single basket making loop longer
    k = 0

    while candies:
        k += 1
        tempcount = {}
        for basket in baskets:

            if k == 1:  # count singles
                for single in itertools.combinations(basket, 1):
                    try:
                        tempcount[single] += 1
                    except:
                        tempcount[single] = 1
            if k == 2:  # count pairs
                thinbasket = intersection(basket, flatSingle)  # only frequent single can produce frequent pair
                for pair in itertools.combinations(thinbasket, 2):
                    try:
                        tempcount[pair] += 1
                    except:
                        tempcount[pair] = 1

            if k > 2:  # count triples..... where candidates number is much less than if we iterate through subset of basket
                if len(basket) >= k:
                    for candy in candies:
                        if set(candy).issubset(set(basket)):
                            try:
                                tempcount[candy] += 1
                            except:
                                tempcount[candy] = 1

        freqSub[k] = sorted([items for items, count in tempcount.items() if count >= sub_s])

        if k == 1:
            flatSingle = [single[0] for single in freqSub[k]]
            candies = [pair for pair in itertools.combinations(flatSingle, 2)]
        if k > 1: candies = Size3Candidate(freqSub[k], k + 1)

    allsubFreq = []
    for _, candy in freqSub.items():
        allsubFreq.extend(candy)

    yield allsubFreq


def countOnWhole(basket, candidates):
    result = []
    for candy in candidates:
        if set(candy).issubset(set(basket)):
            result.extend([(tuple(candy), 1)])
    yield result


def groupByLength(ItemsSets):
    dic = {}
    for key, group in itertools.groupby(ItemsSets, lambda items: len(items)):
        dic[key] = sorted(list(group), key=lambda x: x)
    return dic


def strItemSets(ItemSets):
    ItemSetsD = groupByLength(ItemSets)
    output = ''
    for _, items in ItemSetsD.items():
        output += str([list(x) for x in items])[1:-1].replace('[', '(').replace(' (', '(').replace(']', ')') + '\n\n'
    return output[:-2]


# main function#########

start = time.time()

# arguments
threshold = 70
support = 50
inputFilePath = "../out/user_business.csv"
outputFilePath = "../out/task_10_ans.txt"

conf = SparkConf()
conf.setMaster("local")
conf.setAppName("hw2")
conf.set("spark.executor.memory", "4g")
conf.set("spark.driver.memory", "4g")
sc = SparkContext.getOrCreate(conf)

inputFile = sc.textFile(inputFilePath)
header = inputFile.first()

inputRDD = inputFile.filter(lambda line: line != header).map(lambda line: line.split(',')).map(lambda t: tuple(t)) \
    .groupByKey().mapValues(set).mapValues(sorted).mapValues(tuple)

inputRDDfilter = inputRDD.filter(lambda t: len(t[1]) > threshold).map(lambda t: (1, t[1])).persist(
    StorageLevel.DISK_ONLY)

# count distinct items
distinctItems = inputRDDfilter.flatMapValues(tuple) \
    .map(lambda t: (t[1], t[0])).groupByKey() \
    .map(lambda t: (t[0])).collect()
distinctItems.sort()

# whole baskets number
whole_number = inputRDDfilter.count()

baskets = inputRDDfilter.map(lambda t: t[1]).persist(StorageLevel.DISK_ONLY)

# first phase > find candidate in subsets
FreqinSample = baskets.mapPartitions(lambda part: freqinSubset(part, distinctItems, support, whole_number)) \
    .flatMap(lambda x: x).distinct().sortBy(lambda t: (len(t), t)).collect()

# 2nd phase > count on whole
frequentI = baskets.flatMap(lambda basket: countOnWhole(basket, FreqinSample)).flatMap(lambda x: x).reduceByKey(add) \
    .filter(lambda items: items[1] >= support) \
    .map(lambda items: items[0]).sortBy(lambda t: (len(t), t)).collect()

output = 'Candidates:\n' + strItemSets(FreqinSample) + '\n\n' + 'Frequent Itemsets:\n' + strItemSets(frequentI)

with open(outputFilePath, 'w') as fd:
    fd.write(output)
    fd.close()

print("Duration: %d seconds" % (time.time() - start))