import collections
import math
import random


def findMedian(sorted_list_obj):
    list_size = len(sorted_list_obj)
    if list_size % 2 == 0:
        median = (sorted_list_obj[list_size // 2] + sorted_list_obj[list_size // 2 - 1]) / 2
        sorted_list_obj[0] = median
    if list_size % 2 == 1:
        median = sorted_list_obj[(list_size - 1) // 2]
        sorted_list_obj[0] = median
    return sorted_list_obj[0]


def split2NChunks(list_data, n):
    size = int(math.ceil(len(list_data) / float(n)))
    return [list_data[i:i + size] for i in range(0, len(list_data), size)]


if __name__ == '__main__':
    a = [[1, 3, 2],
         [4, 0, 1]]

    b = [[1, 3],
         [0, 1],
         [5, 2]]

    for l in range(len(a)):
        for val in a[l]:
            for k in range(2):
                print("({},{})".format((l + 1, k + 1), val))

