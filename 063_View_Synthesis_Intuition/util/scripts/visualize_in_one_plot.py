import csv
import matplotlib.pyplot as plt
import sys

if __name__ == "__main__":

    if len(sys.argv) != 3:
        raise ValueError("Usage: " + sys.argv[0] + " <path_to_train.csv> <path_to_val.csv>")

    train = csv.reader(open(sys.argv[1]))
    next(train, None) # skip header
    train = [(line[1], float(line[2])) for line in train]

    val = csv.reader(open(sys.argv[2]))
    next(val, None)  # skip header
    val = [(line[1], float(line[2])) for line in val]

    print([t[1] for t in train])

    plt.plot([t[0] for t in train], [t[1] for t in train], "b-") # train is blue
    plt.plot([t[0] for t in val], [t[1] for t in val], "r-") # val is red
    plt.show()