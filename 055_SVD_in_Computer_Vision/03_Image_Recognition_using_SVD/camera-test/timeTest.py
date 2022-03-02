import time
import numpy as np
import random as rand

startTime = time.time()
listOfRandomNumbers = []
for i in range(10000000):
    listOfRandomNumbers.append(rand.randint)

print(time.time()- startTime, "execution time")

startTime2 = time.time()
listOfNPNumber = np.array(0)
for i in range(10000000):
    np.append(listOfNPNumber, np.random.randint(0,10))
print(time.time()- startTime2, "execution time2")