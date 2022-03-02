import numpy as np
import time
startTime = time.time()

arr1 = np.arange(20).reshape(4,5)
for row in arr1:
    for value in np.nditer(row, op_flags=['readwrite']):
        if value % 2 == 0:
            value[...] = value * 2
print("arr1: ", arr1)

arr2 = np.array(arr1, dtype=float)
for value in np.nditer(arr2, op_flags=['readwrite']):
    value[...] = np.exp(value)
print("arr2: ", arr2)

maxRows = 5
maxColumns = 5
pictureDataMatrix = np.arange(maxRows * maxColumns*2 ).reshape(2,5,5)
testArr = np.arange(25).reshape(5,5)
for value in np.nditer(testArr, op_flags=['readwrite']):
    value[...] = value * 50
pictureDataMatrix[0] = testArr
print("testArr: ", testArr)
print("flattened: ", testArr.flatten())
print("big matrix: ", pictureDataMatrix)
print("flattened: ", pictureDataMatrix.flatten())

testArr2 = np.arange(30, dtype=float).reshape(3,10)
print("testArr2: ", testArr2)
print("flattened: ", testArr2.flatten())
print("test ARR 2: ", testArr2[1:2,2:4][0]) #prints 2 line and 3rd and 4th colums. The 0 at the end takes off the double brackets

sums = np.sum(testArr2, axis=1)
print("sums: ",sums)
average = np.divide(sums, len(testArr2[0]))
print("average: ", average)

subtract = np.repeat(average, 10, axis=0).reshape(3,10)
print(subtract, " subtract")

testArr2 = np.subtract(testArr2, subtract)
print("testARR2: ", testArr2)

arr3 = np.array([[[1,2,3,65],[4,5,6,84],[7,8,9,.002]]])
U, Sig, V = np.linalg.svd(arr3)
print("U: ", U, "\n sig: ", Sig, "\n V", V)

bigArr = np.arange(100).reshape(10,10)
print("bigARR: ", bigArr[:,[0,1]])
result = np.matmul(np.arange(50).reshape(5,10), bigArr[:,[0,1]])
print("result: ", result)

print("execution time: ",time.time() - startTime)

integer = 5
