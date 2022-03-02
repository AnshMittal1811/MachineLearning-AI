#@ Author Bryson Meiling November 2018
# CV2 works best with png files

import ope
import numpy as np
import time
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)

def smallestDimensions(imageList):
    smallestColumns = 10000  # gives a very large starting columns and rows
    smallestRows = 10000
    for image in imageList:
        img = cv2.imread(image)  # reads in images one by one.
        rows, columns, channels = img.shape
        if rows < smallestRows:  # if image has a smaller width, then makes that the smallest width.
            smallestRows = rows
        if columns < smallestColumns:
            smallestColumns = columns
    return smallestRows, smallestColumns

# def makePictureMatrixBGR(imageList, maxRows, maxColumns): #makes the numpy array with pixel data
#     pictureDataMatrix = np.arange(maxRows * maxColumns * len(imageList)*3).reshape(len(imageList),3, maxRows * maxColumns)
#     for image in imageList: #takes image one at a time
#         img = cv2.imread(image)  # reads in images one by one.
#         rows, columns, channels = img.shape
#         startPosX = (columns - maxColumns) // 2  # centers the area of interest in the middle despite differences in pixel count
#         startPosY = (rows - maxRows) // 2
#         counter = 0
#         for x in range(startPosX):
#             for y in range(startPosY):
#                 blue, green, red = (img[x + startPosX, y + startPosY])  # makes blue, green, red arrays for each picture
#
#                 counter +=1
#         imageIndex +=1
#     return pictureDataMatrix

def makePictureMatrixGray(imageList, maxRows, maxColumns):
    pictureDataMatrix = np.arange(maxRows * maxColumns * len(imageList)). reshape(len(imageList), maxRows * maxColumns) #reshapes matrix into a 4 x n-dimensions
    i = 0
    for image in imageList:
        img = cv2.imread(image,0)  # reads in images one by one in grayscale
        rows , columns = img.shape
        startPosX , startPosY = ((columns - maxColumns) // 2) , ((rows - maxRows) // 2)  # centers the area of interest in the middle despite differences in pixel count
        img = img[startPosY : startPosY + maxRows, startPosX : startPosX + maxColumns] #slices right dimensions out
        pictureDataMatrix[i] = img.flatten()
        i += 1
    return pictureDataMatrix

def muCenteredData(matrix):
    sums = np.sum(matrix, axis=1)
    average = np.divide(sums, len(matrix[0])) #finds average of a row
    average = np.repeat(average, len(matrix[0])).reshape(len(matrix), len(matrix[0])) #makes matrix with all the rows equaling the found averages
    matrix = np.subtract(matrix, average) #subtracts average from every value in the matrix
    return matrix

def covarianceCentered(baseDataMatrix):
    pass

def makeMoreEigenVectors(matrix , limitVectors):
    matrixM = np.matmul(np.transpose(matrix), matrix)  # gives really big rank one matrix
    uVectors, eigenValues, vVectors = np.linalg.svd(matrixM)  # SVD decomposition
    sum = 0
    neededVectors = 0
    significant = True

    for cell in eigenValues:  # finding the number of wanted vectors based on the eigenvectors that make up 95% of the data
        sum += cell
    count = 0
    while significant and count < limitVectors: #keeps significance and also takes in a maximum returned vectors to user
        if eigenValues[neededVectors] / sum > .05:  # if the eigenvalue is less that 5% of sum, then its not considered
            neededVectors += 1
        else:
            significant = False

    return vVectors[:, [0, neededVectors - 1]]

def make2EigenVectors(matrix):
    matrixM = np.matmul(np.transpose(matrix), matrix) #gives really big rank one matrix
    print("svd...")
    uVectors , eigenValues, vVectors = np.linalg.svd(matrixM) #SVD decomposition
    return vVectors[:,[0,1]] #for now, i take only two columns so that i can nicely print out two numbers, x and y, on a graph

def getMatrixValues(testedMatrix, eigenvectorMatrix):
    xAndYPoints = np.matmul( testedMatrix, eigenvectorMatrix)
    return xAndYPoints


def main():
    startTime2 = time.time()
    imageList = ["C:\\Users\\Bryson M\\Documents\\CS 1400\\MORE PYTHON\\Image Recognition\\Pictures\\50x50_painting.png",
                 "C:\\Users\\Bryson M\\Documents\\CS 1400\\MORE PYTHON\\Image Recognition\\Pictures\\50x50_painting2.png",
                 "C:\\Users\\Bryson M\\Documents\\CS 1400\\MORE PYTHON\\Image Recognition\\Pictures\\50x50_painting3.png",
                 "C:\\Users\\Bryson M\\Documents\\CS 1400\\MORE PYTHON\\Image Recognition\\Pictures\\50x50_painting4.png"]
    observedImage = ["C:\\Users\\Bryson M\\Documents\\CS 1400\\MORE PYTHON\\Image Recognition\\Pictures\\50x50_Test_Painting.png"]

    # imageList = ["C:\\Users\\Bryson M\\Documents\\CS 1400\\MORE PYTHON\\Image Recognition\\camera-test\\venv\\Images\\SpongbobJump.png",
    #              "C:\\Users\\Bryson M\\Documents\\CS 1400\\MORE PYTHON\\Image Recognition\\camera-test\\venv\\Images\\SpongebobHug.png",
    #              "C:\\Users\\Bryson M\\Documents\\CS 1400\\MORE PYTHON\\Image Recognition\\camera-test\\venv\\Images\\SpongebobJogging.png",
    #              "C:\\Users\\Bryson M\\Documents\\CS 1400\\MORE PYTHON\\Image Recognition\\camera-test\\venv\\Images\\SpongbobShowing.png"]
    # observedImage = ["C:\\Users\\Bryson M\\Documents\\CS 1400\\MORE PYTHON\\Image Recognition\\camera-test\\venv\\Images\\mario.png"]

#find dimensions
    smallestRows, smallestColumns = smallestDimensions(imageList + observedImage) #finds the smallest dimentions out of all the data for clean linalg operations

#extract data
    pictureDataMatrix = makePictureMatrixGray(imageList, smallestRows, smallestColumns)
    observedDataMatrix = makePictureMatrixGray(observedImage, smallestRows, smallestColumns)

#clean data
    cleanedData = muCenteredData(pictureDataMatrix) #cleans the data for a more consistant value
    cleanedObservedData = muCenteredData(observedDataMatrix)

    startTime3 = time.time()

#make Eigenvector comparison array
    eigenComparisionMatrix = make2EigenVectors(cleanedData) #makes the eigenvectors that will be used to compare against other pictures

#find x y points
    for image in imageList:
        baseLinePoints = getMatrixValues(cleanedData, eigenComparisionMatrix)
    print(baseLinePoints, ": base line points")

    observedPoint = getMatrixValues(cleanedObservedData, eigenComparisionMatrix)

    endTime = time.time()
#plot points
    plt.subplot(1,1,1)
    #graph.suptitle("Base pictures plotted")
    plt.scatter(*zip(*baseLinePoints)) #base points are red stars. https://stackoverflow.com/questions/21519203/plotting-a-list-of-x-y-coordinates-in-python-matplotlib
    plt.plot(observedPoint[0][0], observedPoint[0][1], 'or')
    plt.show()
    plt.waitforbuttonpress()
    print(observedPoint, " observed point!")

    print("Total Calulation Time: ", endTime - startTime)

startTime = time.time()

if __name__ == "__main__":
    main()

print("total run time: ",time.time() - startTime)