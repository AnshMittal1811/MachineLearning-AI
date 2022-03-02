import os
import cv2
import time

startTime = time.time()
'''
def displayImage(imageList):
    for paths in imagesList:
        img1 = cv2.imread(paths)
        cv2.imshow("picture", img1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

BASE_DIR= os.path.dirname(os.path.abspath(__file__)) #find the file path of this program
IMAGE_DIR = os.path.join(BASE_DIR, "Images") #finds the folder of images
imagesList = []
for root,dirs, files in os.walk(IMAGE_DIR):
    for file in files:
        if file.endswith("png"): #if the files is the right picture type it selects it.
            path = os.path.join(root,file)
            imagesList.append(path) #adds path to a list

print(imagesList)
#displayImage(imagesList)
a = ["C:\\Users\\Bryson M\\Documents\\CS 1400\\MORE PYTHON\\Image Recognition\\camera-test\\venv\\Images\\apple_ex.png"]
img1 = cv2.imread(a[0])
cv2.imshow("apple", img1) #displays apple before modification
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
width, height, channels = img1.shape #finds aspects of the picture
print(width,height,channels)
widthI = int(width)
heightI = int(height)
startTime = time.time()
"""
def makePictureArray(file):
    img1 = cv2.imread(file)
    width, height, channels = img1.shape
    bluePixelData = []
    greenPixelData = []
    redPixelData = []
    for x in range(width): #changes the portion of the picture to black
        for y in range(height):
            blue, green, red=(img1[x,y])
            bluePixelData.append([blue]) #adds pixels values to three seperate lists
            greenPixelData.append([green])
            redPixelData.append([red])
            img1[x,y] = [0,0,0]
    cv2.imshow("apple", img1)  # displays picture after modification
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def makePictureMatrix(imageFile, minRows, minColumns):
    imgT = cv2.imread(imageFile)
    rows, columns, channels = imgT.shape
    bluePixelData = []
    greenPixelData = []
    redPixelData = []
    for x in range(rows//5): #turns bottom 4/5 of picture dark
        for y in range(columns//2): #turns right half dark
            blue, green, red = (imgT[x,y])
            #append data
            imgT[x,y]=[0,0,0]
    cv2.imshow("test of theory: ", imgT)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




cv2.imshow("apple", img1) #displays picture after modification
cv2.waitKey(0)
cv2.destroyAllWindows()

print( img1.shape)
px = img1[150,150]
print("px: ",px)
#print(bluePixelData[100])
#print(greenPixelData[100])
#print(redPixelData[100])

print( time.time() - startTime , "seconds")

string1 = "img"
nameList = ([string1 + str(i) for i in range(1,6)])
print(nameList)
makePictureArray(a[0])

makePictureMatrix(a[0],100,100)
'''
imageList = [r"C:\Users\Bryson M\Documents\CS 1400\MORE PYTHON\Image Recognition\Pictures\50x50_painting.png",
                 "C:\\Users\\Bryson M\\Documents\\CS 1400\\MORE PYTHON\\Image Recognition\\Pictures\\50x50_painting2.png",
                 "C:\\Users\\Bryson M\\Documents\\CS 1400\\MORE PYTHON\\Image Recognition\\Pictures\\50x50_painting3.png",
                 "C:\\Users\\Bryson M\\Documents\\CS 1400\\MORE PYTHON\\Image Recognition\\Pictures\\50x50_painting4.png"]

for im in imageList:
    img3 = cv2.imread(im)
    cv2.imshow("image", img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()





