import numpy as np

import os
import csv

from PIL import Image
import shutil    
        
att = open('./Celeb/attrs.txt','r')
imgs = os.listdir('./Celeb/data1024')
imgs.sort()

# print(len(attlines))
attlines = []
while True:
    attline = att.readline().split()
    if not attline:break
    attlines.append(attline)

# print(attlines)


att_male = open('./Celeb/male.txt','w')
att_female = open('./Celeb/female.txt','w')

os.mkdir('./Celeb/mult/males')
os.mkdir('./Celeb/mult/females')

for i in range(30000):
    nums = int(attlines[i][0])+1
    iname = str(nums).zfill(5) +'.jpg'
    print(iname)
    if attlines[i][21] =='1':
        att_male.write(' '.join(attlines[i])+'\n')
        shutil.copy(os.path.join('./Celeb/data1024',iname),os.path.join('./Celeb/mult/males',iname))
    elif attlines[i][21] =='-1':
        att_female.write(' '.join(attlines[i])+'\n')
        shutil.copy(os.path.join('./Celeb/data1024',iname),os.path.join('./Celeb/mult/females',iname))
    