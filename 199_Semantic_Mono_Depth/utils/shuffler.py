import os.path
import sys
import re
import random

lines = []
src = open('filenames/kitti_semantic_stereo_2015_test_files.txt')
train_file = open('filenames/kitti_semantic_stereo_2015_train_split.txt',"w")  
test_file = open('filenames/kitti_semantic_stereo_2015_test_split.txt',"w") 
for i in range(0,200):
  line = src.readline()
  lines.append(line)
  
random.shuffle(lines)
for i in range(0,160):
  train_file.write(lines[i])
  
for i in range(160,200):
  test_file.write(lines[i])  
