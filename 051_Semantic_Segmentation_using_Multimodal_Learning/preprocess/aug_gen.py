import cv2
import os
from keras.preprocessing.image import flip_axis, random_zoom, random_shear, random_shift, random_rotation
import numpy as np
from datetime import datetime
class gen_args:
    data_dir = None
    data_ext = None
    def __init__(self,dirr,ext):
        self.data_dir = dirr
        self.data_ext = ext

def data_augmentor(x,mode,row_axis=1,col_axis=0,channel_axis=-1):
    #temp = [0,0,0,0,0,0]
    #for i in range(len(mode)):
	#temp[6-i-1] = int(str(mode[-i-1]))	
	
    if int(mode[0]):
        x = flip_axis(x, 0)
        #print x.shape
 	
    if int(mode[1]):
        x = flip_axis(x,1)
        #print x.shape

    if int(mode[2]):
        x = random_rotation(x,360, row_axis, col_axis, channel_axis,fill_mode='reflect')
        x = np.swapaxes(x,0,1)
        x = np.swapaxes(x,1,2)
        '''M = cv2.getRotationMatrix2D((x.shape[1],x.shape[0]),np.random.randint(360),1)   #last argument is scale in rotation
        x = cv2.warpAffine(x,M,(x.shape[1],x.shape[0]), borderMode=cv2.BORDER_REFLECT)'''
        #print x.shape
		#del M

    if int(mode[3]):
        x = random_shift(x, 0.1,0.1, row_axis, col_axis, channel_axis,fill_mode='reflect')
        x = np.swapaxes(x,0,1)
        x = np.swapaxes(x,1,2)
        '''M = np.float32([[1,0,np.random.randint(x.shape[0])],[0,1,np.random.randint(x.shape[1])]])
        x = cv2.warpAffine(x,M,(x.shape[1],x.shape[0]), borderMode = cv2.BORDER_REFLECT)'''
        #print x.shape
        #del M

    if int(mode[4]):
        x = random_shear(x, 1, row_axis, col_axis, channel_axis,fill_mode='reflect')
        x = np.swapaxes(x,0,1)
        x = np.swapaxes(x,1,2)
        '''pts1 = np.float32([[np.random.randint(x.shape[0]),np.random.randint(x.shape[1])],[np.random.randint(x.shape[0]),np.random.randint(x.shape[1])],[np.random.randint(x.shape[0]),np.random.randint(x.shape[1])]])
        pts2 = np.float32([[np.random.randint(x.shape[0]),np.random.randint(x.shape[1])],[np.random.randint(x.shape[0]),np.random.randint(x.shape[1])],[np.random.randint(x.shape[0]),np.random.randint(x.shape[1])]])
        M = cv2.getAffineTransform(pts1,pts2)
        x = cv2.warpAffine(x,M,(x.shape[1],x.shape[0]),borderMode = cv2.BORDER_REFLECT)'''
        #print x.shape
        #del M
		#del pts1
		#del pts2

    if int(mode[5]):
        x = random_zoom(x, (1.1,1.1), row_axis, col_axis, channel_axis,fill_mode='reflect')
        x = np.swapaxes(x,0,1)
        x = np.swapaxes(x,1,2)
        #print x.shape
        

    return x

file_path = open('/home/captain_jack/Downloads/freiburg_forest_dataset/train/train.txt','r')
names = file_path.readlines()
file_path.close()

rgb_args = gen_args ('/home/captain_jack/Downloads/freiburg_forest_dataset/train/rgb/','.jpg')
nir_args = gen_args ('/home/captain_jack/Downloads/freiburg_forest_dataset/train/nir_color/','.png')
label_args = gen_args ('/home/captain_jack/Downloads/freiburg_forest_dataset/train/GT_color/','.png')
 
for i in range(len(names[:4])):
    print '=========================   IMAGE '+str(i) + '============================'
    file_name =  names[i].strip('\n')
    num = 1
    while num<64:
		s = int(str(datetime.now()).split('.')[1][3:])
		im_rgb = cv2.imread(rgb_args.data_dir+file_name+rgb_args.data_ext)
		im_nir = cv2.imread(nir_args.data_dir+file_name+nir_args.data_ext)
		im_label = cv2.imread(label_args.data_dir+file_name+label_args.data_ext)
		mode = bin(num)[2:]
		mode = '0'*(6-len(mode))+mode
		print '-------- mode '+str(num)+' - '+mode+ ' ----------'
		np.random.seed(s)
		im_rgb = data_augmentor(im_rgb,mode)
		np.random.seed(s)        
		im_nir = data_augmentor(im_nir,mode)
		np.random.seed(s)        
		im_label = data_augmentor(im_label,mode)
		cv2.imwrite(rgb_args.data_dir+'Augmented/'+file_name+'_'+mode+rgb_args.data_ext,im_rgb)
		cv2.imwrite(nir_args.data_dir+'Augmented/'+file_name+'_'+mode+nir_args.data_ext,im_nir)
		cv2.imwrite(label_args.data_dir+'Augmented/'+file_name+'_'+mode+label_args.data_ext,im_label)
		num = num + 1
