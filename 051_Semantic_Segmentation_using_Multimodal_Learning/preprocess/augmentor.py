import cv2
from datetime import datetime
from keras.preprocessing.image import *

'''
class aug_state:
    def __init__(self,flip_axis_index=0,rotation_range=360,height_range=0.2,width_range=0.2,shear_intensity=1,color_intensity=40,fill_mode='reflect',zoom_range=(1.2,1.2)):
         self.flip_axis_index=flip_axis_index
         self.rotation_range=rotation_range
         self.height_range=height_range
         self.width_range=width_range
         self.shear_intensity=shear_intensity
         self.color_intensity=color_intensity
         self.zoom_range=zoom_range
         self.fill_mode=fill_mode


def data_augmentor(x,state,row_axis=1,col_axis=0,channel_axis=-1,
    bool_flip_axis=0,
    bool_random_rotation=0,
    bool_random_shift=0,
    bool_random_shear=0,
    bool_random_zoom=0):

'''
class aug_state:
    def __init__(self,flip_axis_index=0,rotation_range=360,zoom_range=(1.5,1.5)):
         self.flip_axis_index=flip_axis_index
         self.rotation_range=rotation_range
         self.zoom_range=zoom_range
         
def data_augmentor(x,state,row_axis=1,col_axis=0,channel_axis=-1):
    temp = np.random.randint(2,size=5)
    if temp[0]:
        x = flip_axis(x, state.flip_axis_index)
        
    if temp[1]:
        M = cv2.getRotationMatrix2D((x.shape[1]/2,x.shape[0]/2),np.random.randint(state.rotation_range),1)   #last argument is scale in rotation
        x = cv2.warpAffine(x,M,(x.shape[1],x.shape[0]), borderMode=cv2.BORDER_REFLECT)
        
    if temp[2]:
        M = np.float32([[1,0,np.random.randint(x.shape[0])],[0,1,np.random.randint(x.shape[1])]])
        x = cv2.warpAffine(x,M,(x.shape[1],x.shape[0]), borderMode = cv2.BORDER_REFLECT)
        
    if temp[3]:
        pts1 = np.float32([[np.random.randint(x.shape[0]),np.random.randint(x.shape[1])],[np.random.randint(x.shape[0]),np.random.randint(x.shape[1])],[np.random.randint(x.shape[0]),np.random.randint(x.shape[1])]])
        pts2 = np.float32([[np.random.randint(x.shape[0]),np.random.randint(x.shape[1])],[np.random.randint(x.shape[0]),np.random.randint(x.shape[1])],[np.random.randint(x.shape[0]),np.random.randint(x.shape[1])]])
        M = cv2.getAffineTransform(pts1,pts2)
        dst = cv2.warpAffine(x,M,(x.shape[1],x.shape[0]),borderMode = cv2.BORDER_REFLECT)
        
    if temp[4]:
        x = random_zoom(x, state.zoom_range, row_axis, col_axis, channel_axis,fill_mode='reflect')
        x = np.swapaxes(x,0,1)
        x = np.swapaxes(x,1,2)
        

    return x
'''

    d1 = datetime.now()
    if temp[0]:
        y = flip_axis(y, state.flip_axis_index)
    if temp[1]:
        y = random_rotation(y, state.rotation_range, row_axis, col_axis, channel_axis,fill_mode=state.fill_mode)
        y = np.swapaxes(y,0,1)
        y = np.swapaxes(y,1,2)
        
    if temp[2]:
        y = random_shift(y, state.width_range, state.height_range, row_axis, col_axis, channel_axis,fill_mode=state.fill_mode)
        y= np.swapaxes(y,0,1)
        y= np.swapaxes(y,1,2)
        
    if temp[3]:
        y = random_shear(y, state.shear_intensity, row_axis, col_axis, channel_axis,fill_mode=state.fill_mode)
        y = np.swapaxes(y,0,1)
        y = np.swapaxes(y,1,2)
        
        
    if temp[4]:
        y = random_zoom(y, state.zoom_range, row_axis, col_axis, channel_axis,fill_mode=state.fill_mode)
        y = np.swapaxes(y,0,1)
        y = np.swapaxes(y,1,2)
    d2 = datetime.now()
    a = str(d2-d1)[2:].split(':')
    t2 = float(a[0])*60+float(a[1])
    print 'opencv: '+str(t1)
    print 'keras : '+str(t2)
    print 'diff  : '+str(t2-t1)
    print 'improv: '+str((t1-t2)*100/t2)
     
    return x,y
'''


state_aug = aug_state() 

im = cv2.imread('/home/captain_jack/Downloads/freiburg_forest_dataset/train/rgb/b137-492.jpg')

cv2.imshow('window',im)
while True:
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

im_aug = data_augmentor(im,state_aug)




