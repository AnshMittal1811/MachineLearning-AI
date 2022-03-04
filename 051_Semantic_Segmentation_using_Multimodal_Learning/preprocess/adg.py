import cv2
files = open('/home/captain_jack/Downloads/freiburg_forest_annotated/Otherformats/train.txt')
names = files.readlines()
files.close()


print len(names)
for i in range(len(names)):
    path = '/home/captain_jack/Downloads/freiburg_forest_annotated/Otherformats/train/nir_color/'+names[i].strip('\n')+'.png'
    print path
    #a  =  cv2.imread(path)
    #print a
    #print a.shape

    #b = fix_size(a,[512,960])
    #print b.shape
