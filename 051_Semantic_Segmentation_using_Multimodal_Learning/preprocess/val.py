import os
import shutil
train_path = '/home/captain_jack/Downloads/freiburg_forest_annotated/train/'
test_path = '/home/captain_jack/Downloads/freiburg_forest_annotated/test/'
val_path = '/home/captain_jack/Downloads/freiburg_forest_annotated/valid/'


train_dirs = os.listdir(train_path)
test_dirs = os.listdir(test_path)


for folder in train_dirs:
    os.makedirs('/home/captain_jack/Downloads/freiburg_forest_annotated/valid/'+folder)
    image_paths = sorted(os.listdir(train_path+folder))
    for f in range(30):
        #print len(os.listdir('/home/captain_jack/Downloads/freiburg_forest_annotated/valid/'+folder))
        shutil.move(train_path+folder+'/'+image_paths[f], val_path+folder+'/'+image_paths[f])
        
for folder in test_dirs:
    image_paths = sorted(os.listdir(test_path+folder))
    for f in range(36):
        shutil.move(test_path+folder+'/'+image_paths[f], val_path+folder+'/'+image_paths[f])


