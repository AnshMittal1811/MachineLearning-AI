
#====================================================================================================================

import numpy as np
import glob
from multiprocessing import Pool
import os.path
import os
import random

#==================================================================================================================== 

colors = [" 0 0 0 255       ", " 139 0 0 255     ", "  0 128 0 255    ", " 173 216 230 255 ",
          " 0 0 255 255     ", " 255 0 0 255     ", " 218 165 32 255  ", " 210 180 140 255 ", 
          " 128 0   128 255 ", " 0  0 139 255    ", " 255 255 0 255   ", " 128 128 128 255 ",
          " 0 100 0 255     ", " 255 165 0 255   ", " 138 118 200 255 ", " 236 206 244 255 ",
          " 126 172 209 255 ", " 237 112 24 255  ", " 158 197 220 255 ", " 21 240 24 255   ",
          " 90 29 205 255   ", " 183 246 66 255  ", " 224 54 238 255  ", " 39 129 50 255   ",
          " 252 204 171 255 ", " 255 18 39 255   ", " 118 76 69 255   ", " 139 212 79 255  ",
          " 46 14 67 255    ", " 142 113 129 255 ", " 30 14 35 255    ", " 17 90 54 255    ",
          " 125 89 247 255  ", " 166 18 75 255   ", " 129 142 18 255  ", " 147 10 255 255  ",
          " 32 168 135 255  ", " 245 199 6 255   ", " 231 118 238 255 ", " 84 35 213 255   ",
          " 214 230 80 255  ", " 236 23 17 255   ", " 92 207 229 255  ", " 49 243 237 255  ",
          " 252 23 25 255   ", " 209 224 126 255 ", " 111 54 3 255    ", " 96 11 79 255    ",
          " 169 56 226 255  ", " 169 68 202 255  ", " 107 32 121 255  ", " 158 3 146 255   ",
          " 68 57 54 255    ", " 212 200 217 255 ", " 17 30 170 255   ", " 254 162 238 255 ",
          " 16 120 52 255   ", " 104 48 251 255  ", " 176 49 253 255  ", " 67 84 223 255   ",
          " 101 88 52 255   ", " 204 50 193 255  ", " 56 209 118 255  ", " 79 74 216 255   ",
          " 104 142 255 255 ", " 15 228 195 255  ", " 185 168 157 255 ", " 227 7 222 255   ",
          " 243 188 17 255  ", " 20 85 135 255   ", " 95 27 18 255    ", " 189 126 21 255  ",
          " 69 254 247 255  ", " 84 91 111 255   ", " 8 153 222 255   ", " 188 72 148 255  ",
          " 218 50 8 255    ", " 183 217 27 255  ", " 61 4 234 255    ", " 31 113 81 255   ",
          " 75 130 78 255   ", " 128 232 57 255  ", " 16 183 77 255   ", " 91 43 145 255   ",
          " 38 19 130 255   ", " 64 236 113 255  ", " 248 3 144 255   ", " 194 157 62 255  ",
          " 143 219 101 255 ", " 136 37 208 255  ", " 102 144 241 255 ", " 158 126 247 255 ",
          " 40 207 130 255  ", " 88 131 224 255  ", " 175 30 23 255   ", " 42 224 197 255  ",
          " 23 175 34 255   ", " 118 144 216 255 ", " 32 128 149 255  ", " 200 185 126 255 ",
          " 114 11 76 255   ", " 28 60 36 255    ", " 168 148 36 255  ", " 57 246 83 255   "]
          
#====================================================================================================================

def write_cost_accuray_plot(directory, train_cost, valid_cost, train_accu1, train_accu2, valid_accu1, valid_accu2): 
    output = open(directory + "/costs.py" , 'w') 
    output.write( "import matplotlib.pyplot as plt" + "\r\n" )
    output.write( "train_cost = []" + "\r\n" )
    output.write( "valid_cost = []" + "\r\n" )
    output.write( "steps      = []" + "\r\n" ) 
    for i in range(len(train_cost)):
        output.write( "steps.append("+ str(i) +")" + "\r\n" )
    for i in range(len(train_cost)):
        output.write( "train_cost.append("+ str(train_cost[i]) +")" + "\r\n" )
    output.write( "\r\n \r\n \r\n" )
    for i in range(len(valid_cost)):
        for j in range(100):
            output.write( "valid_cost.append("+ str(valid_cost[i]) +")" + "\r\n" )   
    output.write( "plt.plot( steps , train_cost, color ='b', lw=1 )   " + "\r\n" )
    # output.write( "plt.plot( steps , valid_cost, color ='g', lw=1 )   " + "\r\n" )
    output.write( "plt.xlabel('Steps', fontsize=14)                   " + "\r\n" )
    output.write( "plt.ylabel('Cost',  fontsize=14)                   " + "\r\n" )
    output.write( "plt.suptitle('Blue: Train Cost, Green: Valid Cost')" + "\r\n" )
    output.write( "plt.show()                                         " + "\r\n" )  
    print ("costs.py file is created!")
    
    #-----------------------------------------------------------------------------
    
    output = open(directory + "/accuracy.py" , 'w') 
    output.write( "import matplotlib.pyplot as plt" + "\r\n" )
    output.write( "train_accu1 = []" + "\r\n" )
    output.write( "train_accu2 = []" + "\r\n" )
    output.write( "valid_accu1 = []" + "\r\n" )
    output.write( "valid_accu2 = []" + "\r\n" )
    output.write( "steps      = []" + "\r\n" ) 
    for i in range(len(train_accu1)):
        output.write( "steps.append("+ str(i) +")" + "\r\n" )
    output.write( "\r\n \r\n \r\n" )
    for i in range(len(train_accu1)):
        output.write( "train_accu1.append("+ str(train_accu1[i]) +")" + "\r\n" )
    output.write( "\r\n \r\n \r\n" )
    for i in range(len(train_accu2)):
        output.write( "train_accu2.append("+ str(train_accu2[i]) +")" + "\r\n" )       
    output.write( "\r\n \r\n \r\n" )
    for i in range(len(valid_accu1)): 
        output.write( "valid_accu1.append("+ str(valid_accu1[i]) +")" + "\r\n" )   
    output.write( "\r\n \r\n \r\n" )
    for i in range(len(valid_accu2)): 
        output.write( "valid_accu2.append("+ str(valid_accu2[i]) +")" + "\r\n" ) 
    output.write( "plt.plot( steps , train_accu1, color ='b', lw=3 )   " + "\r\n" )
    output.write( "plt.plot( steps , train_accu2, color ='b', lw=1 )   " + "\r\n" )
    output.write( "plt.plot( steps , valid_accu1, color ='g', lw=3 )   " + "\r\n" )
    output.write( "plt.plot( steps , valid_accu2, color ='g', lw=1 )   " + "\r\n" )
    output.write( "plt.xlabel('Steps', fontsize=14)                   " + "\r\n" )
    output.write( "plt.ylabel('Accuracy',  fontsize=14)               " + "\r\n" )
    output.write( "plt.suptitle('Blue: Train Accu, Green: Valid Accu')" + "\r\n" )
    output.write( "plt.show()                                         " + "\r\n" )  
    print ("accuracy.py file is created!")  
    
#====================================================================================================================

# TODO: add directory to save as input argument
def npy_to_ply(name, input_npy_file):  # the input is a npy file
    output_scene = input_npy_file
    output = open(str(name) + ".ply", 'w')
    ply = ""
    ver_num = 0
    if len(output_scene.shape) > 2:
        for idx1 in range(output_scene.shape[0]):
            for idx2 in range(output_scene.shape[1]):
                for idx3 in range(output_scene.shape[2]):
                    if output_scene[idx1][idx2][idx3] >= 1:
                        ply = ply + str(idx1) + " " + str(idx2) + " " + str(idx3) + str(
                            colors[int(output_scene[idx1][idx2][idx3])]) + "\n"
                        ver_num += 1
    else:
        for idx1 in range(output_scene.shape[0]):
            for idx2 in range(output_scene.shape[1]): 
                if output_scene[idx1][idx2] >= 1:
                    ply = ply + str(idx1) + " " + str(idx2) + " 0" + str(
                        colors[int(output_scene[idx1][idx2])]) + "\n"
                    ver_num += 1
    output.write("ply" + "\n")
    output.write("format ascii 1.0" + "\n")
    output.write("comment VCGLIB generated" + "\n")
    output.write("element vertex " + str(ver_num) + "\n")
    output.write("property float x" + "\n")
    output.write("property float y" + "\n")
    output.write("property float z" + "\n")
    output.write("property uchar red" + "\n")
    output.write("property uchar green" + "\n")
    output.write("property uchar blue" + "\n")
    output.write("property uchar alpha" + "\n")
    output.write("element face 0" + "\n")
    output.write("property list uchar int vertex_indices" + "\n")
    output.write("end_header" + "\n")
    output.write(ply)
    output.close()
    print (str(name) + ".ply is Done.!") 
    
    """ 91 classes:
    81 shoes
    43 picture_frame
    66 ottoman
    9 chair
    32 stairs
    17 window
    18 stand
    87 safe
    41 bathroom_stuff
    85 drinkbar
    40 fan
    8 desk
    61 outdoor_lamp
    44 fireplace
    35 recreation
    34 sofa
    53 gym_equipment
    22 table_and_chair
    5 hanger
    60 fence
    56 garage_door
    13 computer
    37 bed
    27 mirror
    69 candle
    49 clock
    72 pet
    14 wardrobe_cabinet
    82 trinket
    33 rug
    86 cart
    88 mailbox
    21 sink
    54 headstone
    6 kitchen_cabinet
    65 bench_chair
    50 trash_can
    76 kitchen_set
    68 whiteboard
    67 workplace
    78 pillow
    42 heater
    16 indoor_lamp
    45 hanging_kitchen_cabinet
    1 wall
    71 ATM
    79 magazines
    12 tv_stand
    10 table
    24 toy
    4 unknown
    83 outdoor_spring
    26 music
    58 vehicle
    47 curtain
    7 kitchen_appliance
    0 empty
    2 ceiling
    15 door
    80 tripod
    84 cloth
    30 kitchenware
    63 grill
    11 television
    28 shoes_cabinet
    39 air_conditioner
    38 household_appliance
    75 roof
    23 shower
    59 column
    29 books
    64 bathtub
    55 coffin
    31 toilet
    62 outdoor_seating
    89 storage_bench
    57 decoration
    25 dressing_table
    36 shelving
    3 floor
    73 outdoor_cover
    77 wood_board
    19 dresser
    20 plant
    46 vase
    74 arch
    70 pool
    52 partition
    51 person
    48 switch
    """

#====================================================================================================================

def npy_cutter(item, scene_shape):
    x, y, z = scene_shape[0], scene_shape[1], scene_shape[2]
    scene = np.zeros((x, y, z))
    try:
        x_, y_, z_ = item.shape
    
        if   x<=x_ and y<=y_ and z<=z_: 
            scene           =item[:x, :y, :z] 
        elif x<=x_ and y>=y_ and z<=z_:
            scene[:, :y_, :]=item[:x, :, :z] 
        elif x<=x_ and y<=y_ and z>=z_: 
            scene[:, :, ((z-z_)/2):(z_+(z-z_)/2)]=item[:x, :y, :] 
        elif x<=x_ and y>=y_ and z>=z_: 
            scene[:, :y_, ((z-z_)/2):(z_+(z-z_)/2)]=item[:x, :, :]  
        elif x>=x_ and y<=y_ and z<=z_:
            scene[:x_, :, :]=item[:, :y, :z] 
        elif x>=x_ and y>=y_ and z<=z_:
            scene[:x_, :y_, :]=item[:, :, :z] 
        elif x>=x_ and y<=y_ and z>=z_:
            scene[:x_, :, ((z-z_)/2):(z_+(z-z_)/2)]=item[:, :y, :] 
        elif x>=x_ and y>=y_ and z>=z_:
            scene[:x_, :y_, ((z-z_)/2):(z_+(z-z_)/2)]=item 
        else: 
            pass 
    except: 
        pass
        
    return scene

#====================================================================================================================

def validity_test():
    test_arr = []
    test_arr.append(np.ones((100,100,100))) 
    test_arr.append(np.ones((100,40,100)) )
    test_arr.append(np.ones((100,100,40)) )
    test_arr.append(np.ones((100,40,40))  )  
    test_arr.append(np.ones((40,100,100)) )
    test_arr.append(np.ones((40,40,100))  )
    test_arr.append(np.ones((40,100,40))  )
    test_arr.append(np.ones((40,40,40))   )
    test_arr.append(np.ones((84,46,84))   )
    test_arr.append(np.ones((90,46,80))   )
    for item in test_arr: 
        npy_cutter(item, item.shape) 
        
#====================================================================================================================

def load_time_test():
    counter = 0
    for npy_file in glob.glob('house/*.npy'):
        counter += 1
        item = np.load(npy_file)
        npy_cutter(item, item.shape)
        if counter % 128==0:
            print counter
            print datetime.datetime.now().time()

#====================================================================================================================

def scene_load_and_visualize_test():   
    for npy_file in glob.glob('house/*.npy'):  
        tr_scene, tr_label = [], [] 
        scene = npy_cutter(np.load(npy_file), np.load(npy_file).shape)  
        tr_scene = scene[ 0:84, 0:44, 0:42  ]  # input 
        tr_label = scene[ 0:84, 0:44, 42:84 ]  # gt   
        
        npy_to_ply(str(npy_file) + "_scene_", tr_scene)
        npy_to_ply(str(npy_file) + "_label_", tr_scene)
        npy_to_ply(str(npy_file) + "_self_", npy_cutter(np.load(npy_file), np.load(npy_file).shape))
        break
        
#====================================================================================================================

def show_scene_size():
    counter = 0
    all = 0
    for npy_file in glob.glob('house/*.npy'): 
        all += 1
        dims = np.load(npy_file).shape
        if dims[0] < 84 or dims[1] < 44 or dims[2] < 84:
            counter += 1
            if counter % 1000 == 0:
                print counter, all
    print ("final count: ", counter, all)
    
#====================================================================================================================

def npy_cutter_test():
    for npy_file in glob.glob('house/*.npy'):
        if np.load(npy_file).shape[2] <= 84:
            print "file name: " , str(npy_file)
            item = np.load(npy_file)
            print item.shape
            scene = npy_cutter(item, item.shape) 
            train_scene = scene[ :, : ,  0:42]
            label_scene = scene[ :, : , 42:88]
            npy_to_ply( str(npy_file) + "train_scene", train_scene)
            npy_to_ply( str(npy_file) + "label_scene", label_scene)
            npy_to_ply( str(npy_file) + "scene", scene)

#====================================================================================================================

def reduce_classes_to_13(npy_file): 
    scene   = np.load(npy_file)   
    new_scn = np.zeros(scene.shape)
     
    new_scn[np.where(scene==2)]  = 0   # 'ceiling'   original is 1  
    new_scn[np.where(scene==3)]  = 2   # 'floor'    
    new_scn[np.where(scene==1)]  = 3   # 'wall'      
    new_scn[np.where(scene==17)] = 4   # 'window'              
    new_scn[np.where(scene==15)] = 5   # 'door'                
    new_scn[np.where(scene==9)]  = 6   # 'chair'               
    new_scn[np.where(scene==37)] = 7   # 'bed'                 
    new_scn[np.where(scene==34)] = 8   # 'sofa'                
    new_scn[np.where(scene==10)] = 9   # 'table'               
    new_scn[np.where(scene==6)]  = 10  # 'kitchen_cabinet'     
    new_scn[np.where(scene==36)] = 11  # 'shelving'            
    new_scn[np.where(scene==14)] = 12  # 'wardrobe_cabinet'    
    new_scn[np.where(scene==38)] = 13  # 'household_appliance'   
    
    
    if len(new_scn[np.where(new_scn >= 6)]) > 0:
        np.save('house_2/' + str(npy_file[6:-4]) + ".npy", new_scn) 
        
#====================================================================================================================

def reduce_classes_to_13_main():
    index = 0 
    batch_size = 5
    p = Pool(batch_size)
    batch_arr = []
    counter = 0
    
    for npy_file in glob.glob('house/*.npy'): 
        if not os.path.isfile('house_2/' + str(npy_file[6:])):
            print npy_file
            if counter < batch_size:
                batch_arr.append(npy_file)
                counter += 1
            else:
                counter = 0
                p.map(reduce_classes_to_13, batch_arr)
                batch_arr = [npy_file]
                counter += 1
                index += 1
                print index

    # one by one
    for npy_file in batch_arr:
        reduce_classes_to_13(npy_file)

#====================================================================================================================

def fetch_test_set(directory='house_2/',test_st_size=10000):
    for i in xrange(test_st_size):  
        file = random.choice(os.listdir(directory))
        print str(file)
        npy_file = np.load(directory + file)
        np.save("test_data/" + str(file), npy_file)
        os.remove(directory + file)

#====================================================================================================================

def fetch_random_batch(directory, bs):
    batch_arr = []
    for i in xrange(bs):
        file = random.choice(os.listdir(directory))
        batch_arr.append(directory + file)
    return batch_arr

#====================================================================================================================

def project_on_2D(npyFile):
    scene = npy_cutter(np.load(npyFile), [84, 44, 84])
    scene2D = np.zeros((84, 84))
    
    for x in range(84):
        for y in range(84):
            topVoxel = np.argmax(scene[x, :, y]) 
            scene2D[x, y] = scene[x, topVoxel, y] 
        
    np.save('house_2d/' + str(npyFile[8:-4]) + ".npy", scene2D)
    
#====================================================================================================================

def project_on_2D_main():   
    index = 0
    
    batch_size = 50
    p          = Pool(batch_size)
    batchArr   = []
    counter    = 0 

    #50 by 50
    for npyFile in glob.glob('house_2/*.npy'):
        index += 1
        batch = [] 
        
        if counter < batch_size:    
            batchArr.append(npyFile)
            counter += 1 
        else:
            counter = 0
            batch.append(p.map(project_on_2D, batchArr))
            batchArr = [] 
            batchArr.append(npyFile)  
            counter += 1
            print (index)
    
    #1 by 1 
    for npyFile in batchArr:    
        project_on_2D(npyFile)

#====================================================================================================================

def precision_recall(score, label, batch_size, classes_count): 
    tp, fp, fn = [], [], []
    
    # TP: True Positive, means region segmented as mass that proved to be mass.
    for i in range(classes_count):
        tp.append(len(np.where((score==label)&(score==i))[0]))
    
    # FP: False Positive, means region segmented as mass that proved to be not mass.    
    for i in range(classes_count):
        indices = np.where(score==i)
        vox_labels = label[indices]
        fp.append(len(np.where(vox_labels!=i)[0]))
        
    # FN: False Negative, means regon segmented as not mass that proved to be mass.
    for i in range(classes_count):
        indices = np.where(label==i)
        vox_score = score[indices]
        fn.append(len(np.where(vox_labels!=i)[0]))
        
    tp = np.asarray(tp) / batch_size * 1.0
    fp = np.asarray(fp) / batch_size * 1.0
    fn = np.asarray(fn) / batch_size * 1.0
    
    precision = np.zeros(classes_count)
    recall = np.zeros(classes_count)
    
    for i in range(len(tp)):
        precision[i] = (tp[i] / (tp[i] + fp[i])) if (tp[i] + fp[i]) != 0.0 else 0.0
        recall[i]    = (tp[i] / (tp[i] + fn[i])) if (tp[i] + fn[i]) != 0.0 else 0.0
    
    return precision, recall
    
#====================================================================================================================

if __name__ == '__main__':
    # load_time_test()
    # scene_load_and_visualize_test() 
    # show_scene_size()
    # npy_cutter_test()
    # reduce_classes_to_13_main() 
    # fetch_test_set()
    # print len(fetch_random_batch('test_data/', 64))
    # project_on_2D_main()
    pass 
