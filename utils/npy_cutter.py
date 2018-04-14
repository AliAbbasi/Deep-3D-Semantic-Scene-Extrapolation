import numpy as np
import glob
import sys
import datetime
import utils

#------------------------------------------------------------------------ 

def npy_cutter(item):
    x, y, z = 84, 44, 84 
    scene = np.zeros((x, y, z))
    try:
        x_, y_, z_ = item.shape
    
        if   x<=x_ and y<=y_ and z<=z_: 
            scene           =item[:x, :y, :z]
            # print "1q11"
        elif x<=x_ and y>=y_ and z<=z_:
            scene[:, :y_, :]=item[:x, :, :z]
            # print "22"
        elif x<=x_ and y<=y_ and z>=z_:
            scene[:, :, :z_]=item[:x, :y, :]
            # print "3333"
        elif x<=x_ and y>=y_ and z>=z_: 
            scene[:, :y_, :z_]=item[:x, :, :] 
            # print "4444"
        elif x>=x_ and y<=y_ and z<=z_:
            scene[:x_, :, :]=item[:, :y, :z]
            # print "555"
        elif x>=x_ and y>=y_ and z<=z_:
            scene[:x_, :y_, :]=item[:, :, :z]
            # print "666"
        elif x>=x_ and y<=y_ and z>=z_:
            scene[:x_, :, :z_]=item[:, :y, :]
            # print "7777"
        elif x>=x_ and y>=y_ and z>=z_:
            scene[:x_, :y_, :z_]=item
            # print "8888"
        else: 
            pass
            # print "999"
    except: 
        pass
    return scene

#------------------------------------------------------------------------

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
        npy_cutter(item) 
        
#------------------------------------------------------------------------

def load_time_test():
    counter = 0
    for npy_file in glob.glob('house/*.npy'):
        counter += 1
        item = np.load(npy_file)
        npy_cutter(item)
        if counter % 128==0:
            print counter
            print datetime.datetime.now().time()

#------------------------------------------------------------------------

def scene_load_and_visualize_test():  

    for npyFile in glob.glob('house/*.npy'): 
        tr_scene, tr_label = [], [] 
        scene = npy_cutter(np.load(npyFile))  
        tr_scene = scene[ 0:84, 0:44, 0:42  ]  # input 
        tr_label = scene[ 0:84, 0:44, 42:84 ]  # gt   
        
        
        utils.npy_to_ply(str(npyFile) + "_scene_", tr_scene)
        utils.npy_to_ply(str(npyFile) + "_label_", tr_scene)
        utils.npy_to_ply(str(npyFile) + "_self_", npy_cutter(np.load(npyFile)))
        break
        
#------------------------------------------------------------------------

if __name__ == '__main__': 
    # load_time_test()
    scene_load_and_visualize_test()
    
    
    