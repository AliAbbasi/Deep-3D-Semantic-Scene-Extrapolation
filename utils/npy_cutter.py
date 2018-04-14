import numpy as np
import glob
import sys
import datetime

#------------------------------------------------------------------------

x, y, z = 84, 46, 84 
scene = np.zeros((x, y, z))

#------------------------------------------------------------------------

def npy_cutter(item):
    global scene
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

if __name__ == '__main__':
    counter = 0
    for npy_file in glob.glob('house/*.npy'):
        counter += 1
        item = np.load(npy_file)
        npy_cutter(item)
        if counter % 128==0:
            print counter
            print datetime.datetime.now().time()