
#====================================================================================================================

import numpy as np
import glob

#====================================================================================================================

colors = [" 0 0 0 255       ", " 173 216 230 255 ", " 0 128 0 255     ", " 0 128 0 255     ",
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

def write_cost_and_accuracy(directory, train_cost, valid_cost, train_accu1, train_accu2, valid_accu1, valid_accu2):
    output = open(directory + "/costs.py", 'w')
    output.write("import matplotlib.pyplot as plt" + "\r\n")
    output.write("train_cost = []" + "\r\n")
    output.write("valid_cost = []" + "\r\n")
    output.write("steps      = []" + "\r\n")
    for i in range(len(train_cost)):
        output.write("steps.append(" + str(i) + ")" + "\r\n")
    for i in range(len(train_cost)):
        output.write("train_cost.append(" + str(train_cost[i]) + ")" + "\r\n")
    output.write("\r\n \r\n \r\n")
    for i in range(len(valid_cost)):
        for j in range(100):
            output.write("valid_cost.append(" + str(valid_cost[i]) + ")" + "\r\n")
    output.write("plt.plot( steps , train_cost, color ='b', lw=3 )     " + "\r\n")
    output.write("plt.plot( steps , valid_cost, color ='g', lw=3 )     " + "\r\n")
    output.write("plt.xlabel('Steps', fontsize=14)                     " + "\r\n")
    output.write("plt.ylabel('Cost',  fontsize=14)                     " + "\r\n")
    output.write("plt.suptitle('Blue: Train Cost, Green: Valid Cost')  " + "\r\n")
    output.write("plt.show()                                           " + "\r\n")
    print("costs.py file is created!")

    output = open(directory + "/accuracy.py", 'w')
    output.write("import matplotlib.pyplot as plt" + "\r\n")
    output.write("train_accu1 = []" + "\r\n")
    output.write("train_accu2 = []" + "\r\n")
    output.write("valid_accu1 = []" + "\r\n")
    output.write("valid_accu2 = []" + "\r\n")
    output.write("steps      = []" + "\r\n")
    for i in range(len(train_accu1)):
        output.write("steps.append(" + str(i) + ")" + "\r\n")
    output.write("\r\n \r\n \r\n")
    for i in range(len(train_accu1)):
        output.write("train_accu1.append(" + str(train_accu1[i]) + ")" + "\r\n")
    output.write("\r\n \r\n \r\n")
    for i in range(len(train_accu2)):
        output.write("train_accu2.append(" + str(train_accu2[i]) + ")" + "\r\n")
    output.write("\r\n \r\n \r\n")
    for i in range(len(valid_accu1)):
        for j in range(100):
            output.write("valid_accu1.append(" + str(valid_accu1[i]) + ")" + "\r\n")
    output.write("\r\n \r\n \r\n")
    for i in range(len(valid_accu2)):
        for j in range(100):
            output.write("valid_accu2.append(" + str(valid_accu2[i]) + ")" + "\r\n")
    output.write("plt.plot( steps , train_accu1, color ='b', lw=3 )   " + "\r\n")
    output.write("plt.plot( steps , train_accu2, color ='b', lw=1 )   " + "\r\n")
    output.write("plt.plot( steps , valid_accu1, color ='g', lw=3 )   " + "\r\n")
    output.write("plt.plot( steps , valid_accu2, color ='g', lw=1 )   " + "\r\n")
    output.write("plt.xlabel('Steps', fontsize=14)                   " + "\r\n")
    output.write("plt.ylabel('Accuracy',  fontsize=14)               " + "\r\n")
    output.write("plt.suptitle('Blue: Train Accu, Green: Valid Accu')" + "\r\n")
    output.write("plt.show()                                         " + "\r\n")
    print("accuracy.py file is created!")

#====================================================================================================================

def backup(directory, sess, saver, writer, train_cost, valid_cost, train_accu1, train_accu2, valid_accu1, valid_accu2):
    print("Saving the model...")
    saver.save(sess, directory + '/my-model')
    write_cost_and_accuracy(directory, train_cost, valid_cost, train_accu1, train_accu2, valid_accu1, valid_accu2)
    writer.close()

    # Visualize Validation Set
    print("Creating ply files...")

    colors = []
    colors.append(" 0 0 0 255  ")  # balck      for 0  'empty'
    colors.append(" 139 0 0 255")  # dark red   for 1  'ceiling'
    colors.append(" 0 128 0 255")  # green      for 2  'floor'
    colors.append(" 173 216 230 255")  # light blue for 3  'wall'
    colors.append(" 0 0 255 255")  # blue       for 4  'window'
    colors.append(" 255 0 0 255")  # red        for 5  'door'
    colors.append(" 218 165 32 255")  # goldenrod  for 6  'chair'
    colors.append(" 210 180 140 255")  # tan        for 7  'bed'
    colors.append(" 128 0   128 255")  # purple     for 8  'sofa'
    colors.append(" 0  0 139 255")  # dark blue  for 9  'table'
    colors.append(" 255 255 0 255")  # yellow     for 10 'coffe table'
    colors.append(" 128 128 128 255")  # gray       for 11 'shelves'
    colors.append(" 0 100 0 255")  # dark green for 12 ' '
    colors.append(" 255 165 0 255")  # orange     for 13 'furniture'

    flag = 0
    npy = '*.npy'
    npytest = '*.npytest'

    while True:
        counter = 0

        if flag == 0:
            files = npytest
        else:
            files = npy
            flag = 1

        for test in glob.glob(files):
            scene = np.load(test)
            trData, trLabel = [], []

            scene = np.load(test)
            temp = np.zeros((26, 30, 60))
            for dd in range(0, 60):
                temp[0:26, 0:30, dd] = scene[dd, 0:26, 0:30]
            scene = temp

            trData = scene[0:26, 0:30, 0:30]  # input
            trLabel = scene[0:26, 0:30, 30:60]  # gt

            trData = np.reshape(trData, (-1, 30 * 26 * 30))
            score = sess.run(CnnSE_class.score, feed_dict={x: trData, keep_prob: 1.0, phase: False})
            score = np.reshape(score, (26, 30, 30, 14))
            scn = np.full((26, 30, 100), -1, dtype=int)
            trData = np.reshape(trData, (26, 30, 30))

            scn[0:26, 0:30, 0:30] = trData  # add input data
            scn[0:26, 0:30, 33:63] = trLabel  # add ground truth

            for idx1 in range(0, 26):  # add predicted voxels
                for idx2 in range(0, 30):
                    for idx3 in range(66, 96):
                        maxIdx = np.argmax(score[idx1][idx2][idx3 - 66])
                        scn[idx1][idx2][idx3] = maxIdx

            output = open(directory + "/" + test + ".ply", 'w')
            ply = ""
            numOfVrtc = 0
            for idx1 in range(26):
                for idx2 in range(30):
                    for idx3 in range(100):
                        if scn[idx1][idx2][idx3] > 0:
                            ply = ply + str(idx1) + " " + str(idx2) + " " + str(idx3) + str(
                                colors[int(scn[idx1][idx2][idx3])]) + "\n"
                            numOfVrtc += 1

            output.write("ply" + "\n")
            output.write("format ascii 1.0" + "\n")
            output.write("comment VCGLIB generated" + "\n")
            output.write("element vertex " + str(numOfVrtc) + "\n")
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
            print(test + ".ply" + " is Done!")
            counter += 1

            if counter == 8:
                if flag == 1:
                    print
                    ".ply files are done!"
                    return
                else:
                    flag = 1
                    break
                    
#====================================================================================================================

def npy_to_ply(name, input_npy_file):  # the input is a npy file
    output_scene = input_npy_file
    output = open(str(name) + ".ply", 'w')
    ply = ""
    ver_num = 0
    for idx1 in range(output_scene.shape[0]):
        for idx2 in range(output_scene.shape[1]):
            for idx3 in range(output_scene.shape[2]):
                if output_scene[idx1][idx2][idx3] >= 1:
                    ply = ply + str(idx1) + " " + str(idx2) + " " + str(idx3) + str(
                        colors[int(output_scene[idx1][idx2][idx3])]) + "\n"
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

#====================================================================================================================

def npy_cutter(item):
    x, y, z = 84, 44, 84 
    scene = np.zeros((x, y, z))
    try:
        x_, y_, z_ = item.shape
    
        if   x<=x_ and y<=y_ and z<=z_: 
            scene           =item[:x, :y, :z] 
        elif x<=x_ and y>=y_ and z<=z_:
            scene[:, :y_, :]=item[:x, :, :z] 
        elif x<=x_ and y<=y_ and z>=z_:
            scene[:, :, :z_]=item[:x, :y, :] 
        elif x<=x_ and y>=y_ and z>=z_: 
            scene[:, :y_, :z_]=item[:x, :, :]  
        elif x>=x_ and y<=y_ and z<=z_:
            scene[:x_, :, :]=item[:, :y, :z] 
        elif x>=x_ and y>=y_ and z<=z_:
            scene[:x_, :y_, :]=item[:, :, :z] 
        elif x>=x_ and y<=y_ and z>=z_:
            scene[:x_, :, :z_]=item[:, :y, :] 
        elif x>=x_ and y>=y_ and z>=z_:
            scene[:x_, :y_, :z_]=item 
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
        npy_cutter(item) 
        
#====================================================================================================================

def load_time_test():
    counter = 0
    for npy_file in glob.glob('house/*.npy'):
        counter += 1
        item = np.load(npy_file)
        npy_cutter(item)
        if counter % 128==0:
            print counter
            print datetime.datetime.now().time()

#====================================================================================================================

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
        
#====================================================================================================================

def show_scene_size():
    counter = 0
    for npyFile in glob.glob('house/*.npy'): 
        dims = np.load(npyFile).shape
        if dims[0] < 84 or dims[1] < 44 or dims[2] < 84:
            counter += 1
        if counter % 1000 == 0:
            print counter
    print ("final count: ", counter)
    
#====================================================================================================================

if __name__ == '__main__':
    # load_time_test()
    # scene_load_and_visualize_test()
    show_scene_size()
    pass
