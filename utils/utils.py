
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
    output.write( "plt.plot( steps , valid_cost, color ='g', lw=1 )   " + "\r\n" )
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
        for j in range(100):
            output.write( "valid_accu1.append("+ str(valid_accu1[i]) +")" + "\r\n" )   
    output.write( "\r\n \r\n \r\n" )
    for i in range(len(valid_accu2)):
        for j in range(100):
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
    
    """ 91 classes
    1 empty 
    2 wall 
    3 ceiling 
    4 floor 
    5 unknown 
    6 hanger 
    7 kitchen_cabinet 
    8 kitchen_appliance 
    9 desk 
    10 chair 
    11 table 
    12 television 
    13 tv_stand 
    14 computer 
    15 wardrobe_cabinet 
    16 door 
    17 indoor_lamp 
    18 window 
    19 stand 
    20 dresser 
    21 plant 
    22 sink 
    23 table_and_chair 
    24 shower 
    25 toy 
    26 dressing_table 
    27 music 
    28 mirror 
    29 shoes_cabinet 
    30 books 
    31 kitchenware 
    32 toilet 
    33 stairs 
    34 rug 
    35 sofa 
    36 recreation 
    37 shelving 
    38 bed 
    39 household_appliance 
    40 air_conditioner 
    41 fan 
    42 bathroom_stuff 
    43 heater 
    44 picture_frame 
    45 fireplace 
    46 hanging_kitchen_cabinet 
    47 vase 
    48 curtain 
    49 switch 
    50 clock 
    51 trash_can 
    52 person 
    53 partition 
    54 gym_equipment 
    55 headstone 
    56 coffin 
    57 garage_door 
    58 decoration 
    59 vehicle 
    60 column 
    61 fence 
    62 outdoor_lamp 
    63 outdoor_seating 
    64 grill 
    65 bathtub 
    66 bench_chair 
    67 ottoman 
    68 workplace 
    69 whiteboard 
    70 candle 
    71 pool 
    72 ATM 
    73 pet 
    74 outdoor_cover 
    75 arch 
    76 roof 
    77 kitchen_set 
    78 wood_board 
    79 pillow 
    80 magazines 
    81 tripod 
    82 shoes 
    83 trinket 
    84 outdoor_spring 
    85 cloth 
    86 drinkbar 
    87 cart 
    88 safe 
    89 mailbox 
    90 storage_bench 
    91 stand
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

def reduce_classes_to_13():
    pass

#====================================================================================================================

if __name__ == '__main__':
    # load_time_test()
    # scene_load_and_visualize_test() 
    # show_scene_size()
    # npy_cutter_test()
    pass 
