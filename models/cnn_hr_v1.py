
#====================================================================================================================================================

# 91 category of objects    
# scene size: 84 x 44 x 84     

#====================================================================================================================================================

import sys, glob, datetime, random, os, os.path, shutil, logging
import numpy            as     np
from   random           import randint
from   numpy            import array 
from   collections      import Counter
from   multiprocessing  import Pool 
import tensorflow       as     tf 
import utils # TODO fix it later

#====================================================================================================================================================

scene_shape = [84, 44, 84]
halfed_scene_shape = scene_shape[2] / 2

classes_count = 91

to_train   = True
to_restore = False

logging.basicConfig(filename='cnn_hr_v1.log',level=logging.DEBUG)
directory  = 'cnn_hr_v1'
if not os.path.exists(directory):
    os.makedirs(directory) 
    
#=====================================================================================================================================================

def writeCostNaccu(train_cost, valid_cost, train_accu1, train_accu2, valid_accu1, valid_accu2): 
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
    output.write( "plt.plot( steps , train_cost, color ='b', lw=3 )   " + "\r\n" )
    output.write( "plt.plot( steps , valid_cost, color ='g', lw=3 )   " + "\r\n" )
    output.write( "plt.xlabel('Steps', fontsize=14)                   " + "\r\n" )
    output.write( "plt.ylabel('Cost',  fontsize=14)                   " + "\r\n" )
    output.write( "plt.suptitle('Blue: Train Cost, Green: Valid Cost')" + "\r\n" )
    output.write( "plt.show()                                         " + "\r\n" ) 
    logging.info("costs.py file is created!")
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
    logging.info("accuracy.py file is created!")
    print ("accuracy.py file is created!")

#=====================================================================================================================================================

class ConvNet( object ):

    def paramsFun(self): 
        params_w = {
                    'w1'   : tf.Variable(tf.truncated_normal( [ 5 , 5 , halfed_scene_shape , 64               ], stddev = 0.01 )),  
                    'w2'   : tf.Variable(tf.truncated_normal( [ 3 , 3 , 64 , 64                               ], stddev = 0.01 )),  
                    'w3'   : tf.Variable(tf.truncated_normal( [ 3 , 3 , 64 , 64                               ], stddev = 0.01 )),
                    'w4'   : tf.Variable(tf.truncated_normal( [ 1 , 1 , 64 , 64                               ], stddev = 0.01 )),  
                    'w5'   : tf.Variable(tf.truncated_normal( [ 3 , 3 , 64 , 64                               ], stddev = 0.01 )),  
                    'w6'   : tf.Variable(tf.truncated_normal( [ 3 , 3 , 64 , 64                               ], stddev = 0.01 )),  
                    'w7'   : tf.Variable(tf.truncated_normal( [ 1 , 1 , 64 , 64                               ], stddev = 0.01 )),  
                    'w8'   : tf.Variable(tf.truncated_normal( [ 3 , 3 , 64 , 64                               ], stddev = 0.01 )),  
                    'w9'   : tf.Variable(tf.truncated_normal( [ 3 , 3 , 64 , 64                               ], stddev = 0.01 )),  
                    'w10'  : tf.Variable(tf.truncated_normal( [ 1 , 1 , 64 , 64                               ], stddev = 0.01 )),   
                    'w11'  : tf.Variable(tf.truncated_normal( [ 3 , 3 , 64 , 64                               ], stddev = 0.01 )),   
                    'w12'  : tf.Variable(tf.truncated_normal( [ 1 , 1 , 64 , 64                               ], stddev = 0.01 )),  
                    'w13'  : tf.Variable(tf.truncated_normal( [ 1 , 1 , 64 , classes_count*halfed_scene_shape ], stddev = 0.01 ))
                   } 
        params_b = {
                    'b1'   : tf.Variable(tf.truncated_normal( [ 64                               ], stddev = 0.01 )),  
                    'b2'   : tf.Variable(tf.truncated_normal( [ 64                               ], stddev = 0.01 )),  
                    'b3'   : tf.Variable(tf.truncated_normal( [ 64                               ], stddev = 0.01 )), 
                    'b4'   : tf.Variable(tf.truncated_normal( [ 64                               ], stddev = 0.01 )), 
                    'b5'   : tf.Variable(tf.truncated_normal( [ 64                               ], stddev = 0.01 )), 
                    'b6'   : tf.Variable(tf.truncated_normal( [ 64                               ], stddev = 0.01 )), 
                    'b7'   : tf.Variable(tf.truncated_normal( [ 64                               ], stddev = 0.01 )), 
                    'b8'   : tf.Variable(tf.truncated_normal( [ 64                               ], stddev = 0.01 )), 
                    'b9'   : tf.Variable(tf.truncated_normal( [ 64                               ], stddev = 0.01 )), 
                    'b10'  : tf.Variable(tf.truncated_normal( [ 64                               ], stddev = 0.01 )), 
                    'b11'  : tf.Variable(tf.truncated_normal( [ 64                               ], stddev = 0.01 )), 
                    'b12'  : tf.Variable(tf.truncated_normal( [ 64                               ], stddev = 0.01 )), 
                    'b13'  : tf.Variable(tf.truncated_normal( [ classes_count*halfed_scene_shape ], stddev = 0.01 ))
                   } 
                   
        return params_w,params_b

    #=================================================================================================================================================

    def scoreFun(self): 
    
        def conv2d(x, w, b, name="conv_biased", strides=1):
            with tf.name_scope(name):
                x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
                x = tf.nn.bias_add(x, b)
                tf.summary.histogram("weights", w)
                tf.summary.histogram("biases",  b) 
                return x  
                
        #---------------------------------------------------------------------------------------------------------------------------------------------
        
        def maxpool2d(x, k=2):
            return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
        
        #---------------------------------------------------------------------------------------------------------------------------------------------
        
        self.x_   = tf.reshape(x, shape = [-1, scene_shape[0], scene_shape[1], halfed_scene_shape]) 
        
        conv_1    = conv2d( self.x_, self.params_w_['w1'], self.params_b_['b1'], "conv_1" )
        
        # Residual Block #1
        conv_r1_1 = tf.nn.relu( conv_1 )  
        conv_r1_2 = tf.nn.relu( conv2d( conv_r1_1, self.params_w_['w2'], self.params_b_['b2'], "conv_r1_2" ) )   
        conv_r1_3 = tf.nn.relu( conv2d( conv_r1_2, self.params_w_['w3'], self.params_b_['b3'], "conv_r1_3" ) ) 
        conv_r1_4 =             conv2d( conv_r1_3, self.params_w_['w4'], self.params_b_['b4'], "conv_r1_4" ) 

        merge_1   = tf.add_n([conv_1, conv_r1_4])  
        
        # Residual Block #2
        conv_r2_1 = tf.nn.relu( merge_1 )  
        conv_r2_2 = tf.nn.relu( conv2d( conv_r2_1, self.params_w_['w5'], self.params_b_['b5'], "conv_r2_2" ) )   
        conv_r2_3 = tf.nn.relu( conv2d( conv_r2_2, self.params_w_['w6'], self.params_b_['b6'], "conv_r2_3" ) ) 
        conv_r2_4 =             conv2d( conv_r2_3, self.params_w_['w7'], self.params_b_['b7'], "conv_r2_4" ) 
        
        merge_2   = tf.add_n([merge_1, conv_r2_4])  
        
        # Residual Block #3
        conv_r3_1 = tf.nn.relu( merge_2 )  
        conv_r3_2 = tf.nn.relu( conv2d( conv_r3_1, self.params_w_['w8'],  self.params_b_['b8'],  "conv_r3_2" ) )   
        conv_r3_3 = tf.nn.relu( conv2d( conv_r3_2, self.params_w_['w9'],  self.params_b_['b9'],  "conv_r3_3" ) ) 
        conv_r3_4 =             conv2d( conv_r3_3, self.params_w_['w10'], self.params_b_['b10'], "conv_r3_4" )  
        
        merge_3   = tf.nn.relu( tf.add_n([merge_2, conv_r3_4]) ) 
        
        conv_2    = tf.nn.relu( conv2d( merge_3, self.params_w_['w11'], self.params_b_['b11'], "conv_2" ) )  
        conv_3    = tf.nn.relu( conv2d( conv_2,  self.params_w_['w12'], self.params_b_['b12'], "conv_3" ) )
        
        merge_4   = tf.nn.relu( tf.add_n([merge_3, conv_3]) ) 
        conv_4    =             conv2d( merge_4,  self.params_w_['w13'], self.params_b_['b13'], "conv_4" )  
        
        netOut    = tf.contrib.layers.flatten(conv_4)
        
        return netOut
        
    #---------------------------------------------------------------------------------------------------------------------------------------------------

    def costFun(self):  
    
        logits = tf.reshape(self.score, [-1, classes_count])
        labels = tf.reshape(self.y,     [-1               ]) 
        
        total  = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits( logits=logits, labels=labels ))
        
        for w in self.params_w_:
            total += tf.nn.l2_loss(self.params_w_[w]) * 0.001 
            
        # penalty term
        logits       = tf.reshape(self.score, [-1, scene_shape[0], scene_shape[1], halfed_scene_shape, classes_count])
        labels       = tf.reshape(self.y,     [-1, scene_shape[0], scene_shape[1], halfed_scene_shape               ])
        split_logits = tf.split(axis=3, num_or_size_splits=halfed_scene_shape, value=logits)
        split_labels = tf.split(axis=3, num_or_size_splits=halfed_scene_shape, value=labels)
        
        for i in range(1,len(split_logits)):
            total += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=split_logits[i], labels=split_labels[i-1]))
            
        tf.summary.scalar("loss", total)  
        return total
        
    #------------------------------------------------------------------------------------------------------------------------------------------------    
    
    def updateFun(self):
        return tf.train.AdamOptimizer(learning_rate = self.lr).minimize(self.cost)

    #------------------------------------------------------------------------------------------------------------------------------------------------
    
    def sumFun(self):
        return tf.summary.merge_all()
        
   #--------------------------------------------------------------------------------------------------------------------------------------------------

    def __init__(self,x,y,lr,keepProb,phase):                    
        self.x_        = x
        self.y         = y
        self.lr        = lr 
        self.keepProb  = keepProb
        self.phase     = phase 

        [self.params_w_, self.params_b_] = ConvNet.paramsFun(self) # initialization and packing the parameters
        self.score                       = ConvNet.scoreFun (self) # Computing the score function     
        self.cost                        = ConvNet.costFun  (self) # Computing the cost function 
        self.update                      = ConvNet.updateFun(self) # Computing the update function
        self.sum                         = ConvNet.sumFun   (self) # summary logger 4 TensorBoard
        
#===================================================================================================================================================

def backup(sess, saver, train_cost, valid_cost, train_accu1, train_accu2, valid_accu1, valid_accu2) :
    logging.info("Saving the model...") 
    print       ("Saving the model...") 
    saver.save(sess, directory + '/my-model')    
    writeCostNaccu(train_cost, valid_cost, train_accu1, train_accu2, valid_accu1, valid_accu2)    
    
    return  # TODO 
    
    # Visualize Validation Set 
    logging.info("Creating ply files...") 
    print       ("Creating ply files...") 
    
    flag    = 0
    files   = ''
    npy     = '*.npy'
    npytest = '*.npytest'
    
    while(True):
        counter = 0
        
        if flag == 0:
            files = npytest
        else:
            files = npy
            flag  = 1
            
        for test in glob.glob(files): 
            scene = np.load(test)  
            trData, trLabel = [], [] 
            
            scene = np.load(test) 

            trData  = scene[ 0:scene_shape[0] , 0:scene_shape[1] , 0 : halfed_scene_shape              ]  # input 
            trLabel = scene[ 0:scene_shape[0] , 0:scene_shape[1] , halfed_scene_shape : scene_shape[2] ]  # gt     
            
            trData  = np.reshape( trData,  ( -1, scene_shape[0] * scene_shape[1] * halfed_scene_shape ))  
            score   = sess.run( ConvNet_class.score , feed_dict={x: trData, keepProb: 1.0, phase: False})  
            score   = np.reshape( score, ( scene_shape[0], scene_shape[1], halfed_scene_shape, classes_count ))  
            score   = np.argmax ( score, 3)     
            score   = np.reshape( score, ( scene_shape[0], scene_shape[1], halfed_scene_shape ))
            score   = score[0:scene_shape[0], 0:scene_shape[1], 1:halfed_scene_shape]            
            trData  = np.reshape( trData, (scene_shape[0], scene_shape[1], halfed_scene_shape))
            
            scn     = np.concatenate(( trData , score ), axis=2 )
            
            output    = open( directory + "/" + test + ".ply", 'w') 
            ply       = ""
            numOfVrtc = 0
            for idx1 in range(scene_shape[0]):
                for idx2 in range(scene_shape[1]): 
                    for idx3 in range(scene_shape[2] - 1): 
                        if scn[idx1][idx2][idx3] > 0:  
                            ply = ply + str(idx1)+ " " +str(idx2)+ " " +str(idx3) + str(colors[ int(scn[idx1][idx2][idx3]) ]) + "\n" 
                            numOfVrtc += 1
                            
            output.write("ply"                                   + "\n")
            output.write("format ascii 1.0"                      + "\n")
            output.write("comment VCGLIB generated"              + "\n")
            output.write("element vertex " +  str(numOfVrtc)     + "\n")
            output.write("property float x"                      + "\n")
            output.write("property float y"                      + "\n")
            output.write("property float z"                      + "\n")
            output.write("property uchar red"                    + "\n")
            output.write("property uchar green"                  + "\n")
            output.write("property uchar blue"                   + "\n")
            output.write("property uchar alpha"                  + "\n")
            output.write("element face 0"                        + "\n")
            output.write("property list uchar int vertex_indices"+ "\n")
            output.write("end_header"                            + "\n")
            output.write( ply                                          ) 
            output.close()
            logging.info(test + ".ply" + " is Done!") 
            print (test + ".ply" + " is Done!") 
            counter += 1
            
            if counter == 8:
                if flag == 1:
                    logging.info(".ply files are done!")
                    print (".ply files are done!")
                    return
                else:
                    flag = 1
                    break
                    
#===================================================================================================================================================

def show_result(sess):

    # Visualize Validation Set
    logging.info("Creating ply files...")
    print ("Creating ply files...")
    
    bs = 0  
    trData, trLabel = [], [] 
    batch_arr = []
    for test in glob.glob('test/*.npy'):  
        batch_arr.append(npy_cutter(np.load(test)))
        bs += 1 
        
    batch_arr = np.reshape( batch_arr, ( bs, scene_shape[0], scene_shape[1], scene_shape[2] ))
    trData  = batch_arr[ :, 0:scene_shape[0], 0:scene_shape[1], 0:halfed_scene_shape ]               # input 
    trLabel = batch_arr[ :, 0:scene_shape[0], 0:scene_shape[1], halfed_scene_shape:scene_shape[2] ]  # gt     
    trData  = np.reshape(trData, (-1, scene_shape[0] * scene_shape[1] * halfed_scene_shape))  
    score   = sess.run(ConvNet_class.score , feed_dict={x: trData, keepProb: 1.0, phase: False}) 
    accu1, accu2 = accuFun(sess, trData, trLabel, bs)     
    logging.info("A1: %g, A2: %g" % (accu1, accu2))
    print       ("A1: %g, A2: %g" % (accu1, accu2))
    
    for test in glob.glob('test/*.npy'): 
        scene = npy_cutter(np.load(test))  
        trData, trLabel = [], []   

        trData  = scene[ 0:scene_shape[0] , 0:scene_shape[1] , 0:halfed_scene_shape ]               # input 
        trLabel = scene[ 0:scene_shape[0] , 0:scene_shape[1] , halfed_scene_shape:scene_shape[2] ]  # gt 
        
        trData  = np.reshape( trData, ( -1, scene_shape[0] * scene_shape[1] * halfed_scene_shape ))  
        score   = sess.run( ConvNet_class.score , feed_dict={x: trData, keepProb: 1.0, phase: False})  
        score   = np.reshape( score, ( scene_shape[0], scene_shape[1], halfed_scene_shape, classes_count ))  
        score   = np.argmax ( score, 3)     
        score   = np.reshape( score, ( scene_shape[0], scene_shape[1], halfed_scene_shape ))
        score   = score[0:scene_shape[0], 0:scene_shape[1], 1:halfed_scene_shape]            
        trData  = np.reshape( trData, (scene_shape[0], scene_shape[1], halfed_scene_shape))
        
        scn = np.concatenate(( trData, score ), axis=2 ) 
        
        output = open( directory + "/" + test[5:] + ".ply" , 'w') 
        ply       = ""
        numOfVrtc = 0
        for idx1 in range(scene_shape[0]):
            for idx2 in range(scene_shape[1]): 
                for idx3 in range(scene_shape[2] - 1):  
                    if scn[idx1][idx2][idx3] > 0:  
                        ply = ply + str(idx1)+ " " +str(idx2)+ " " +str(idx3) + str(utils.colors[ int(scn[idx1][idx2][idx3]) ]) + "\n" 
                        numOfVrtc += 1
                        
        output.write("ply"                                   + "\n")
        output.write("format ascii 1.0"                      + "\n")
        output.write("comment VCGLIB generated"              + "\n")
        output.write("element vertex " +  str(numOfVrtc)     + "\n")
        output.write("property float x"                      + "\n")
        output.write("property float y"                      + "\n")
        output.write("property float z"                      + "\n")
        output.write("property uchar red"                    + "\n")
        output.write("property uchar green"                  + "\n")
        output.write("property uchar blue"                   + "\n")
        output.write("property uchar alpha"                  + "\n")
        output.write("element face 0"                        + "\n")
        output.write("property list uchar int vertex_indices"+ "\n")
        output.write("end_header"                            + "\n")
        output.write( ply                                          ) 
        output.close()
        logging.info(test + ".ply" + " is Done!")
        print       (test + ".ply" + " is Done!") 
    
    logging.info("A1: %g, A2: %g" % (accu1, accu2))    
    print       ("A1: %g, A2: %g" % (accu1, accu2))   
    
#===================================================================================================================================================
  
def accuFun(sess, trData, trLabel, batch_size):

    score   = sess.run( ConvNet_class.score , feed_dict={x: trData, keepProb: 1.0, phase: False})  
    score   = np.reshape( score,   ( batch_size, scene_shape[0], scene_shape[1], halfed_scene_shape, classes_count ) )  
    trLabel = np.reshape( trLabel, ( batch_size, scene_shape[0], scene_shape[1], halfed_scene_shape ))   
    
    totalAccuOveral   = 0.0
    totalAccuOccupied = 0.0
    
    for idxBatch in range(0, batch_size): 
        positiveOverall  = 0.0
        positiveOccupied = 0.0 
        totalOccupied    = 0.0
        
        for idx2 in range(0, scene_shape[0]):
            for idx3 in range(0, scene_shape[1]):   
                for idx4 in range(0, halfed_scene_shape):   
                    maxIdxPred = np.argmax(score[idxBatch][idx2][idx3][idx4])  
                    
                    if maxIdxPred == trLabel[idxBatch][idx2][idx3][idx4]:
                        positiveOverall+= 1.0
                        if maxIdxPred > 0:
                            positiveOccupied += 1
                            
                    if trLabel[idxBatch][idx2][idx3][idx4] > 0:
                        totalOccupied+= 1    
                    
        totalAccuOveral += (positiveOverall / (scene_shape[0] * scene_shape[1] * halfed_scene_shape * 1.0))    
        if totalOccupied == 0:
            totalOccupied = (scene_shape[0] * scene_shape[1] * halfed_scene_shape * 1.0)
        totalAccuOccupied += (positiveOccupied / totalOccupied) 
        
    totalAccuOveral   =  totalAccuOveral   / (batch_size * 1.0)
    totalAccuOccupied =  totalAccuOccupied / (batch_size * 1.0)
    
    return totalAccuOveral, totalAccuOccupied

#===================================================================================================================================================

def npy_cutter(item):   # Should take 'scene_shape' as input argument then add this fun to utils
    scene = np.zeros((scene_shape[0], scene_shape[1], scene_shape[2]))
    try:
        x_, y_, z_ = item.shape
    
        if   scene_shape[0]<=x_ and scene_shape[1]<=y_ and scene_shape[2]<=z_: 
            scene           =item[:scene_shape[0], :scene_shape[1], :scene_shape[2]] 
        elif scene_shape[0]<=x_ and scene_shape[1]>=y_ and scene_shape[2]<=z_:
            scene[:, :y_, :]=item[:scene_shape[0], :, :scene_shape[2]] 
        elif scene_shape[0]<=x_ and scene_shape[1]<=y_ and scene_shape[2]>=z_:
            scene[:, :, :z_]=item[:scene_shape[0], :scene_shape[1], :] 
        elif scene_shape[0]<=x_ and scene_shape[1]>=y_ and scene_shape[2]>=z_: 
            scene[:, :y_, :z_]=item[:scene_shape[0], :, :]  
        elif scene_shape[0]>=x_ and scene_shape[1]<=y_ and scene_shape[2]<=z_:
            scene[:x_, :, :]=item[:, :scene_shape[1], :scene_shape[2]] 
        elif scene_shape[0]>=x_ and scene_shape[1]>=y_ and scene_shape[2]<=z_:
            scene[:x_, :y_, :]=item[:, :, :scene_shape[2]] 
        elif scene_shape[0]>=x_ and scene_shape[1]<=y_ and scene_shape[2]>=z_:
            scene[:x_, :, :z_]=item[:, :scene_shape[1], :] 
        elif scene_shape[0]>=x_ and scene_shape[1]>=y_ and scene_shape[2]>=z_:
            scene[:x_, :y_, :z_]=item 
        else: 
            pass 
    except: 
        pass
        
    # TODO: convert to X or Z align
    # TODO: shift to the middle 
    return scene
        
#===================================================================================================================================================

if __name__ == '__main__':

    input         = scene_shape[0] * scene_shape[1] * halfed_scene_shape
    out           = scene_shape[0] * scene_shape[1] * halfed_scene_shape 
    logPath       = '/tmp/' + directory
    x             = tf.placeholder(tf.float32, [ None, input ])
    y             = tf.placeholder(tf.int32  , [ None, out   ])   
    lr            = tf.placeholder(tf.float32                 )   
    keepProb      = tf.placeholder(tf.float32                 )
    phase         = tf.placeholder(tf.bool                    )
    dropOut       = 0.5
    batch_size    = 32
    maxEpoch      = 50
    ConvNet_class = ConvNet(x, y, lr, keepProb, phase)
    init_var      = tf.global_variables_initializer() 
    saver         = tf.train.Saver()
    
    # prevent to add extra node to graph during training
    if to_train:
        tf.get_default_graph().finalize()
    # log_device_placement: shows the log of which task will work on which device.
    # allow_soft_placement: TF choose automatically the available device
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:  
        sess.run(init_var)
        
        if os.path.exists(logPath):
            shutil.rmtree(logPath) 
        
        # restore model weights
        if to_restore:
            if os.path.exists(directory + '/my-model.meta'): 
                new_saver = tf.train.import_meta_graph(directory + '/my-model.meta')
                new_saver.restore(sess, tf.train.latest_checkpoint(directory)) 
                logging.info("\r\n------------ Saved weights restored. ------------")
                print       ("\r\n------------ Saved weights restored. ------------")
                
        # -------------- test phase --------------
        if to_train == False:  
            show_result(sess) 
            logging.info(".ply files are created.!")
            print (".ply files are created.!")
            sys.exit(0)
        
        # -------------- train phase --------------
        step         = 0
        counter      = 0
        epoch        = 1 
        alr          = 0.00001
        train_cost   = []
        valid_cost   = []
        train_accu1  = []
        train_accu2  = []
        valid_accu1  = [] 
        valid_accu2  = [] 
        batch        = []  
        
        accu1tr, accu2tr = 0, 0
        
        while(epoch < maxEpoch):    
            saver.save(sess, directory + '/my-model') 
            logging.info("\r\n Model saved! \r\n") 
            print ("\r\n Model saved! \r\n") 
            
            for npyFile in glob.glob('house/*.npy'): 
                trData, trLabel = [], [] 
                
                if counter < batch_size:  
                    scene = npy_cutter(np.load(npyFile)) 
                    batch.append(scene)
                    counter += 1   
                else: 
                    counter = 0  
                    batch = np.reshape( batch, ( -1, scene_shape[0], scene_shape[1], scene_shape[2] ))   
                
                    trData  = batch[ : , 0:scene_shape[0] , 0:scene_shape[1], 0:halfed_scene_shape ]               # input 
                    trLabel = batch[ : , 0:scene_shape[0] , 0:scene_shape[1], halfed_scene_shape:scene_shape[2] ]  # gt  

                    trData  = np.reshape( trData,  ( -1, scene_shape[0] * scene_shape[1] * halfed_scene_shape ))
                    trLabel = np.reshape( trLabel, ( -1, scene_shape[0] * scene_shape[1] * halfed_scene_shape )) 
                   
                    sess.run(ConvNet_class.update, feed_dict={x: trData, y: trLabel, lr: alr, keepProb: dropOut, phase: True})   
                    cost = sess.run(ConvNet_class.cost, feed_dict={x: trData, y: trLabel, keepProb: 1.0, phase: True}) 
                    train_cost.append(cost)

                    if step%1 == 0: 
                        logging.info("%s , E:%g , S:%3g , lr:%g , accu1: %4.3g , accu2: %4.3g , Cost: %2.3g "% ( str(datetime.datetime.now().time())[:-7], epoch, step, alr, accu1tr, accu2tr, cost ))
                        print       ("%s , E:%g , S:%3g , lr:%g , accu1: %4.3g , accu2: %4.3g , Cost: %2.3g "% ( str(datetime.datetime.now().time())[:-7], epoch, step, alr, accu1tr, accu2tr, cost ))
                    # -------------- accuracy calculator --------------  
                    if step%500 == 0:   
                        accu1tr, accu2tr = accuFun(sess, trData, trLabel, batch_size)  
                        train_accu1.append(accu1tr)
                        train_accu2.append(accu2tr)
                        
                    # -------------- write cost and accuracy --------------  
                    if step%1000 == 0: 
                        backup(sess, saver, train_cost, valid_cost, train_accu1, train_accu2, valid_accu1, valid_accu2)  
                    
                    step += 1  
                    batch = []    
                    
            # END for binFile in glob 
            epoch += 1     
            logging.info(" --- \r\n --- \r\n  The Epoch: " + str(epoch) + " is Started. \r\n --- \r\n ---") 
            print       (" --- \r\n --- \r\n  The Epoch: " + str(epoch) + " is Started. \r\n --- \r\n ---") 
        logging.info(" --- \r\n --- \r\n  Trainig process is done after " + str(maxEpoch) + " epochs. \r\n --- \r\n ---")
        print       (" --- \r\n --- \r\n  Trainig process is done after " + str(maxEpoch) + " epochs. \r\n --- \r\n ---")
        
#========================================================================================================================================================
 

