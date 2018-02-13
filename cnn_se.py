
#====================================================================================================================================================

import sys, glob, datetime, random, os , os.path, shutil
import numpy            as     np
from   random           import randint
from   numpy            import array 
from   collections      import Counter
from   multiprocessing  import Pool 
import tensorflow       as     tf 

#====================================================================================================================================================

to_train   = False
to_restore = True

directory  = 'cnn_se_directory'

argv_list = str(sys.argv) 
if 'SUN' in argv_list:
    data_directory  = 'data/SUN' 
elif 'NYU' in argv_list:
    data_directory  = 'data/NYU' 
else:
    print("Invalid Arguments.!")
    sys.exit(0)
    
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

#=====================================================================================================================================================

class ConvNet( object ):

    def paramsFun(self): 
        params_w = {
                    'w1'   : tf.Variable(tf.truncated_normal( [ 5 , 5 , 30  , 128   ], stddev = 0.01 )), 
                    
                    'w2'   : tf.Variable(tf.truncated_normal( [ 3 , 3 , 128 , 256   ], stddev = 0.01 )),  
                    'w3'   : tf.Variable(tf.truncated_normal( [ 3 , 3 , 256 , 128   ], stddev = 0.01 )),
                    'w4'   : tf.Variable(tf.truncated_normal( [ 1 , 1 , 128 , 128   ], stddev = 0.01 )), 
                    
                    'w5'   : tf.Variable(tf.truncated_normal( [ 3 , 3 , 128 , 256   ], stddev = 0.01 )),  
                    'w6'   : tf.Variable(tf.truncated_normal( [ 3 , 3 , 256 , 256   ], stddev = 0.01 )),  
                    'w7'   : tf.Variable(tf.truncated_normal( [ 1 , 1 , 256 , 128   ], stddev = 0.01 )),  
                    
                    'w8'   : tf.Variable(tf.truncated_normal( [ 3 , 3 , 128 , 256   ], stddev = 0.01 )),  
                    'w9'   : tf.Variable(tf.truncated_normal( [ 3 , 3 , 256 , 256   ], stddev = 0.01 )),  
                    'w10'  : tf.Variable(tf.truncated_normal( [ 1 , 1 , 256 , 128   ], stddev = 0.01 )),  
                    
                    'w11'  : tf.Variable(tf.truncated_normal( [ 3 , 3 , 128 , 256   ], stddev = 0.01 )),   
                    'w12'  : tf.Variable(tf.truncated_normal( [ 1 , 1 , 256 , 128   ], stddev = 0.01 )),  
                    'w13'  : tf.Variable(tf.truncated_normal( [ 1 , 1 , 128 , 14*30 ], stddev = 0.01 ))
                   }
                    
        params_b = {
                    'b1'   : tf.Variable(tf.truncated_normal( [ 128   ], stddev = 0.01 )),  
                    'b2'   : tf.Variable(tf.truncated_normal( [ 256   ], stddev = 0.01 )),  
                    'b3'   : tf.Variable(tf.truncated_normal( [ 128   ], stddev = 0.01 )), 
                    'b4'   : tf.Variable(tf.truncated_normal( [ 128   ], stddev = 0.01 )), 
                    'b5'   : tf.Variable(tf.truncated_normal( [ 256   ], stddev = 0.01 )), 
                    'b6'   : tf.Variable(tf.truncated_normal( [ 256   ], stddev = 0.01 )), 
                    'b7'   : tf.Variable(tf.truncated_normal( [ 128   ], stddev = 0.01 )), 
                    'b8'   : tf.Variable(tf.truncated_normal( [ 256   ], stddev = 0.01 )), 
                    'b9'   : tf.Variable(tf.truncated_normal( [ 256   ], stddev = 0.01 )), 
                    'b10'  : tf.Variable(tf.truncated_normal( [ 128   ], stddev = 0.01 )), 
                    'b11'  : tf.Variable(tf.truncated_normal( [ 256   ], stddev = 0.01 )), 
                    'b12'  : tf.Variable(tf.truncated_normal( [ 128   ], stddev = 0.01 )), 
                    'b13'  : tf.Variable(tf.truncated_normal( [ 14*30 ], stddev = 0.01 ))
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
        
        self.x_   = tf.reshape(x, shape = [-1, 26, 30, 30]) 
        
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
    
        logits = tf.reshape(self.score, [-1, 14])
        labels = tf.reshape(self.y,     [-1    ]) 
        
        total  = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits( logits=logits, labels=labels ))

        for w in self.params_w_ :
            total += tf.nn.l2_loss(self.params_w_[w]) * 0.001 
            
        # penalty term
        logits       = tf.reshape( self.score, [-1, 26, 30, 30, 14] )
        labels       = tf.reshape( self.y,     [-1, 26, 30, 30    ] )
        split_logits = tf.split( axis=3, num_or_size_splits=30, value=logits )
        split_labels = tf.split( axis=3, num_or_size_splits=30, value=labels )
        
        for i in range(1,len(split_logits)):
            total += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits( logits=split_logits[i], labels=split_labels[i-1] ))
            
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
    print ("Saving the model..." )
    saver.save(sess, directory + '/my-model')   
    writeCostNaccu(train_cost, valid_cost, train_accu1, train_accu2, valid_accu1, valid_accu2)   
    
    # Visualize Validation Set ---------------------------------
    print ("Creating ply files..."  )
    
    colors  = []  
    colors.append(" 0 0 0 255  ")     # balck      for 0  'empty'
    colors.append(" 139 0 0 255")     # dark red   for 1  'ceiling'
    colors.append(" 0 128 0 255")     # green      for 2  'floor'
    colors.append(" 173 216 230 255") # light blue for 3  'wall'
    colors.append(" 0 0 255 255")     # blue       for 4  'window'
    colors.append(" 255 0 0 255")     # red        for 5  'door'
    colors.append(" 218 165 32 255")  # goldenrod  for 6  'chair'
    colors.append(" 210 180 140 255") # tan        for 7  'bed'
    colors.append(" 128 0   128 255") # purple     for 8  'sofa'
    colors.append(" 0  0 139 255")    # dark blue  for 9  'table'
    colors.append(" 255 255 0 255")   # yellow     for 10 'coffe table'
    colors.append(" 128 128 128 255") # gray       for 11 'shelves'
    colors.append(" 0 100 0 255")     # dark green for 12 ' '
    colors.append(" 255 165 0 255")   # orange     for 13 'furniture'  
 
    
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
            scene = scene.transpose(1, 2, 0) 

            trData = scene[ 0:26 , 0:30 ,  0:30 ]  # input 
            trLabel= scene[ 0:26 , 0:30 , 30:60 ]  # gt     
            
            trData  = np.reshape( trData,  ( -1, 30 * 26 * 30 ))  
            score   = sess.run( ConvNet_class.score , feed_dict={x: trData, keepProb: 1.0, phase: False})  
            score   = np.reshape( score, ( 26, 30, 30, 14 ))  
            score   = np.argmax ( score, 3)     
            score   = np.reshape( score, ( 26, 30, 30 ))
            score   = score[0:26, 0:30, 1:30]            
            trData  = np.reshape( trData, (26,30,30))
            
            scn     = np.concatenate(( trData , score ), axis=2 )
            
            output = open( directory + "/" + test + ".ply" , 'w') 
            ply       = ""
            numOfVrtc = 0
            for idx1 in range(26):
                for idx2 in range(30): 
                    for idx3 in range(59):  
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
            print (test + ".ply" + " is Done!" )
            counter += 1
            
            if counter == 8:
                if flag == 1:
                    print (".ply files are done!")
                    return
                else:
                    flag = 1
                    break
                    
#===================================================================================================================================================

def show_result(sess) : 

    # Visualize Validation Set ---------------------------------
    print("The model is running ...")
    
    colors  = []  
    colors.append(" 0 0 0 255  ")     # black      for 0  'empty'
    colors.append(" 139 0 0 255")     # dark red   for 1  'ceiling'
    colors.append(" 0 128 0 255")     # green      for 2  'floor'
    colors.append(" 173 216 230 255") # light blue for 3  'wall'
    colors.append(" 0 0 255 255")     # blue       for 4  'window'
    colors.append(" 255 0 0 255")     # red        for 5  'door'
    colors.append(" 218 165 32 255")  # goldenrod  for 6  'chair'
    colors.append(" 210 180 140 255") # tan        for 7  'bed'
    colors.append(" 128 0   128 255") # purple     for 8  'sofa'
    colors.append(" 0  0 139 255")    # dark blue  for 9  'table'
    colors.append(" 255 255 0 255")   # yellow     for 10 'coffee table'
    colors.append(" 128 128 128 255") # gray       for 11 'shelves'
    colors.append(" 0 100 0 255")     # dark green for 12 'cabinets'
    colors.append(" 255 165 0 255")   # orange     for 13 'furniture'  
 
    bs               = 0  
    trData, trLabel  = [], [] 
    batch_arr = []
    for test in glob.glob(data_directory + '/*.npy'): 
        scene = np.load(test) 
        scene = scene.transpose(1, 2, 0)
        batch_arr.append( scene )
        bs += 1
    
    batch_arr = np.reshape( batch_arr, ( bs, 26, 30, 60 ))
    trData  = batch_arr[ : , 0:26 , 0:30 ,  0:30 ]  # input 
    trLabel = batch_arr[ : , 0:26 , 0:30 , 30:60 ]  # gt     
    trData  = np.reshape( trData,  ( -1, 30 * 26 * 30 ))  
    score   = sess.run( ConvNet_class.score , feed_dict={x: trData, keepProb: 1.0, phase: False}) 
    accu1, accu2 = accuFun ( sess, trData, trLabel, bs )     
    print ("A1: ", accu1, " A2:", accu2)
    
    print("Creating .ply files ...")
    for test in glob.glob(data_directory + '/*.npy'): 
        scene = np.load(test)  
        trData, trLabel = [], [] 
        
        scene = np.load(test) 
        scene = scene.transpose(1, 2, 0)

        trData = scene[ 0:26 , 0:30 ,  0:30 ]  # input 
        trLabel= scene[ 0:26 , 0:30 , 30:60 ]  # gt     
        
        trData  = np.reshape( trData,  ( -1, 30 * 26 * 30 ))  
        score   = sess.run( ConvNet_class.score , feed_dict={x: trData, keepProb: 1.0, phase: False})  
        score   = np.reshape( score, ( 26, 30, 30, 14 ))  
        score   = np.argmax ( score, 3)     
        score   = np.reshape( score, ( 26, 30, 30 ))
        score   = score[0:26, 0:30, 1:30]            
        trData  = np.reshape( trData, (26,30,30))
        
        scn     = np.concatenate(( trData , score ), axis=2 )
        
        output = open( directory + "/" + test[9:] + ".ply" , 'w') 
        ply       = ""
        numOfVrtc = 0
        for idx1 in range(26):
            for idx2 in range(30): 
                for idx3 in range(59):  
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
        # print(test + ".ply" + " is Done!")
    print(".ply files are created.!")
#===================================================================================================================================================
  
def accuFun(sess,trData,trLabel,batch_size):

    score   = sess.run( ConvNet_class.score , feed_dict={x: trData, keepProb: 1.0, phase: False})  
    score   = np.reshape( score,   ( batch_size, 26 , 30 , 30 , 14 ) )  
    trLabel = np.reshape( trLabel, ( batch_size, 26 , 30 , 30      ) )   
    
    totalAccuOveral   = 0.0
    totalAccuOccupied = 0.0
    
    for idxBatch in range(0,batch_size): 
        positiveOverall  = 0.0
        positiveOccupied = 0.0 
        totalOccupied    = 0.0
        
        for idx2 in range(0,26):
            for idx3 in range(0,30):   
                for idx4 in range(0,30):   
                    maxIdxPred = np.argmax( score  [idxBatch][idx2][idx3][idx4] )  
                    
                    if maxIdxPred == trLabel[idxBatch][idx2][idx3][idx4]:
                        positiveOverall+= 1.0
                        if maxIdxPred > 0:
                            positiveOccupied += 1
                            
                    if trLabel[idxBatch][idx2][idx3][idx4] > 0:
                        totalOccupied+= 1    
                    
        totalAccuOveral   += (positiveOverall  / 23400.0)    
        if totalOccupied == 0:
            totalOccupied = 23400
        totalAccuOccupied += (positiveOccupied / totalOccupied) 
        
    totalAccuOveral   =  totalAccuOveral   / (batch_size * 1.0)
    totalAccuOccupied =  totalAccuOccupied / (batch_size * 1.0)
    
    return totalAccuOveral, totalAccuOccupied

#===================================================================================================================================================

if __name__ == '__main__':

    input         = 30 * 26 * 30
    out           = 26 * 30 * 30  
    # logPath       = '/tmp/' + directory
    x             = tf.placeholder(tf.float32, [ None, input ])
    y             = tf.placeholder(tf.int32  , [ None, out   ])   
    lr            = tf.placeholder(tf.float32                 )   
    keepProb      = tf.placeholder(tf.float32                 )
    phase         = tf.placeholder(tf.bool                    )
    dropOut       = 0.5
    batch_size    = 100
    maxEpoch      = 20
    ConvNet_class = ConvNet(x,y,lr,keepProb,phase)
    initVar       = tf.global_variables_initializer() 
    saver         = tf.train.Saver()
    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess: 
    
        sess.run(initVar)
        # if os.path.exists(logPath):
            # shutil.rmtree(logPath)
        # writer = tf.summary.FileWriter(logPath, sess.graph)
        
        # restore model weights
        if to_restore:
            if os.path.exists(directory + '/my-model.meta'): 
                new_saver = tf.train.import_meta_graph(directory + '/my-model.meta')
                new_saver.restore(sess, tf.train.latest_checkpoint(directory)) 
                print ("\r\n------------ Saved weights restored. ! ------------" )
        
        print("\r\n"+ str(datetime.datetime.now().time())[:-7] + " ------------------------------------------------------")   
        
        # -------------- test phase --------------
        if to_train == False:  
            show_result(sess) 
            print (".ply files are created.!")
            sys.exit(0)
        
        # -------------- train phase --------------
        step         = 0
        counter      = 0
        epoch        = 1 
        alr          = 0.00005 
        train_cost   = []
        valid_cost   = []
        train_accu1  = []
        train_accu2  = []
        valid_accu1  = [] 
        valid_accu2  = [] 
        batch        = []  
        
        accu1tr, accu2tr = 0, 0
        
        while(True):  
            for axisX in range(30,40):    # augmentation loop
            
                print ("\r\n ----- This is the loop: " + str(axisX - 29) + " of 10 augmentation loops. -----" )
                saver.save(sess, directory + '/my-model') 
                print ("\r\n Model saved! \r\n" )
                
                for npyFile in glob.glob('*.npy'): 
                    trData, trLabel = [], [] 
                    
                    if counter < batch_size:  
                        scene = np.load(npyFile)      
                        temp  = np.zeros((75,26,30))  
                        temp[15:75,:,:] = scene
                        batch.append( temp )
                        counter += 1   
                    else: 
                        counter = 0  
                        batch = np.reshape( batch, ( -1, 75, 26, 30 ))    
                        batch = batch.transpose(0,2,3,1)                        # transpose to 26x30x60
                    
                        trData  = batch[ : , 0:26, 0:30, axisX-30:axisX    ]  # input 
                        trLabel = batch[ : , 0:26, 0:30, axisX   :axisX+30 ]  # gt  

                        trData  = np.reshape( trData,  ( -1, 30 * 26 * 30 ))
                        trLabel = np.reshape( trLabel, ( -1, 26 * 30 * 30 )) 
                       
                        sess.run               ( ConvNet_class.update , feed_dict={x: trData, y: trLabel, lr: alr, keepProb: dropOut, phase: True} )  
                        # summary      = sess.run( ConvNet_class.sum    , feed_dict={x: trData, y: trLabel         , keepProb: 1.0    , phase: True} )
                        cost         = sess.run( ConvNet_class.cost   , feed_dict={x: trData, y: trLabel         , keepProb: 1.0    , phase: True} )
 
                        if step%10 == 0: 
                            print("%s , E:%g , S:%3g , lr:%g , accu1: %4.3g , accu2: %4.3g , Cost: %2.3g "% ( str(datetime.datetime.now().time())[:-7], epoch, step, alr, accu1tr, accu2tr, cost )) 
                        
                        # train_cost.append(cost)
                        # train_accu1.append(accu1tr )
                        # train_accu2.append(accu2tr)
                        # writer.add_summary(summary, step) 
                        
                        # -------------- validation --------------  
                        if step%500 == 0:   
                            accu1tr, accu2tr = accuFun ( sess, trData, trLabel, batch_size )  
                            
                        # -------------- write cost and accuracy --------------  
                        if step%1000 == 0: 
                            backup(sess, saver, train_cost, valid_cost, train_accu1, train_accu2, valid_accu1, valid_accu2) 
                            print ("axisX: " + str(axisX)   )
                        
                        step += 1  
                        batch = []    
                        
                # END for binFile in glob  
            # END for axisX in range(30,60)
            # --------------------------------------------------------------
            
            epoch += 1    
            if epoch > maxEpoch:
                print (" --- \r\n --- \r\n  Trainig process is done after " + str(maxEpoch) + " epochs and 10 augmentation loops in each epoch.! \r\n --- \r\n ---" )
                sys.exit(0) 
        
#========================================================================================================================================================
 

