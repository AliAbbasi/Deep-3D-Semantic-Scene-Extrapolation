
import sys, glob, datetime, random, os , os.path, shutil
import numpy            as     np
from   random           import randint
from   numpy            import array 
from   collections      import Counter
from   multiprocessing  import Pool 
import tensorflow       as     tf 

directory = 'v30-8'
if not os.path.exists(directory):
    os.makedirs(directory)
    
#  ---------------------------------------------------------------------------------------------------------------------

class CnnSE( object ): 
    def params(self):
        params_w = {
                    'w1'   : tf.Variable(tf.truncated_normal( [ 7 , 7 , 30  , 128   ], stddev = 0.01 )),
                    'w2'   : tf.Variable(tf.truncated_normal( [ 1 , 1 , 128 , 256   ], stddev = 0.01 )),  
                    'w3'   : tf.Variable(tf.truncated_normal( [ 3 , 3 , 256 , 128   ], stddev = 0.01 )),
                    'w4'   : tf.Variable(tf.truncated_normal( [ 1 , 1 , 128 , 128   ], stddev = 0.01 )),
                    'w5'   : tf.Variable(tf.truncated_normal( [ 1 , 1 , 128 , 256   ], stddev = 0.01 )),  
                    'w6'   : tf.Variable(tf.truncated_normal( [ 3 , 3 , 256 , 256   ], stddev = 0.01 )),  
                    'w7'   : tf.Variable(tf.truncated_normal( [ 1 , 1 , 256 , 128   ], stddev = 0.01 )),
                    'w8'   : tf.Variable(tf.truncated_normal( [ 1 , 1 , 128 , 256   ], stddev = 0.01 )),  
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

        #  ---------------------------------------------------------------------------------------------------------------------

    def score(self):
    
        def conv2d(x, w, b, name="conv_biased", strides=1):
            with tf.name_scope(name):
                x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
                x = tf.nn.bias_add(x, b)
                tf.summary.histogram("weights", w)
                tf.summary.histogram("biases",  b) 
                return x  

        #---------------------------------------------------------------------------------------------------------------------------------------------
        
        self.x_ = tf.reshape(x, shape = [-1, 26, 30, 30])
        
        for d in ['/gpu:0', '/gpu:1']:
            with tf.device(d):
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

    def cost(self):
    
        logits = tf.reshape(self.score, [-1, 14])
        labels = tf.reshape(self.y,     [-1    ]) 
        
        total  = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits( logits=logits, labels=labels ))

        for w in self.params_w_ :
            total += tf.nn.l2_loss(self.params_w_[w]) * 0.001 
            
        # smoothness term
        logits       = tf.reshape( self.score, [-1, 26, 30, 60, 14] )
        labels       = tf.reshape( self.y,     [-1, 26, 30, 60    ] )
        split_logits = tf.split( axis=3, num_or_size_splits=60, value=logits )
        split_labels = tf.split( axis=3, num_or_size_splits=60, value=labels )
        
        for i in range(0,len(split_logits)-1):
            total += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits( logits=split_logits[i], labels=split_labels[i+1] ))
            
        tf.summary.scalar("loss", total)  
        return total
        
    #------------------------------------------------------------------------------------------------------------------------------------------------    
    
    def update(self):
        return tf.train.AdamOptimizer(learning_rate = self.lr).minimize(self.cost)

    #--------------------------------------------------------------------------------------------------------------------------------------------------

    def __init__(self, x, y, lr, keep_prob, phase):                    
        self.x_        = x
        self.y         = y
        self.lr        = lr 
        self.keep_prob  = keep_prob
        self.phase     = phase 

        [self.params_w_, self.params_b_] = CnnSE.params(self)
        self.score                       = CnnSE.score(self)
        self.cost                        = CnnSE.cost(self)
        self.update                      = CnnSE.update(self)
        
#===================================================================================================================================================

                    
#===================================================================================================================================================

def accuFun(sess,trData,trLabel,batch_size):

    score   = sess.run( CnnSE_class.score , feed_dict={x: trData, keep_prob: 1.0, phase: False})  
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
                    
        totalAccuOveral   += (positiveOverall  / 23400.0        )    
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
    logPath       = '/tmp/' + directory
    x             = tf.placeholder(tf.float32, [ None, input ])
    y             = tf.placeholder(tf.int32  , [ None, out   ])   
    lr            = tf.placeholder(tf.float32                 )   
    keep_prob      = tf.placeholder(tf.float32                 )
    phase         = tf.placeholder(tf.bool                    )
    dropOut       = 0.5
    batch_size    = 100
    maxEpoch      = 50
    CnnSE_class = CnnSE(x,y,lr,keep_prob,phase)
    initVar       = tf.global_variables_initializer() 
    saver         = tf.train.Saver()
    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess: 
    
        sess.run(initVar)
        if os.path.exists(logPath):
            shutil.rmtree(logPath)
        writer = tf.summary.FileWriter(logPath, sess.graph)
        
        # restore model weights
        if os.path.exists(directory + '/my-model.meta'): 
            new_saver = tf.train.import_meta_graph(directory + '/my-model.meta')
            new_saver.restore(sess, tf.train.latest_checkpoint(directory)) 
            print "\r\n------------ Saved weights restored. ! ------------" 
        
        print("\r\n"+ str(datetime.datetime.now().time())[:-7] + " ------------------------------------------------------")   
        
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
            for axisX in range(30,35):  
            
                print "\r\n ----- This is the loop: " + str(axisX - 29) + " of 5 augmentation loops. -----" 
                saver.save(sess, directory + '/my-model') 
                print "\r\n Model saved! \r\n" 
                
                for npyFile in glob.glob('*.npy'): 
                    trData, trLabel = [], [] 
                    
                    if counter < batch_size:  
                        scene = np.load(npyFile)           # the default input is Z long channels
                        # convert input to X long channels 
                        temp  = np.zeros((26,30,75))       # create an numpy matrix with 15 extra zero channels
                        for z in range(0,60):
                            temp[ 0:26, 0:30, z+15 ] = scene[ z, 0:26, 0:30 ]
                        scene = temp 
                        
                        batch.append(scene)
                        counter += 1   
                    else: 
                        counter = 0  
                        for item in batch:
                            trData .append(item[ 0:26, 0:30, axisX-30:axisX    ]) # input 
                            trLabel.append(item[ 0:26, 0:30, axisX   :axisX+30 ]) # gt  

                        trData  = np.reshape( trData,  ( -1, 30 * 26 * 30 ))
                        trLabel = np.reshape( trLabel, ( -1, 26 * 30 * 30 )) 
                       
                        sess.run               ( CnnSE_class.update , feed_dict={x: trData, y: trLabel, lr: alr, keep_prob: dropOut, phase: True} )  
                        summary      = sess.run( CnnSE_class.sum    , feed_dict={x: trData, y: trLabel         , keep_prob: 1.0    , phase: True} )
                        cost         = sess.run( CnnSE_class.cost   , feed_dict={x: trData, y: trLabel         , keep_prob: 1.0    , phase: True} )
 
                        if step%10 == 0: 
                            print("%s , E:%g , S:%3g , lr:%g , accu1: %4.3g , accu2: %4.3g , Cost: %2.3g "% ( str(datetime.datetime.now().time())[:-7], epoch, step, alr, accu1tr, accu2tr, cost )) 
                        
                        train_cost.append(cost)
                        train_accu1.append(accu1tr )
                        train_accu2.append(accu2tr)
                        writer.add_summary(summary, step) 
                        step += 1  
                        
                        # -------------- validation -------------- 
                        
                        if step%100 == 0:   
                            accu1tr, accu2tr = accuFun ( sess, trData, trLabel, batch_size ) 
                            trData, trLabel, arrValid, batch = [], [], [], []
                            for npytest in glob.glob('*.npytest'):  
                            
                                scene = np.load(npytest) 
                                temp  = np.zeros((26,30,60))
                                for dd in range(0,60):
                                    temp[ 0:26, 0:30, dd ] = scene[ dd, 0:26, 0:30 ]
                                scene = temp
                                batch.append(scene)
                                
                            for item in batch:
                                trData .append(item[ 0:26, 0:30,  0:30 ]) # input 
                                trLabel.append(item[ 0:26, 0:30, 30:60 ]) # gt 
                                
                            trData  = np.reshape( trData,  ( -1, 30 * 26 * 30 ))
                            trLabel = np.reshape( trLabel, ( -1, 26 * 30 * 30 ))  
                            
                            cost         = sess.run( CnnSE_class.cost , feed_dict={x: trData, y: trLabel, keep_prob: 1.0, phase: True} )
                            accu1, accu2 = accuFun ( sess, trData, trLabel, 8 )
                            print("------->>>>> Validation >>>> accu1: %4.3g , accu2: %4.3g , Cost: %2.3g <<<<<<<<"% (accu1, accu2, cost))                             
                            valid_cost.append(cost)
                            valid_accu1.append(accu1)   
                            valid_accu2.append(accu2)   
                            
                        # -------------- write cost and accuracy --------------  
                        if step%1000 == 0: 
                            backup(sess, saver, writer, train_cost, valid_cost, train_accu1, train_accu2, valid_accu1, valid_accu2) 
                            print "axisX: " + str(axisX)   
                        
                        batch = []    
                        
                # END for binFile in glob  
            # END for axisX in range(30,60)
            # --------------------------------------------------------------
            
            epoch += 1    
            if epoch > maxEpoch:
                print " --- \r\n --- \r\n  Trainig process is done after " + str(maxEpoch) + " epochs and 5 augmentation loops.! \r\n --- \r\n ---" 
                sys.exit(0) 
        
#========================================================================================================================================================
 

