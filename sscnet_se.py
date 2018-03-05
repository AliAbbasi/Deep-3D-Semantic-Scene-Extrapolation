# ====================================================================================================================

import datetime
import glob
import os.path
import sys
import numpy as np
import tensorflow as tf

# ====================================================================================================================

to_train = False
to_restore = True
directory = 'sscnet_se_directory'

argv_list = str(sys.argv)
if 'SUN' in argv_list:
    data_directory = 'data/SUN'
elif 'NYU' in argv_list:
    data_directory = 'data/NYU'
else:
    print("Invalid Arguments.!")
    sys.exit(0)
    
if not os.path.exists(directory):
    os.makedirs(directory)

# ====================================================================================================================

def writeCostNaccu(train_cost, valid_cost, train_accu1, train_accu2, valid_accu1, valid_accu2):
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
    output.write("plt.plot( steps , train_cost, color ='b', lw=3 )   " + "\r\n")
    output.write("plt.plot( steps , valid_cost, color ='g', lw=3 )   " + "\r\n")
    output.write("plt.xlabel('Steps', fontsize=14)                   " + "\r\n")
    output.write("plt.ylabel('Cost',  fontsize=14)                   " + "\r\n")
    output.write("plt.suptitle('Blue: Train Cost, Green: Valid Cost')" + "\r\n")
    output.write("plt.show()                                         " + "\r\n")
    print("costs.py file is created!")

    # -----------------------------------------------------------------------------

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

# ====================================================================================================================

class ConvNet(object):

    def paramsFun(self):
        params_w = {
            'w1':    tf.Variable(tf.truncated_normal([7, 7, 7, 1,   8], stddev=0.01)),
            'w2':    tf.Variable(tf.truncated_normal([3, 3, 3, 8,  16], stddev=0.01)),
            'w3':    tf.Variable(tf.truncated_normal([3, 3, 3, 16, 16], stddev=0.01)),
            'wRes1': tf.Variable(tf.truncated_normal([1, 1, 1, 8,  16], stddev=0.01)),
            'w4':    tf.Variable(tf.truncated_normal([3, 3, 3, 16, 32], stddev=0.01)),
            'w5':    tf.Variable(tf.truncated_normal([3, 3, 3, 32, 32], stddev=0.01)),
            'wRes2': tf.Variable(tf.truncated_normal([1, 1, 1, 32, 32], stddev=0.01)),
            'w6':    tf.Variable(tf.truncated_normal([3, 3, 3, 32, 32], stddev=0.01)),
            'w7':    tf.Variable(tf.truncated_normal([3, 3, 3, 32, 32], stddev=0.01)),
            'w8':    tf.Variable(tf.truncated_normal([3, 3, 3, 32, 32], stddev=0.01)),
            'w9':    tf.Variable(tf.truncated_normal([3, 3, 3, 32, 32], stddev=0.01)),
            'w10':   tf.Variable(tf.truncated_normal([3, 3, 3, 32, 32], stddev=0.01)),
            'w11':   tf.Variable(tf.truncated_normal([3, 3, 3, 32, 32], stddev=0.01)),
            'w12':   tf.Variable(tf.truncated_normal([1, 1, 1, 96, 32], stddev=0.01)),
            'w13':   tf.Variable(tf.truncated_normal([1, 1, 1, 32, 32], stddev=0.01)),
            'wOut':  tf.Variable(tf.truncated_normal([1, 1, 1, 32, 14], stddev=0.01))
        }

        params_b = {
            'b1':    tf.Variable(tf.truncated_normal([ 8], stddev=0.01)),
            'b2':    tf.Variable(tf.truncated_normal([16], stddev=0.01)),
            'b3':    tf.Variable(tf.truncated_normal([16], stddev=0.01)),
            'bRes1': tf.Variable(tf.truncated_normal([16], stddev=0.01)),
            'b4':    tf.Variable(tf.truncated_normal([32], stddev=0.01)),
            'b5':    tf.Variable(tf.truncated_normal([32], stddev=0.01)),
            'bRes2': tf.Variable(tf.truncated_normal([32], stddev=0.01)),
            'b6':    tf.Variable(tf.truncated_normal([32], stddev=0.01)),
            'b7':    tf.Variable(tf.truncated_normal([32], stddev=0.01)),
            'b8':    tf.Variable(tf.truncated_normal([32], stddev=0.01)),
            'b9':    tf.Variable(tf.truncated_normal([32], stddev=0.01)),
            'b10':   tf.Variable(tf.truncated_normal([32], stddev=0.01)),
            'b11':   tf.Variable(tf.truncated_normal([32], stddev=0.01)),
            'b12':   tf.Variable(tf.truncated_normal([32], stddev=0.01)),
            'b13':   tf.Variable(tf.truncated_normal([32], stddev=0.01)),
            'bOut':  tf.Variable(tf.truncated_normal([14], stddev=0.01))
        }

        return params_w, params_b

    # ================================================================================================================

    def scoreFun(self):
        def conv3d(x, w, b, strides=1):
            x = tf.nn.conv3d(x, w, strides=[1, strides, strides, strides, 1], padding='SAME')
            x = tf.nn.bias_add(x, b)
            return x

            # ------------------------------------------------------------------------------------------------------------

        def conv3d_transpose(x, w, b, output_shape, strides=2):
            x = tf.nn.conv3d_transpose(x, w, output_shape=output_shape,
                                       strides=[1, strides, strides, strides, 1], padding='SAME')
            x = tf.nn.bias_add(x, b)
            return x

            # ------------------------------------------------------------------------------------------------------------

        def d_conv3d(x, w, b, d_rate=1, stride=1):
            x = tf.nn.convolution(x, w, padding='SAME', strides=[stride, stride, stride],
                                  dilation_rate=[d_rate, d_rate, d_rate])
            x = tf.nn.bias_add(x, b)
            return x

        # ------------------------------------------------------------------------------------------------------------

        def maxpool3d(x, k=2):
            return tf.nn.avg_pool3d(x, ksize=[1, k, k, k, 1], strides=[1, k, k, k, 1], padding='SAME')

        # ------------------------------------------------------------------------------------------------------------

        self.x_ = tf.reshape(x, shape=[-1, 26, 30, 30, 1])

        conv_1 = conv3d(self.x_, self.params_w_['w1'], self.params_b_['b1'])
        conv_2 = conv3d(conv_1, self.params_w_['w2'], self.params_b_['b2'])
        conv_3 = conv3d(conv_2, self.params_w_['w3'], self.params_b_['b3'])
        res1 = conv3d(conv_1, self.params_w_['wRes1'], self.params_b_['bRes1'])
        merge_1 = tf.add_n([conv_3, res1], "merge_1")
        
        conv_4 = conv3d(merge_1, self.params_w_['w4'], self.params_b_['b4'])
        conv_5 = conv3d(conv_4, self.params_w_['w5'], self.params_b_['b5'])
        res2 = conv3d(conv_5, self.params_w_['wRes2'], self.params_b_['bRes2'])
        merge_2 = tf.add_n([conv_5, res2], "merge_2")
        conv_6 = conv3d(merge_2, self.params_w_['w6'], self.params_b_['b6'])
        conv_7 = conv3d(conv_6, self.params_w_['w7'], self.params_b_['b7'])
        merge_3 = tf.add_n([conv_6, conv_7], "merge_3")
        convD_8 = d_conv3d(merge_3, self.params_w_['w8'], self.params_b_['b8'], d_rate=2, stride=1)
        convD_9 = d_conv3d(convD_8, self.params_w_['w9'], self.params_b_['b9'], d_rate=2, stride=1)
        merge_4 = tf.add_n([convD_8, convD_9], "merge_4")
        convD_10 = d_conv3d(merge_4, self.params_w_['w10'], self.params_b_['b10'], d_rate=2, stride=1) 
        convD_11 = d_conv3d(convD_10, self.params_w_['w11'], self.params_b_['b11'], d_rate=2, stride=1)
        merge_5 = tf.add_n([convD_10, convD_11], "merge_5")
        
        concat = tf.concat(axis=4, values=[merge_3, merge_4, merge_5]) 
        
        convT_12 = conv3d(concat, self.params_w_['w12'], self.params_b_['b12'] )
        convT_13 = conv3d(convT_12, self.params_w_['w13'], self.params_b_['b13'] )
        
        conv_out = conv3d(convT_13, self.params_w_['wOut'], self.params_b_['bOut']) 
        
        netOut = tf.contrib.layers.flatten(conv_out)
        return netOut

    # ------------------------------------------------------------------------------------------------------------

    def costFun(self):
        logits = tf.reshape(self.score, [-1, 14])
        labels = tf.reshape(self.y, [-1])
        total = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        return total

    # ------------------------------------------------------------------------------------------------------------

    def updateFun(self):
        return tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.cost)

    # ------------------------------------------------------------------------------------------------------------

    def sumFun(self):
        return tf.summary.merge_all()

    # ------------------------------------------------------------------------------------------------------------

    def __init__(self, x, y, lr, keepProb, phase, batchSize):
        self.x_ = x
        self.y = y
        self.lr = lr
        self.keepProb = keepProb
        self.phase = phase
        self.batchSize = batchSize

        [self.params_w_, self.params_b_] = ConvNet.paramsFun(self)  # initialization and packing the parameters
        self.score = ConvNet.scoreFun(self)  # Computing the score function
        self.cost = ConvNet.costFun(self)  # Computing the cost function
        self.update = ConvNet.updateFun(self)  # Computing the update function
        self.sum = ConvNet.sumFun(self)  # summary logger 4 TensorBoard

# ====================================================================================================================

def backup(sess, saver, train_cost, valid_cost, train_accu1, train_accu2, valid_accu1, valid_accu2):
    print("Saving the model...")
    saver.save(sess, directory + '/my-model')
    writeCostNaccu(train_cost, valid_cost, train_accu1, train_accu2, valid_accu1, valid_accu2)

    # Visualize Validation Set ---------------------------------
    print("Creating ply files...")

    colors = []
    colors.append(" 0 0 0 255  ")  # black      for 0  'empty'
    colors.append(" 139 0 0 255")  # dark red   for 1  'ceiling'
    colors.append(" 0 128 0 255")  # green      for 2  'floor'
    colors.append(" 173 216 230 255")  # light blue for 3  'wall'
    colors.append(" 0 0 255 255")  # blue       for 4  'window'
    colors.append(" 255 0 0 255")  # red        for 5  'door'
    colors.append(" 218 165 32 255")  # goldenrod  for 6  'chair'
    colors.append(" 210 180 140 255")  # tan        for 7  'bed'
    colors.append(" 128 0   128 255")  # purple     for 8  'sofa'
    colors.append(" 0  0 139 255")  # dark blue  for 9  'table'
    colors.append(" 255 255 0 255")  # yellow     for 10 'coffee table'
    colors.append(" 128 128 128 255")  # gray       for 11 'shelves'
    colors.append(" 0 100 0 255")  # dark green for 12 'cabinets'
    colors.append(" 255 165 0 255")  # orange     for 13 'furniture'

    flag = 0
    files = ''
    npy = '*.npy'
    npytest = '*.npytest'

    while (True):
        counter = 0

        if flag == 0:
            files = npytest
        else:
            files = npy
            flag = 1

        for test in glob.glob(files):
            scene = np.load(test) 
            scene = scene.transpose(1, 2, 0) 

            trData = scene[0:26, 0:30, 0:30]  # input
            trLabel = scene[0:26, 0:30, 30:60]  # gt

            trData = np.reshape(trData, (-1, 30 * 26 * 30))
            score = sess.run(ConvNet_class.score, feed_dict={x: trData, keepProb: 1.0, phase: False, batchSize: 1})
            score = np.reshape(score, (26, 30, 30, 14))
            score = np.argmax(score, 3)
            score = np.reshape(score, (26, 30, 30))
            score = score[0:26, 0:30, 1:30]
            trData = np.reshape(trData, (26, 30, 30))

            scn = np.concatenate((trData, score), axis=2)

            output = open(directory + "/" + test + ".ply", 'w')
            ply = ""
            numOfVrtc = 0
            for idx1 in range(26):
                for idx2 in range(30):
                    for idx3 in range(59):
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
                    print(".ply files are done!")
                    return
                else:
                    flag = 1
                    break

# ====================================================================================================================

def show_result(sess):
    # Visualize Validation Set ---------------------------------
    print("The model is running ...")

    colors = []
    colors.append(" 0 0 0 255  ")      # black      for 0  'empty'
    colors.append(" 139 0 0 255")      # dark red   for 1  'ceiling'
    colors.append(" 0 128 0 255")      # green      for 2  'floor'
    colors.append(" 173 216 230 255")  # light blue for 3  'wall'
    colors.append(" 0 0 255 255")      # blue       for 4  'window'
    colors.append(" 255 0 0 255")      # red        for 5  'door'
    colors.append(" 218 165 32 255")   # goldenrod  for 6  'chair'
    colors.append(" 210 180 140 255")  # tan        for 7  'bed'
    colors.append(" 128 0   128 255")  # purple     for 8  'sofa'
    colors.append(" 0  0 139 255")     # dark blue  for 9  'table'
    colors.append(" 255 255 0 255")    # yellow     for 10 'coffee table'
    colors.append(" 128 128 128 255")  # gray       for 11 'shelves'
    colors.append(" 0 100 0 255")      # dark green for 12 'cabinets'
    colors.append(" 255 165 0 255")    # orange     for 13 'furniture'

    bs = 64 
    counter = 0
    a1, a2 = [], []
    trData, trLabel, batch_arr = [], [], []
    for test in glob.glob(data_directory + '/*.npy'): 
        if counter < bs:
            scene = np.load(test) 
            scene = scene.transpose(1, 2, 0)  
            batch_arr.append(scene)
            counter += 1
        else:
            counter = 0 
            batch_arr = np.reshape(batch_arr, (bs, 26, 30, 60))
            trData = batch_arr[:, 0:26, 0:30, 0:30]  # input
            trLabel = batch_arr[:, 0:26, 0:30, 30:60]  # gt
            trData = np.reshape(trData, (-1, 30 * 26 * 30))
            score = sess.run(ConvNet_class.score, feed_dict={x: trData, keepProb: 1.0, phase: False, batchSize: bs})
            accu1, accu2 = accuFun(sess, trData, trLabel, bs)
            a1.append(accu1)
            a2.append(accu2) 
            trData, trLabel, batch_arr = [], [], []
    
    if len(batch_arr) > 0: 
        bs = len(batch_arr)
        batch_arr = np.reshape(batch_arr, (bs, 26, 30, 60))
        trData = batch_arr[:, 0:26, 0:30, 0:30]  # input
        trLabel = batch_arr[:, 0:26, 0:30, 30:60]  # gt
        trData = np.reshape(trData, (-1, 30 * 26 * 30))
        score = sess.run(ConvNet_class.score, feed_dict={x: trData, keepProb: 1.0, phase: False, batchSize: bs})
        accu1, accu2 = accuFun(sess, trData, trLabel, bs)
        a1.append(accu1)
        a2.append(accu2) 
        
    accu1_avg = sum(a1) / (len(a1) * 1.0 )  
    accu2_avg = sum(a2) / (len(a2) * 1.0 )  
    print("Accuracy: ", accu1_avg, " Completeness:", accu2_avg)
    
    print("Creating .ply files ...")
    for test in glob.glob(data_directory + '/*.npy'):
        scene = np.load(test)
        trData, trLabel = [], []

        scene = np.load(test) 
        scene = scene.transpose(1, 2, 0)

        trData = scene[0:26, 0:30, 0:30]  # input
        trLabel = scene[0:26, 0:30, 30:60]  # gt

        trData = np.reshape(trData, (-1, 30 * 26 * 30))
        score = sess.run(ConvNet_class.score, feed_dict={x: trData, keepProb: 1.0, phase: False, batchSize: 1})
        score = np.reshape(score, (26, 30, 30, 14))
        score = np.argmax(score, 3)
        score = np.reshape(score, (26, 30, 30))
        score = score[0:26, 0:30, 1:30]
        trData = np.reshape(trData, (26, 30, 30))

        scn = np.concatenate((trData, score), axis=2)

        output = open( directory + "/" + test[9:] + ".ply" , 'w') 
        ply = ""
        numOfVrtc = 0
        for idx1 in range(26):
            for idx2 in range(30):
                for idx3 in range(59):
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
        # print(test + ".ply" + " is Done!")
        
# ====================================================================================================================

def accuFun(sess, trData, trLabel, batch_size):
    score = sess.run(ConvNet_class.score, feed_dict={x: trData, keepProb: 1.0, phase: False, batchSize: batch_size})
    score = np.reshape(score, (batch_size, 26, 30, 30, 14))
    trLabel = np.reshape(trLabel, (batch_size, 26, 30, 30))

    totalAccuOveral = 0.0
    totalAccuOccupied = 0.0

    for idxBatch in range(0, batch_size):
        positiveOverall = 0.0
        positiveOccupied = 0.0
        totalOccupied = 0.0

        for idx2 in range(0, 26):
            for idx3 in range(0, 30):
                for idx4 in range(0, 30):
                    maxIdxPred = np.argmax(score[idxBatch][idx2][idx3][idx4])

                    if maxIdxPred == trLabel[idxBatch][idx2][idx3][idx4]:
                        positiveOverall += 1.0
                        if maxIdxPred > 0:
                            positiveOccupied += 1

                    if trLabel[idxBatch][idx2][idx3][idx4] > 0:
                        totalOccupied += 1

        totalAccuOveral += (positiveOverall / 23400.0)
        if totalOccupied == 0:
            totalOccupied = 23400
        totalAccuOccupied += (positiveOccupied / totalOccupied)

    totalAccuOveral = totalAccuOveral / (batch_size * 1.0)
    totalAccuOccupied = totalAccuOccupied / (batch_size * 1.0)

    return totalAccuOveral, totalAccuOccupied

# ====================================================================================================================

if __name__ == '__main__':

    input_shape = 26 * 30 * 30
    out_shape = 26 * 30 * 30
    x = tf.placeholder(tf.float32, [None, input_shape])
    y = tf.placeholder(tf.int32, [None, out_shape])
    lr = tf.placeholder(tf.float32)
    keepProb = tf.placeholder(tf.float32)
    batchSize = tf.placeholder(tf.int32)
    phase = tf.placeholder(tf.bool)
    dropOut = 0.5
    batch_size = 100
    maxEpoch = 50
    ConvNet_class = ConvNet(x, y, lr, keepProb, phase, batchSize)
    initVar = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:

        sess.run(initVar)
        # restore model weights
        if to_restore:
            if os.path.exists(directory + '/my-model.meta'):
                new_saver = tf.train.import_meta_graph(directory + '/my-model.meta')
                new_saver.restore(sess, tf.train.latest_checkpoint(directory))
                print("\r\n------------ Saved weights restored. ! ------------")

        print("\r\n---------------------------------------------------") 
        
        # -------------- test phase --------------
        if not to_train:
            show_result(sess)
            print(".ply files are created.!")
            sys.exit(0)

        # -------------- train phase --------------
        step = 0
        counter = 0
        epoch = 1
        alr = 0.00005
        train_cost = []
        valid_cost = []
        train_accu1 = []
        train_accu2 = []
        valid_accu1 = []
        valid_accu2 = []
        batch = []

        accu1tr, accu2tr = 0, 0

        while True:
            for axisX in range(30, 40):

                print("\r\n ----- This is the loop: " + str(axisX - 29) + " of 10 augmentation loops. -----")
                saver.save(sess, directory + '/my-model')
                print("\r\n Model saved! \r\n")

                for npyFile in glob.glob('*.npy'):
                    trData, trLabel = [], []

                    if counter < batch_size:
                        scene = np.load(npyFile)
                        temp = np.zeros((75, 26, 30))
                        temp[15:75, :, :] = scene
                        batch.append(temp)
                        counter += 1
                    else:
                        counter = 0
                        batch = np.reshape(batch, (-1, 75, 26, 30))
                        batch = batch.transpose(0, 2, 3, 1)  # transpose to 26x30x60

                        trData = batch[:, 0:26, 0:30, axisX - 30:axisX]  # input
                        trLabel = batch[:, 0:26, 0:30, axisX:axisX + 30]  # gt

                        trData = np.reshape(trData, (-1, 30 * 26 * 30))
                        trLabel = np.reshape(trLabel, (-1, 26 * 30 * 30))

                        sess.run(ConvNet_class.update, feed_dict={x: trData,
                                                                  y: trLabel,
                                                                  lr: alr,
                                                                  keepProb: dropOut,
                                                                  phase: True,
                                                                  batchSize: batch_size})

                        cost = sess.run(ConvNet_class.cost, feed_dict={x: trData,
                                                                       y: trLabel,
                                                                       keepProb: 1.0,
                                                                       phase: True,
                                                                       batchSize: batch_size})

                        if step % 10 == 0:
                            print("%s , E:%g , S:%3g , lr:%g , accu1: %4.3g , accu2: %4.3g , Cost: %2.3g " %
                                  (str(datetime.datetime.now().time())[:-7], epoch, step, alr, accu1tr, accu2tr, cost))

                        # -------------- validation --------------  
                        if step % 500 == 0:
                            accu1tr, accu2tr = accuFun(sess, trData, trLabel, batch_size)

                        # -------------- write cost and accuracy --------------  
                        if step % 1000 == 0:
                            backup(sess, saver, train_cost, valid_cost,
                                   train_accu1, train_accu2, valid_accu1, valid_accu2)
                            print("axisX: " + str(axisX))

                        step += 1
                        batch = []

                        # END for binFile in glob
            # END for axisX in range(30,60)
            # --------------------------------------------------------------

            epoch += 1
            if epoch > maxEpoch:
                print(" --- \r\n --- \r\n  Trainig process is done after " + str(maxEpoch)
                      + " epochs and 5 augmentation loops.! \r\n --- \r\n ---")
                sys.exit(0)

# =====================================================================================================================