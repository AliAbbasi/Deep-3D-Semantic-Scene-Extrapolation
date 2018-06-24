
# ====================================================================================================================

import tensorflow as tf
import numpy as np 
import os, shutil, sys, datetime, glob

# ====================================================================================================================

to_train   = False
to_restore = True
directory  = "gan_se_directory"  

argv_list = str(sys.argv) 
if 'SUN' in argv_list:
    data_directory  = 'data/SUN' 
elif 'NYU' in argv_list:
    data_directory  = 'data/NYU' 
else:
    print("Invalid Arguments.!")
    sys.exit(0)

batch_size  = 100

# ====================================================================================================================

def gaussian_noise(input, sigma = 0.1): 
    noisy = np.random.normal(0.0, sigma, tf.to_int64(input).get_shape())
    return noisy + input
    
# ====================================================================================================================

def writeCostNaccu(g_loss, d_loss, s_loss, ds_loss): 
    output = open(directory + "/costs.py" , 'w') 
    output.write( "import matplotlib.pyplot as plt" + "\r\n" )
    output.write( "g_loss  = []" + "\r\n" )
    output.write( "d_loss  = []" + "\r\n" )
    output.write( "s_loss  = []" + "\r\n" )
    output.write( "ds_loss = []" + "\r\n" )
    output.write( "steps   = []" + "\r\n" ) 
    for i in range(len(g_loss)):
        output.write( "steps.append("+ str(i) +")" + "\r\n" )
    for i in range(len(g_loss)):
        output.write( "g_loss.append("+ str(g_loss[i]) +")" + "\r\n" )
    output.write( "\r\n \r\n \r\n" )
    for i in range(len(d_loss)): 
        output.write( "d_loss.append("+ str(d_loss[i]) +")" + "\r\n" )  
    for i in range(len(s_loss)): 
        output.write( "s_loss.append("+ str(s_loss[i]) +")" + "\r\n" ) 
    for i in range(len(ds_loss)): 
        output.write( "ds_loss.append("+ str(ds_loss[i]) +")" + "\r\n" ) 
    output.write( "plt.plot( steps , g_loss,  color ='b', lw=1 )         " + "\r\n" ) 
    output.write( "plt.plot( steps , d_loss,  color ='r', lw=1 )         " + "\r\n" ) 
    # output.write( "plt.plot( steps , s_loss,  color ='g', lw=1 )         " + "\r\n" ) 
    # output.write( "plt.plot( steps , ds_loss, color ='y', lw=1 )         " + "\r\n" ) 
    output.write( "plt.xlabel('Epoch', fontsize=14)                      " + "\r\n" )
    output.write( "plt.ylabel('Loss',  fontsize=14)                      " + "\r\n" )
    output.write( "plt.suptitle('Blue: G, Red: D, Green: S, Yellow: DS') " + "\r\n" )
    output.write( "plt.show()                                            " + "\r\n" ) 
    print ("costs.py file is created! \r\n")
    
# ====================================================================================================================

def lrelu(input, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * input + f2 * tf.abs(input)
    
# ====================================================================================================================

def conv2d(x, w, b, name="conv2d", strides=1):
    with tf.name_scope(name):
        x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b) 
        return x  
        
# ====================================================================================================================

def conv2d_transpose(x, w, b, output_shape, name="conv2d_transpose", strides=2):
    with tf.name_scope(name): 
        x = tf.nn.conv2d_transpose(x, w, output_shape=output_shape, strides=[1,strides,strides,1])
        x = tf.nn.bias_add(x, b) 
        return x  

# ====================================================================================================================

def d_conv2d(x, w, b, name="d_conv2d", d_rate=1):
    with tf.name_scope(name): 
        x = tf.nn.convolution(x, w, padding='SAME', strides=[1,1], dilation_rate=[d_rate, d_rate], name=name)
        x = tf.nn.bias_add(x, b) 
        return x 
        
# ====================================================================================================================

G_W1  = tf.Variable(tf.truncated_normal( [ 3 , 3 , 30  , 128 ], stddev = 0.01 )) 
 
G_W2  = tf.Variable(tf.truncated_normal( [ 1 , 1 , 128 , 256 ], stddev = 0.01 ))   
G_W3  = tf.Variable(tf.truncated_normal( [ 3 , 3 , 256 , 128 ], stddev = 0.01 )) 
G_W4  = tf.Variable(tf.truncated_normal( [ 1 , 1 , 128 , 128 ], stddev = 0.01 ))  

G_W5  = tf.Variable(tf.truncated_normal( [ 1 , 1 , 128 , 256 ], stddev = 0.01 ))   
G_W6  = tf.Variable(tf.truncated_normal( [ 3 , 3 , 256 , 256 ], stddev = 0.01 ))   
G_W7  = tf.Variable(tf.truncated_normal( [ 1 , 1 , 256 , 128 ], stddev = 0.01 )) 
  
G_W8  = tf.Variable(tf.truncated_normal( [ 1 , 1 , 128 , 256 ], stddev = 0.01 ))   
G_W9  = tf.Variable(tf.truncated_normal( [ 3 , 3 , 256 , 256 ], stddev = 0.01 ))   
G_W10 = tf.Variable(tf.truncated_normal( [ 1 , 1 , 256 , 128 ], stddev = 0.01 ))   
G_W11 = tf.Variable(tf.truncated_normal( [ 3 , 3 , 128 , 256 ], stddev = 0.01 ))    
G_W12 = tf.Variable(tf.truncated_normal( [ 1 , 1 , 256 , 128 ], stddev = 0.01 ))   
G_W13 = tf.Variable(tf.truncated_normal( [ 1 , 1 , 128 , 14*30  ], stddev = 0.01 )) 

G_b1  = tf.Variable(tf.truncated_normal( [               128 ], stddev = 0.01 ))   
G_b2  = tf.Variable(tf.truncated_normal( [               256 ], stddev = 0.01 ))   
G_b3  = tf.Variable(tf.truncated_normal( [               128 ], stddev = 0.01 ))  
G_b4  = tf.Variable(tf.truncated_normal( [               128 ], stddev = 0.01 ))  
G_b5  = tf.Variable(tf.truncated_normal( [               256 ], stddev = 0.01 ))  
G_b6  = tf.Variable(tf.truncated_normal( [               256 ], stddev = 0.01 ))  
G_b7  = tf.Variable(tf.truncated_normal( [               128 ], stddev = 0.01 ))  
G_b8  = tf.Variable(tf.truncated_normal( [               256 ], stddev = 0.01 ))  
G_b9  = tf.Variable(tf.truncated_normal( [               256 ], stddev = 0.01 ))  
G_b10 = tf.Variable(tf.truncated_normal( [               128 ], stddev = 0.01 ))  
G_b11 = tf.Variable(tf.truncated_normal( [               256 ], stddev = 0.01 ))  
G_b12 = tf.Variable(tf.truncated_normal( [               128 ], stddev = 0.01 ))  
G_b13 = tf.Variable(tf.truncated_normal( [               14*30  ], stddev = 0.01 ))

g_params   = [G_W1, G_W2, G_W3, G_W4, G_W5, G_W6, G_b1, G_b2, G_b3, G_b4, G_b5, G_b6, G_b7 ,G_b8 ,G_b9 ,G_b10,G_b11,G_b12,G_b13,G_W7 ,G_W8 ,G_W9 ,G_W10,G_W11,G_W12,G_W13]   

def generator(f_half_real, keep_prob):   
    inputs    = tf.reshape( f_half_real, [-1, 26, 30, 30] )   
    conv_1    = conv2d( inputs, G_W1, G_b1, "conv_1" ) 
    
    # Residual Block #1
    conv_r1_1 = tf.nn.dropout( lrelu( conv_1 )                                       , keep_prob)
    conv_r1_2 = tf.nn.dropout( lrelu( conv2d( conv_r1_1, G_W2, G_b2, "conv_r1_2" ) ) , keep_prob)  
    conv_r1_3 = tf.nn.dropout( lrelu( conv2d( conv_r1_2, G_W3, G_b3, "conv_r1_3" ) ) , keep_prob)
    conv_r1_4 = tf.nn.dropout(        conv2d( conv_r1_3, G_W4, G_b4, "conv_r1_4" )   , keep_prob) 
    
    merge_1   = tf.add_n([conv_1, conv_r1_4])   
    
    # Residual Dilated Conv Block #2
    conv_r2_1 = tf.nn.dropout( lrelu( merge_1 )                                           , keep_prob )
    conv_r2_2 = tf.nn.dropout( lrelu( d_conv2d( conv_r2_1, G_W5, G_b5, "conv_r2_2", 2 ) ) , keep_prob ) 
    conv_r2_3 = tf.nn.dropout( lrelu( d_conv2d( conv_r2_2, G_W6, G_b6, "conv_r2_3", 4 ) ) , keep_prob )
    conv_r2_4 = tf.nn.dropout(        d_conv2d( conv_r2_3, G_W7, G_b7, "conv_r2_4", 8 )   , keep_prob )
    
    merge_2   = tf.add_n([merge_1, conv_r2_4])    
    
    # Residual Block #3
    conv_r3_1 = tf.nn.dropout( lrelu( merge_2 )                                        , keep_prob)
    conv_r3_2 = tf.nn.dropout( lrelu( conv2d( conv_r3_1, G_W8,  G_b8,  "conv_r3_2" ) ) , keep_prob) 
    conv_r3_3 = tf.nn.dropout( lrelu( conv2d( conv_r3_2, G_W9,  G_b9,  "conv_r3_3" ) ) , keep_prob)
    conv_r3_4 = tf.nn.dropout(        conv2d( conv_r3_3, G_W10, G_b10, "conv_r3_4" )   , keep_prob) 
    
    merge_3   = lrelu( tf.add_n([merge_2, conv_r3_4]) )   
    
    conv_2    = tf.nn.dropout( lrelu( conv2d( merge_3, G_W11, G_b11, "conv_2" ) ) , keep_prob) 
    conv_3    = tf.nn.dropout( lrelu( conv2d( conv_2,  G_W12, G_b12, "conv_3" ) ) , keep_prob) 
    
    merge_4   = lrelu( tf.add_n([merge_3, conv_3]) )  
    
    conv_4    = conv2d( merge_4, G_W13, G_b13, "conv_4" )           
    
    return conv_4 # 26 x 30 x 30 
    
# ====================================================================================================================

D_W1 = tf.Variable(tf.truncated_normal( [ 3 , 3 , 30   , 32   ], stddev = 0.01 ))  
D_W2 = tf.Variable(tf.truncated_normal( [ 3 , 3 , 32   , 64   ], stddev = 0.01 )) 
D_W3 = tf.Variable(tf.truncated_normal( [ 3 , 3 , 64   , 128  ], stddev = 0.01 )) 
D_W4 = tf.Variable(tf.truncated_normal( [ 3 , 3 , 128  , 128  ], stddev = 0.01 ))  

D_b1 = tf.Variable(tf.truncated_normal( [                32   ], stddev = 0.01 )) 
D_b2 = tf.Variable(tf.truncated_normal( [                64   ], stddev = 0.01 )) 
D_b3 = tf.Variable(tf.truncated_normal( [                128  ], stddev = 0.01 )) 
D_b4 = tf.Variable(tf.truncated_normal( [                128  ], stddev = 0.01 ))  

DG_W1 = tf.Variable(tf.truncated_normal( [ 3 , 3 , 60   , 64  ], stddev = 0.01 ))  
DG_W2 = tf.Variable(tf.truncated_normal( [ 3 , 3 , 64   , 128 ], stddev = 0.01 )) 
DG_W3 = tf.Variable(tf.truncated_normal( [ 3 , 3 , 128  , 256 ], stddev = 0.01 )) 
DG_W4 = tf.Variable(tf.truncated_normal( [ 3 , 3 , 256  , 128 ], stddev = 0.01 ))  

DG_b1 = tf.Variable(tf.truncated_normal( [                64  ], stddev = 0.01 )) 
DG_b2 = tf.Variable(tf.truncated_normal( [                128 ], stddev = 0.01 )) 
DG_b3 = tf.Variable(tf.truncated_normal( [                256 ], stddev = 0.01 )) 
DG_b4 = tf.Variable(tf.truncated_normal( [                128 ], stddev = 0.01 ))  

DG_fc_W = tf.Variable(tf.truncated_normal( [         4096 , 1 ], stddev = 0.01 ))  
DG_fc_b = tf.Variable(tf.truncated_normal( [                1 ], stddev = 0.01 )) 

d_params = [D_W1, D_W2, D_W3, D_W4, D_b1, D_b2, D_b3, D_b4, DG_W1, DG_W2, DG_W3, DG_W4, DG_b1, DG_b2, DG_b3, DG_b4, DG_fc_W, DG_fc_b] 

def discriminator(f_half, s_half, keep_prob):      

    inputs = tf.reshape( s_half, [batch_size, 26, 30, 30] )  
    h1     = tf.nn.dropout( lrelu( tf.layers.batch_normalization( tf.nn.bias_add( tf.nn.conv2d(inputs, D_W1, strides=[1,2,2,1], padding='SAME'),  D_b1 ) ) ), keep_prob) 
    h2     = tf.nn.dropout( lrelu( tf.layers.batch_normalization( tf.nn.bias_add( tf.nn.conv2d(h1,     D_W2, strides=[1,2,2,1], padding='SAME'),  D_b2 ) ) ), keep_prob) 
    h3     = tf.nn.dropout( lrelu( tf.layers.batch_normalization( tf.nn.bias_add( tf.nn.conv2d(h2,     D_W3, strides=[1,1,1,1], padding='SAME'),  D_b3 ) ) ), keep_prob) 
    h4     = tf.nn.dropout( lrelu( tf.layers.batch_normalization( tf.nn.bias_add( tf.nn.conv2d(h3,     D_W4, strides=[1,2,2,1], padding='SAME'),  D_b4 ) ) ), keep_prob)  

    f_half = tf.reshape( f_half, [batch_size, 26, 30, 30] )  
    s_half = tf.reshape( s_half, [batch_size, 26, 30, 30] )  
    inputs = tf.concat ( axis=3,  values=[f_half, s_half] )   
    hG1    = tf.nn.dropout( lrelu( tf.layers.batch_normalization( tf.nn.bias_add( tf.nn.conv2d(inputs, DG_W1, strides=[1,2,2,1], padding='SAME'), DG_b1 ) ) ), keep_prob) 
    hG2    = tf.nn.dropout( lrelu( tf.layers.batch_normalization( tf.nn.bias_add( tf.nn.conv2d(hG1,    DG_W2, strides=[1,2,2,1], padding='SAME'), DG_b2 ) ) ), keep_prob) 
    hG3    = tf.nn.dropout( lrelu( tf.layers.batch_normalization( tf.nn.bias_add( tf.nn.conv2d(hG2,    DG_W3, strides=[1,1,1,1], padding='SAME'), DG_b3 ) ) ), keep_prob) 
    hG4    = tf.nn.dropout( lrelu( tf.layers.batch_normalization( tf.nn.bias_add( tf.nn.conv2d(hG3,    DG_W4, strides=[1,2,2,1], padding='SAME'), DG_b4 ) ) ), keep_prob)  
    
    concat  = tf.concat ( axis=3,  values=[h4, hG4] )  
    concat  = tf.reshape( concat, [batch_size, -1 ] )      
    logits  = tf.matmul ( concat, DG_fc_W ) + DG_fc_b 
    
    return logits 
    
# ====================================================================================================================

def show_result(f_half_real, s_half_real, s_half_fake, batch_size, dataType):

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
    
    # real first half
    f_half_real = f_half_real.reshape(( batch_size, 26, 30, 30            ))  
    f_half_real = np.around(         (( f_half_real + 1.0) / 2.0) * 13     )     
    
    # real second half, add 10 empty planes to the end
    s_half_real = s_half_real.reshape(( batch_size, 26, 30, 30            )) 
    s_half_real = np.around(         (( s_half_real + 1.0) / 2.0) * 13     ) 
    temp        = np.zeros           (( batch_size, 26, 30, 10            ))
    s_half_real = np.concatenate     (( s_half_real, temp ), axis=3        )
    
    # generated second half
    s_half_fake = s_half_fake.reshape(( batch_size, 26, 30, 30, 14        )) 
    s_half_fake = np.argmax(s_half_fake, 4)   # convert from 4D to 3D tensor  
    
    # put first and second half together, real and generated
    results1    = np.concatenate(( f_half_real , s_half_real ), axis=3 )  
    results2    = np.concatenate(( f_half_real , s_half_fake ), axis=3 ) 
    results     = np.concatenate(( results1    , results2    ), axis=3 ) 
    
    temp        = np.zeros      (( batch_size  , 26, 30, 10           ))
    f_half_real = np.concatenate(( f_half_real , temp        ), axis=3 ) 
    results     = np.concatenate(( f_half_real , results     ), axis=3 ) 
    
    for i, item in enumerate(results):   
    
        output    = open( data_directory + "/" + dataType[9:] + "_generated_" + str(i) + ".ply" , 'w') 
        ply       = ""
        numOfVrtc = 0
        
        for idx1 in range(26):
            for idx2 in range(30):    
                for idx3 in range(170): 
                    if item[idx1][idx2][idx3] >= 1:  
                        ply = ply + str(idx1)+ " " +str(idx2)+ " " +str(idx3) + str(colors[ int(item[idx1][idx2][idx3]) ]) + "\n" 
                        numOfVrtc += 1 
        output.write("ply"                                    + "\n")
        output.write("format ascii 1.0"                       + "\n")
        output.write("comment VCGLIB generated"               + "\n")
        output.write("element vertex " +  str(numOfVrtc)      + "\n")
        output.write("property float x"                       + "\n")
        output.write("property float y"                       + "\n")
        output.write("property float z"                       + "\n")
        output.write("property uchar red"                     + "\n")
        output.write("property uchar green"                   + "\n")
        output.write("property uchar blue"                    + "\n")
        output.write("property uchar alpha"                   + "\n")
        output.write("element face 0"                         + "\n")
        output.write("property list uchar int vertex_indices" + "\n")
        output.write("end_header"                             + "\n") 
        output.write( ply                                           ) 
        output.close() 
        # print (str(dataType) + "_generated_" + str(i) + ".ply is Done.!") 

# ====================================================================================================================

def accuFun(sess, batch_size, trLabel, generated_scenes): 

    generated_scenes = generated_scenes.reshape(( batch_size, 26, 30, 30, 14 )) 
    generated_scenes = np.argmax( generated_scenes, 4 )   # convert from 4D to 3D tensor 
    trLabel          = np.around( (((trLabel.reshape(( batch_size, 26, 30, 30 ))) + 1.0) / 2.0) * 13 )
    
    accu1 = np.sum(generated_scenes == trLabel) / 23400.0 
    
    accu2 = 0.0
    for b in range(batch_size):
        tp = 0.0
        allTP = 0.0
        for idx1 in range(26):
            for idx2 in range(30):    
                for idx3 in range(30): 
                    if generated_scenes[b][idx1][idx2][idx3] == trLabel[b][idx1][idx2][idx3] and trLabel[b][idx1][idx2][idx3] > 0:
                        tp += 1
                    if trLabel[b][idx1][idx2][idx3] > 0:
                        allTP += 1
                        
        accu2 += (tp / allTP) if allTP != 0 else (tp / 0.000001)
        
    return (accu1 / (batch_size*1.0)), (accu2 / (batch_size*1.0))
    
# ====================================================================================================================

def train(): 
    
    # -------------- place holders -------------- 
    f_half_real = tf.placeholder(tf.float32, [None, 26, 30, 30], name="f_half_real" ) 
    s_half_real = tf.placeholder(tf.float32, [None, 26, 30, 30], name="f_half_real" ) 
    g_labels    = tf.placeholder(tf.int32,   [None            ], name="g_labels"    ) 
    keep_prob   = tf.placeholder(tf.float32, name="keep_prob")
    batchSize   = tf.placeholder(tf.int32,   name="batchSize")
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # -------------- runs -------------- 
    s_half_gen  = generator    (              f_half_real, keep_prob ) 
    real_logits = discriminator( f_half_real, s_half_real, keep_prob ) 
    
    s_h_s_temp  = tf.reshape( s_half_gen, [-1, 26, 30, 30, 14] )
    s_h_s_temp  = tf.argmax ( s_h_s_temp, 4                    ) 
    s_h_s_temp  = tf.cast   ( s_h_s_temp, tf.float32           )
    s_h_s_temp  = 2 * (s_h_s_temp / tf.constant(13.0)) - 1 
    fake_logits = discriminator( f_half_real, s_h_s_temp,  keep_prob )   
    
    # -------------- discriminator loss -------------- 
    D_loss_real  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits, labels=tf.ones_like ( real_logits )))
    D_loss_fake  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.zeros_like( fake_logits )))  
    d_loss       = D_loss_real + D_loss_fake 
    
    # -------------- generator loss -------------- 
    g_logits = tf.reshape(s_half_gen, [-1, 14])
    g_labels = tf.reshape(g_labels,   [-1    ])  
    g_loss_t = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits( logits=g_logits, labels=g_labels ))
    
    # Penalty term for G
    s_half_fake_ = tf.reshape( s_half_gen, [-1, 26, 30, 30, 14] )
    s_half_real_ = tf.reshape( g_labels,   [-1, 26, 30, 30    ] )
    split_logits = tf.split( axis=3, num_or_size_splits=30, value=s_half_fake_ )
    split_labels = tf.split( axis=3, num_or_size_splits=30, value=s_half_real_ )
    for i in range(0,len(split_logits)-1):
        g_loss_t += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits( logits=split_logits[i], labels=split_labels[i+1] ))
    
    g_loss = g_loss_t + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.ones_like ( fake_logits )))  
    
    # -------------- optimization -------------- 
    d_trainer  = tf.train.AdamOptimizer(0.00001).minimize(d_loss,  var_list=d_params )  
    g_trainer  = tf.train.AdamOptimizer(0.00001).minimize(g_loss,  var_list=g_params )   

    # -------------- initialization --------------
    init  = tf.global_variables_initializer() 
    saver = tf.train.Saver() 
    sess  = tf.Session() 
    sess.run(init)

    if to_restore:
        chkpt_fname = tf.train.latest_checkpoint(directory)
        saver.restore(sess, chkpt_fname)
    else:
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.mkdir(directory)
    
    # -------------- test phase --------------
    print("The model is running ...")
    if to_train == False:
        chkpt_fname = tf.train.latest_checkpoint(directory)
        saver.restore(sess, chkpt_fname)
        print("\r\n------------ Saved weights restored. ! ------------") 
        print("\r\n---------------------------------------------------") 
        batch_arr = [] 
        bs        = 0
        for npyFile in glob.glob(data_directory + '/*.npy'): 
            batch_arr.append( np.load(npyFile) )
            bs += 1
        batch_arr = np.reshape( batch_arr, ( bs, 60, 26, 30 ))    
        batch_arr = batch_arr.transpose(0,2,3,1)                        # transpose to 26x30x60 
        batch_arr = 2 * (batch_arr.astype(np.float32) / 13.0) - 1     
        generated_scenes = sess.run( s_half_gen, feed_dict={batchSize: bs, f_half_real: batch_arr[:, :, :, 0:30], keep_prob: np.sum(1.0).astype(np.float32)})
        accu1, accu2     = accuFun(sess, bs, batch_arr[:, :, :, 30:60], generated_scenes)
        print("Accuracy: ", accu1, " Completeness:", accu2)
        
        batch_arr = [] 
        bs        = 1 
        print("Creating .ply files...")
        for npyFile in glob.glob(data_directory + '/*.npy'): 
            batch_arr.append( np.load(npyFile) ) 
            batch_arr = np.reshape( batch_arr, ( bs, 60, 26, 30 ))    
            batch_arr = batch_arr.transpose(0,2,3,1)                        # transpose to 26x30x60 
            batch_arr = 2 * (batch_arr.astype(np.float32) / 13.0) - 1     
            generated_scenes = sess.run( s_half_gen, feed_dict={batchSize: bs, f_half_real: batch_arr[:, :, :, 0:30], keep_prob: np.sum(1.0).astype(np.float32)})
            show_result(batch_arr[:, :, :, 0:30], batch_arr[:, :, :, 30:60], generated_scenes, bs, npyFile) 
            batch_arr = [] 
        print (".ply files are created.!") 
        sys.exit(0)
        
    # -------------- training loop --------------
    
    threshold           = 0.0
    step                = 0
    counter             = 0  
    batch_arr           = [] 
    g_l_plot, d_l_plot  = [], []
    s_l_plot, ds_l_plot = [], []
    accu1, accu2        = 0.0, 0.0
    step_threshold      = 50000

    for i in range(sess.run(global_step), 50):    # epoch loop   
        for aug_idx in range(30,40):              # augmentation loop
        
            print ("\r\n ----- This is the loop: " + str(aug_idx - 29) + " of augmentation loops. ----- \r\n")
            
            for npyFile in glob.glob('/*.npy'):  
                if counter < batch_size:    
                    scene = np.load(npyFile)      
                    temp  = np.zeros((75,26,30))  
                    temp[15:75,:,:] = scene
                    batch_arr.append( temp )
                    counter += 1    
                else:                   
                    counter   = 0     
                    batch_arr = np.reshape( batch_arr, ( -1, 75, 26, 30 ))    
                    batch_arr = batch_arr.transpose(0,2,3,1)                        # transpose to 26x30x60
                    g_gt      = np.zeros((26,30,30), dtype=np.float32)              # gt for smoother 
                    g_gt      = batch_arr[:, :, :, aug_idx:aug_idx+30]
                    g_gt      = np.reshape(g_gt, (batch_size * 26*30*30))
                    batch_arr = 2 * (batch_arr.astype(np.float32) / 13.0) - 1       # normalize between [-1,+1]
                    
                    d_l, g_l = sess.run([d_loss, g_loss], feed_dict={g_labels: g_gt, batchSize: batch_size, f_half_real: batch_arr[:, :, :, aug_idx-30:aug_idx ], s_half_real: batch_arr[:, :, :, aug_idx:aug_idx+30], keep_prob: np.sum(0.5).astype(np.float32)})
                    g_l_plot .append(np.mean(g_l )) 
                    d_l_plot .append(np.mean(d_l )) 
                    
                    # -------------- update G --------------  
                    if step < step_threshold:
                        sess.run(g_trainer,  feed_dict={g_labels: g_gt, batchSize: batch_size, f_half_real: batch_arr[:, :, :, aug_idx-30:aug_idx ], s_half_real: batch_arr[:, :, :, aug_idx:aug_idx+30],keep_prob: np.sum(0.5).astype(np.float32)})

                    # -------------- update All --------------
                    if step > step_threshold:
                        sess.run(g_trainer,  feed_dict={g_labels: g_gt, batchSize: batch_size, f_half_real: batch_arr[:, :, :, aug_idx-30:aug_idx ], s_half_real: batch_arr[:, :, :, aug_idx:aug_idx+30], keep_prob: np.sum(0.5).astype(np.float32)})
                        sess.run(d_trainer,  feed_dict={g_labels: g_gt, batchSize: batch_size, f_half_real: batch_arr[:, :, :, aug_idx-30:aug_idx ], s_half_real: batch_arr[:, :, :, aug_idx:aug_idx+30], keep_prob: np.sum(0.5).astype(np.float32)})
                        
                    # -------------- show accuracy -------------- 
                    if step%500 == 0:
                        generated_scenes = sess.run(s_half_gen, feed_dict={f_half_real: batch_arr[:, :, :, aug_idx-30:aug_idx ], keep_prob: np.sum(1.0).astype(np.float32)})
                        accu1, accu2     = accuFun(sess, batch_size, batch_arr[:, :, :, 30:60], generated_scenes) 
                    
                    # -------------- generate results -------------- 
                    if step%1000 == 0:
                        generated_scenes = sess.run(s_half_gen, feed_dict={f_half_real: batch_arr[0:8, :, :, aug_idx-30:aug_idx], keep_prob: np.sum(1.0).astype(np.float32)})
                        show_result(batch_arr[0:8, :, :, 0:30], batch_arr[0:8, :, :, 30:60], generated_scenes, 8, "train") 
                        
                        batch_arr = [] 
                        for npyFile in glob.glob('*.npytest'): 
                            batch_arr.append( np.load(npyFile) )
                        batch_arr = np.reshape( batch_arr, ( 8, 60, 26, 30 ))    
                        batch_arr = batch_arr.transpose(0,2,3,1)                        # transpose to 26x30x60
                        batch_arr = 2 * (batch_arr.astype(np.float32) / 13.0) - 1  
                        generated_scenes = sess.run(s_half_gen, feed_dict={f_half_real: batch_arr[:, :, :, 0:30], keep_prob: np.sum(1.0).astype(np.float32)})
                        show_result(batch_arr[:, :, :, 0:30], batch_arr[:, :, :, 30:60], generated_scenes, 8, "test")
                        
                        writeCostNaccu(g_l_plot, d_l_plot, s_l_plot, ds_l_plot)  
            
                    # ----------------------------------------------
                    if step%10 == 0:
                        print("%s, E:%g, Step:%3g, D:%.3f, G:%.3f, A1:%.3f, A2:%.3f"%(str(datetime.datetime.now().time())[:-7], i, step, np.mean(d_l), np.mean(g_l), accu1, accu2)) 
        
                    step     += 1 
                    batch_arr = [] 
            # -------------- save model --------------  
            saver.save(sess, os.path.join(directory, "model"), global_step=global_step) 

# ====================================================================================================================

if __name__ == '__main__': 
    train() 