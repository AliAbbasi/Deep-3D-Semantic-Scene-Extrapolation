import numpy as np
import glob

# ====================================================================================================================

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

# ====================================================================================================================

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
                    
# ====================================================================================================================

