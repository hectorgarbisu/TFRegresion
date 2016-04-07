import sys
from random import shuffle

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

num_epochs = 20000
# dataset_lenght = 100
window_size = 5
num_batches = 1
alpha = 0.01
nW_hidden = 5

def setupSLP(input_size,hidden_size,output_size):
    x = tf.placeholder("float", [None, input_size])  # "None" as dimension for versatility between batches and non-batches
    y_ = tf.placeholder("float", [None, output_size])
    W_hidden = tf.Variable(tf.truncated_normal([input_size, hidden_size]))
    b_hidden = tf.Variable(tf.truncated_normal([hidden_size]))
    y_hidden = tf.tanh(tf.matmul(x, W_hidden) + b_hidden)
    W_output = tf.Variable(tf.truncated_normal([hidden_size, output_size]))
    b_output = tf.Variable(tf.truncated_normal([output_size]))
    y = tf.tanh(tf.matmul(y_hidden, W_output) + b_output) # If 2 layers
    error_measure = tf.reduce_sum(tf.square(y_ - y))
    train = tf.train.GradientDescentOptimizer(alpha).minimize(error_measure)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    return x,y_,y,sess,error_measure,train

def get_data(args):
    # if There are arguments
    if (len(args)>1):
        path = args[1]
        # TODO Test for folder instead of one particular file
        # a folder should contain the same signature repeated through files
        # a batch should contain sequences across multiple samples
        try:
            f = open(path,'r')
            period = f.readline()
            xdim, ydim = f.readline().split()
            name = f.readline()
            data = list()
            #  TODO Adjust every signature to a fixed length (1000 points or so)
            for line in f:
                xi,yi = line.split()
                data.append(2*float(xi)/float(xdim)-1)
                data.append(1-2*float(yi)/float(ydim))
            dataset_length = len(data)/2
            f.close()
        # is a directory or file doesnt exist
        except:
            filelist = os.listdir(path)
            for filename in filelist:
                print "file "+path+filename
            print "total: ",len(filelist)," files"
    # if program is called without arguments
    else:
        dataset_length = 100
        t = np.arange(0, 2*np.pi, 2*np.pi/dataset_length)
        # xdata = np.cos(t)/2 # circle
        # ydata = np.sin(t)/2
        # xdata = np.cos(t)/2 + np.cos(10*t)*0.05 # close to circle
        # ydata = np.sin(t)/2 + np.cos(10*t)*0.05
        xdata = np.cos(t)/2 + np.sin(10*t)*0.05 # close to circle
        ydata = np.sin(t)/2 + np.cos(10*t)*0.05
        data = np.asarray(zip(xdata, ydata)).flatten().tolist()
    return data,dataset_length

x,y_,y,sess,error_measure,train = setupSLP(window_size*2,nW_hidden,2)
data,dataset_length = get_data(sys.argv)
print dataset_length
batch_size = dataset_length
print "----------------------"
print "   Start training...  "
print "----------------------"

#NOW: whole set of windows should be generated, then randomized and then batched

random_window_indexes = [2*i for i in range(dataset_length)]
shuffle(random_window_indexes)

# print xdata,ydata,data

i = 0
for epoch in range(num_epochs):
    for current_batch in range(num_batches): #One batch
        xbatch = np.zeros([batch_size,window_size*2])
        ybatch = np.zeros([batch_size,2])
        for yy in range(batch_size): #Each element of the batch is a window-sized set of coordinate pairs
            ri = random_window_indexes[i%len(random_window_indexes)]
            i=i+1
            for jj in range(2*window_size):
                xbatch[yy][jj] = data[(ri+jj)%len(data)]
            ybatch[yy][0] = data[(ri+2*window_size)%len(data)]
            ybatch[yy][1] = data[(ri+2*window_size+1)%len(data)]

        # print "xbatch",xbatch.size
        sess.run(train, feed_dict={x: xbatch, y_: ybatch})
        #print sess.run(error_measure, feed_dict={x: xs, y_: ys})

    if (epoch % (num_epochs//10)) == 0:
        print "error:",sess.run(error_measure, feed_dict={x: xbatch, y_: ybatch})
        #print sess.run(y, feed_dict={x: xs})
        #print "----------------------------------------------------------------------------------"

print "----------------------"
print "   Start testing...  "
print "----------------------"
outs = data[:2*window_size]
test_size = dataset_length
for yy in range(test_size):
    xs = np.atleast_2d([outs[2*yy+i] for i in range(2*window_size)])
    out = sess.run(y, feed_dict={x: xs})
    outs.append(out[0][0])
    outs.append(out[0][1])
    # print xs
    # print outs

plt.plot(data[0::2],data[1::2])
xout = [outs[i] for i in range(0,len(outs),2)]
yout = [outs[i] for i in range(1,len(outs),2)]
# yout = outs[range(1,len(outs),2)]
plt.plot(xout,yout)
plt.plot(xout[-1], yout[-1], 'ro')
plt.show()
