from classNinapro import Ninapro
import numpy as np
from usefulFcns import *

import tensorflow as tf
print(tf.__version__)

Debug = True  # for tensor dimensionality checking
ninapro = Ninapro()
ninapro.splitImagesLabels()

# Train
print('ninapro.TrainImages shape: ', ninapro.TrainImages.shape)   # m x 16 x 30
print('ninapro.TrainLabels shape: ',  ninapro.TrainLabels.shape)  # m x 8
# Test
print('ninapro.TestImages shape: ', ninapro.TestImages.shape)     # m x 16 x 30
print('ninapro.TestLabels shape: ', ninapro.TestLabels.shape)     # m x 8
# Validate
print('ninapro.ValidateImages shape: ', ninapro.ValidateImages.shape) # m x 16 x 30
print('ninapro.ValidateLabels shape: ', ninapro.ValidateLabels.shape) # m x 8

print('Read successfully  done...')
# Scale the original RMS pixel value
#ninapro.TrainImages *= 1000
#ninapro.TestImages *= 1000
#ninapro.ValidateImages *= 1000

# number of total classes of movements, 8 for exampel.
nMV = ninapro.TrainLabels.shape[1]
partIndex = [0,1,2,3,4,5,6,10,11,12,13,14,15] # exclude [7,8,9] these three channels.
nCh = 13
# - build the Convolutional Neural Network

# Setup placeholders for input data

with tf.name_scope('Input'):
    x = tf.placeholder(tf.float32, shape=[None, nCh,30], name='X')
    y = tf.placeholder(tf.float32, shape=[None, nMV], name='Labels')

    if Debug:
        print('input x shape: ', x.shape)
        print('input y shape: ', y.shape)

with tf.name_scope('Flattern'):
    x_flatten = tf.reshape(x, [-1, nCh*30])

if Debug:
    print('x_image shape: ', x_flatten.shape)



firstIn = nCh*30 # 13*30 = 480
firstOut = 1024                                 # ---- number of hidden units in the first layer.
with tf.name_scope('ReLu-1'):
    #w1 = tf.Variable(tf.truncated_normal([firstIn, firstOut], stddev=0.1), name = 'W')
    #b1 = tf.Variable(tf.constant(0.1, shape=[firstOut]), name = 'B' )
    w1 = tf.get_variable('W1', [firstIn, firstOut], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable('B1', [1, firstOut], initializer=tf.contrib.layers.xavier_initializer())
    z1 = tf.add(tf.matmul(x_flatten, w1), b1)
    a1 = tf.nn.relu(z1)
    # summary
    tf.summary.histogram('weights', w1)
    tf.summary.histogram('biases', b1)
    tf.summary.histogram('z', z1)
    tf.summary.histogram('activation', a1) 
    

    # dimensionality checking
    if Debug:
        print('w1 shape: ', w1.shape)
        print('b1 shape: ', b1.shape)
        print('z1 shape: ', z1.shape)
        print('a1 shape: ', a1.shape)




secondIn = firstOut
secondOut = nMV
with tf.name_scope('Softmax'):
    #w3 = tf.Variable(tf.truncated_normal([thirdIn, thirdOut], stddev=0.1), name='W')
    #b3 = tf.Variable(tf.constant(0.1, shape=[thirdOut]), name='B')
    w2 = tf.get_variable('W2', [secondIn, secondOut], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable('B2', [1, secondOut], initializer=tf.contrib.layers.xavier_initializer())
    z2 = tf.add(tf.matmul(a1, w2), b2)
    # summary
    tf.summary.histogram('weights', w2)
    tf.summary.histogram('biases', b2)
    tf.summary.histogram('z', z2)

    # dimensionality checking
    if Debug:
        print('w2 shape: ', w2.shape)
        print('b2 shape: ', b2.shape)
        print('z2 shape: ', z2.shape)


with tf.name_scope('Loss'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=z2, labels=y), name='Loss')
    # summary
    tf.summary.scalar('cross_entropy', cross_entropy)
    

with tf.name_scope('Accuracy'):
    correct_prediction = tf.equal(tf.argmax(z2, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # summary
    tf.summary.scalar('accuracy', accuracy)

# Use an AdamOptimizer to train the network
#train = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
train = tf.train.GradientDescentOptimizer(1e-1).minimize(cross_entropy)

# Visualization directory
graph_dir = 'graphReLuSoftmax'
import usefulFcns
usefulFcns.BuildNewlyDir(graph_dir)

# Train the model

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(graph_dir)
    writer.add_graph(sess.graph)

    for i in range(2000):
        x_batch, y_batch = ninapro.next_batch(100)

        # Occasionaly report accuracy of [train] and [test]
        if i%100==0:
            [train_accuracy] = sess.run([accuracy], feed_dict={x:x_batch[:, partIndex, :], y:y_batch})
            [test_accuracy] = sess.run([accuracy], feed_dict={x:ninapro.TestImages[:, partIndex, :], y:ninapro.TestLabels})
            print('Step %d, training %g, testing %g.' % (i, train_accuracy, test_accuracy) )
    
            # backwards debug
            [y_hat] = sess.run([tf.nn.softmax(z2)], feed_dict={x:x_batch[:, partIndex, :], y:y_batch})
            print(y_batch.shape)
            print(y_hat.shape)
            print(np.argmax(y_batch, axis=1))
            print(np.argmax(y_hat, axis=1))
        # Occasionaly write visualization summary to disk file.
        if i%5==0:
            s = sess.run(merged_summary, feed_dict={x:x_batch[:, partIndex, :], y:y_batch})
            writer.add_summary(s,i)
        # Training the model
        sess.run(train, feed_dict={x:x_batch[:, partIndex, :], y:y_batch})
