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

# number of total classes of movements, 8 for exampel.
nMV = ninapro.TrainLabels.shape[1]

# - build the Convolutional Neural Network

# Setup placeholders for input data

with tf.name_scope('Input'):
    x = tf.placeholder(tf.float32, shape=[None, 16,30], name='X')
    y = tf.placeholder(tf.float32, shape=[None, nMV], name='Labels')

    if Debug:
        print('input x shape: ', x.shape)
        print('input y shape: ', y.shape)

# every sample with the dimensionality, 16x30
x_image = tf.reshape(x, [-1, 16, 30, 1])
if Debug:
    print('x_image shape: ', x_image.shape)

# summary 
#tf.summary.image('input', x, 4)


firstIn = 1
firstOut = 32
with tf.name_scope('First'):
    # convolution
    w1 = tf.Variable(tf.truncated_normal([1,16, firstIn, firstOut], stddev=0.1), name = 'W')
    b1 = tf.Variable(tf.constant(0.1, shape=[firstOut]), name = 'B' )
    s1 = 1
    conv1 = tf.nn.conv2d(x_image, w1, strides=[1, s1, s1, 1], padding='SAME' )
    act1 = tf.nn.relu(conv1 + b1)
    # summary
    tf.summary.histogram('weights', w1)
    tf.summary.histogram('biases', b1)
    tf.summary.histogram('activation', act1) 

    # dimensionality checking
    if Debug:
        print('w1 shape: ', w1.shape)
        print('b1 shape: ', b1.shape)
        print('conv1 shape: ', conv1.shape)
        print('act1 shape: ', act1.shape)


secondIn = firstOut
secondOut = 32
with tf.name_scope('Second'):
    # convolution
    w2 = tf.Variable(tf.truncated_normal([3,3, secondIn, secondOut], stddev=0.1), name='W')
    b2 = tf.Variable(tf.constant(0.1, shape=[secondOut]), name='B')
    s2 = 1
    conv2 = tf.nn.conv2d(act1, w2, strides=[1, s2, s2, 1], padding='SAME')
    # detector
    act2 = tf.nn.relu(conv2 + b2)
    # maxpooling
    k2 = 3
    ms2 = 1
    mp2 = tf.nn.max_pool(act2, ksize=[1, k2,k2, 1], strides=[1,ms2,ms2,1], padding='SAME')
    # summary
    tf.summary.histogram('weights', w2)
    tf.summary.histogram('biases', b2)
    tf.summary.histogram('activation', act2)
    tf.summary.histogram('maxpooling', mp2)

    # dimensionality checking
    if Debug:
        print('w2 shape: ', w2.shape)
        print('b2 shape: ', b2.shape)
        print('conv2 shape: ', conv2.shape)
        print('act2 shape: ', act2.shape)
        print('mp2 shape: ', mp2.shape)

thirdIn = secondOut
thirdOut = 64
with tf.name_scope('Third'):
    # convolution
    w3 = tf.Variable(tf.truncated_normal([5,5, thirdIn, thirdOut], stddev=0.1), name='W')
    b3 = tf.Variable(tf.constant(0.1, shape=[thirdOut]), name='B')
    s3 = 1
    conv3 = tf.nn.conv2d(mp2, w3, strides=[1,s3,s3,1], padding='SAME')
    # detector
    act3 = tf.nn.relu(conv3 + b3)
    # maxpooling
    k3 = 3 # ksize of maxpooling
    ms3 = 1 # maxpooling stride = 3
    mp3 = tf.nn.max_pool(act3, ksize=[1,k3,k3,1], strides=[1, ms3, ms3, 1], padding='SAME')

    # summary
    tf.summary.histogram('weights', w3)
    tf.summary.histogram('biases', b3)
    tf.summary.histogram('activation', act3)
    tf.summary.histogram('maxpooling', mp3)

    # dimensionality checking
    if Debug:
        print('w3 shape: ', w3.shape)
        print('b3 shape: ', b3.shape)
        print('conv3 shape: ', conv3.shape)
        print('act3 shape: ', act3.shape)
        print('mp3 shape: ', mp3.shape)


fourthIn = thirdOut
fourthOut = 64
with tf.name_scope('Fourth'):
    # convolution
    w4 = tf.Variable(tf.truncated_normal([6,1, fourthIn, fourthOut], stddev=0.1), name='W')
    b4 = tf.Variable(tf.constant(0.1, shape=[fourthOut]), name='B')
    s4 = 1
    conv4 = tf.nn.conv2d(mp3, w4, strides=[1,s4,s4,1], padding='SAME')
    # detector
    act4 = tf.nn.relu(conv4 + b4)
    
    # summary
    tf.summary.histogram('weights', w4)
    tf.summary.histogram('biases', b4)
    tf.summary.histogram('activation', act4)

    # dimensionality checking
    if Debug:
        print('w4 shape: ', w4.shape)
        print('b4 shape: ', b4.shape)
        print('conv4 shape: ', conv4.shape)
        print('act4 shape: ', act4.shape)

fifthIn = fourthOut
fifthOut = 8
with tf.name_scope('Fifth'):
    # convolution
    w5 = tf.Variable(tf.truncated_normal([1,1, fifthIn, fifthOut], stddev=0.1), name='W')
    b5 = tf.Variable(tf.constant(0.1, shape=[fifthOut]), name='B')
    s5 = 1
    conv5 = tf.nn.conv2d(act4, w5, strides=[1,s5,s5,1], padding='SAME')
    # detector
    act5 = tf.nn.relu(conv5 + b5)

    # flatten
    with tf.name_scope('Flatten'):
        flatten5 = tf.reshape(act5, [-1, 16*30*fifthOut])
    # fully-connect layer
    with tf.name_scope('FullyCon'):
        wfc5 = tf.Variable(tf.truncated_normal( [16*30*fifthOut, nMV], stddev=0.1), name='W')
        bfc5 = tf.Variable(tf.constant(0.1, shape=[nMV]), name='B')
        y_ = tf.nn.relu(tf.matmul(flatten5, wfc5) + bfc5)

    # summary
    tf.summary.histogram('weights', w5)
    tf.summary.histogram('biases', b5)
    tf.summary.histogram('activation', act5)
    tf.summary.histogram('flatten', flatten5)
    tf.summary.histogram('weights_fc5', wfc5)
    tf.summary.histogram('biases_fc5', bfc5)
    tf.summary.scalar('fifth_weights', w5[0, 0, 0, 0])
    tf.summary.scalar('y_predict', np.argmax(y_[0,:]))
    tf.summary.scalar('ylabels', np.argmax(y[0, :]))


    # dimensionality checking
    if Debug:
        print('w5 shape: ', w5.shape)
        print('b5 shape: ', b5.shape)
        print('conv5 shape: ', conv5.shape)
        print('act5 shape: ', act5.shape)
        print('flatten5 shape: ', flatten5.shape)
        print('weights_fc5 shape: ', wfc5.shape)
        print('biases_fc5 shape: ', bfc5.shape)
        print('y_predict shape: ', y_.shape)


with tf.name_scope('Softmaxloss'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y), name='Loss')
    # summary
    tf.summary.scalar('cross_entropy', cross_entropy)
    

with tf.name_scope('Accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # summary
    tf.summary.scalar('accuracy', accuracy)

# Use an AdamOptimizer to train the network
train = tf.train.AdamOptimizer(1e-1).minimize(cross_entropy)

# Visualization directory
graph_dir = 'sEMGCNN'
import usefulFcns
usefulFcns.BuildNewlyDir(graph_dir)

# Train the model

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(graph_dir)
    writer.add_graph(sess.graph)

    for i in range(2000):
        x_batch, y_batch = ninapro.next_batch(30)

        # Occasionaly report accuracy of [train] and [test]
        if i%100==0:
            [train_accuracy] = sess.run([accuracy], feed_dict={x:x_batch, y:y_batch})
            [test_accuracy] = sess.run([accuracy], feed_dict={x:ninapro.TestImages, y:ninapro.TestLabels})
            [validate_accuracy] = sess.run([accuracy], feed_dict={x:ninapro.ValidateImages, y:ninapro.ValidateLabels} )
            print('Step %d, training %g, testing %g, validate %g.' % (i, train_accuracy, test_accuracy, validate_accuracy) )
    
        # Occasionaly write visualization summary to disk file.
        if i%5==0:
            s = sess.run(merged_summary, feed_dict={x:x_batch, y:y_batch})
            writer.add_summary(s,i)
        # Training the model
        sess.run(train, feed_dict={x:x_batch, y:y_batch})
                


