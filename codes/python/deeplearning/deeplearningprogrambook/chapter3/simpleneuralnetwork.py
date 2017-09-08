# -*- coding: utf-8 -*-
'''
Train a simple neural network model to solve binary classification problem.

@author fanglei
'''

import tensorflow as tf
from numpy.random import RandomState

'''
Generate a data set using RandomState
'''
rdm = RandomState()
dataset_size = 128
X = rdm.rand(dataset_size, 2)
# If (x1 + x2) < 1, it belongs to positive class
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]


'''
Define the parameters of the neural network which has one hidden layer with 3 neurons
'''
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1, name='w1'))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1, name='w2'))


'''
Define the input and output using placeholder with arbitrary batch size
'''
x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-output')


'''
Define the forward propagation algorithm
'''
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)


'''
Define the loss function and backward propagation algorithm
'''
cross_entropy = -tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)


'''
Create and run a session
'''
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(w1))
    print(sess.run(w2))


    '''
    Train the neural network model
    '''
    batch_size = 8
    STEPS = 5000
    for i in range(STEPS):
        # Choose the samples
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)
        
        # Update the parameters with the new samples
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        
        
        # print
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print("After %d training steps, cross entropy on all data is %g" % (i, total_cross_entropy))
        
       
    '''
    Show the result
    ''' 
    print(sess.run(w1))
    print(sess.run(w2))