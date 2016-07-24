import sys
import numpy as np
import data_reader
import time

filename = '/home/kelvin/Desktop/Theano test/HSF_N4x4x4_L200_U9_Mu0_UniformTGrid/N4x4x4_L200_U9_Mu0_T_shuffled_%.2d.HSF.stream'
filename_weight_bias = "./wb.ckpt"
filename_measure = "./HSF_measure.dat"
filenumber = np.arange(1,41,1)
HSF = data_reader.insert_file_info( filename, filenumber )
HSF = HSF.categorize_data()

i_training = 6400
n_feature_map1 = 32
n_feature_map2 = 32
n_feature_map3 = 8
n_feature_map4 = 8
n_spin = 4
n_time_dimension = 200
n_output_neuron = 2
n_fully_connected_neuron = 1024

filter_d = 2
filter_h = filter_d
filter_w = filter_d

import tensorflow as tf
# If you are not using an InteractiveSession, then you should build
# the entire computation graph before starting a session and 
# launching the graph.
sess = tf.InteractiveSession()

# x is a 2D-tensor and None means that a dimesion can be of any length,
# but in this case, it corresponds to the batch size. To start building 
# the computation graph, we'll create nodes for input images and target 
# output classes. The target output classes y_ will consist of a 2D 
# tensor, where each row is a one-hot (one-hot refers to a groups of 
# bits among which only one is (1), the opposite is called one-cold) 2
# -dimensional vector vector indicating which digit class the 
# corresponding HSF data belongs to.
x = tf.placeholder(tf.float32, [None, n_spin*n_spin*n_spin * n_time_dimension])
y_ = tf.placeholder(tf.float32, [None, n_output_neuron])

# To prevent 0 gradients and break symmetry, one should genereally
# initialize weights with a small amount of noise for symmetry breaking.
# To avoid "dead neurons" when using ReLU neurons, it is also a good
# practice to initalize them with a slightly positive initial bias.
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, pad='SAME'):
    # The convolutions uses a stride of one and are zero padded so that
    # the output is the same size as the input.
    # tf.nn.conv2d(input, filter, strides, padding, 
    #              use_cudnn_on_gpu=None, data_format=None, name=None)
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding=pad)

def conv3d(x, W, pad='SAME'):
    # The convolutions uses a stride of one and are zero padded so that
    # the output is the same size as the input.
    # tf.nn.conv3d(input, filter, strides, padding, name=None)
    return tf.nn.conv3d(x, W, strides = [1,1,1,1,1], padding=pad)

def max_pool_2x2(x, pad='SAME'):
    # Max pooling over 2x2 blocks.
    # tf.nn.max_pool(value, ksize, strides, padding, 
    #                data_format='NHWC', name=None)
    # value : shape [batch, height, width, channels]
    # ksize : The size of the max pool window for each dimension of the
    #         input tensor
    # strides : The stride of the sliding window for each dimension of
    #           the input tnesor.
    return tf.nn.max_pool(x, ksize=[1,2,2,1], 
                          strides=[1,2,2,1], padding=pad)

def max_pool_2x2x2(x, pad='SAME'):
    # Max pooling over 2x2 blocks.
    # tf.nn.max_pool(input, ksize, strides, padding, 
    #                data_format='NHWC', name=None)
    # input : shape [batch, depth, rows, cols, channels]
    # ksize : The size of the max pool window for each dimension of the
    #         input tensor
    # strides : The stride of the sliding window for each dimension of
    #           the input tnesor.
    return tf.nn.max_pool3d(x, ksize=[1,2,2,2,1], 
                          strides=[1,2,2,2,1], padding=pad)

# First Convolution Layer + ReLU ======================================================

# The convolution will compute n features for each mxmxm block. Its weight
# tensor will have a shape of [filter_depth, filter_height, filter_width, 
# in_channels, out_channels]].
W_conv1 = weight_variable([filter_d,filter_h,filter_w,n_time_dimension,n_feature_map1])
b_conv1 = bias_variable([n_feature_map1])

# To apply the layer, first reshape x to a 5D tensor, with the second,
# third, and fourth dimensions correspondings to image depth, height, and 
# width, and the final dimension corresponding to the number of color channels.
x_image = tf.reshape(x, [-1,n_spin,n_spin,n_spin,n_time_dimension])

# Then convolve x_image with the weight tensor, add the bias, apply the
# ReLU function, and finally max pool. After applying max pooling, the 
# image has been reduced to 14x14.
# n_feature_map1 x n_spin x n_spin x n_spin
h_conv1 = tf.nn.relu(conv3d(x_image, W_conv1) + b_conv1)

# Second Convolution Layer + ReLU ====================================================
W_conv2 = weight_variable([filter_d,filter_h,filter_w,n_feature_map1,n_feature_map2])
b_conv2 = bias_variable([n_feature_map2])
h_conv2 = tf.nn.relu(conv3d(h_conv1, W_conv2) + b_conv2)

# Max pooling layer ==================================================================
# n_feature_map1 x n_spin/2 x n_spin/2 x n_spin/2
h_pool1 = max_pool_2x2x2(h_conv2)

# Third Convolution Layer + ReLU =====================================================
W_conv3 = weight_variable([filter_d,filter_h,filter_w,n_feature_map2,n_feature_map3])
b_conv3 = bias_variable([n_feature_map3])

# n_feature_map2 x n_spin/2 x n_spin/2 x n_spin/2
h_conv3 = tf.nn.relu(conv3d(h_pool1, W_conv3) + b_conv3)

# Fourth Convolution Layer + ReLU ====================================================
W_conv4 = weight_variable([filter_d,filter_h,filter_w,n_feature_map3,n_feature_map4])
b_conv4 = bias_variable([n_feature_map4])

# n_feature_map2 x n_spin/2 x n_spin/2 x n_spin/2
h_conv4 = tf.nn.relu(conv3d(h_conv3, W_conv4) + b_conv4)

# n_feature_map2 x n_spin/4 x n_spin/4 x n_spini/4
# h_pool2 = max_pool_2x2x2(h_conv2)

# Fully-connected Layer
# Now add a fully-connected layer with n_fully_connected_neurons to 
# allow processing on the entire image. The tensor from the convolution layer 
# is reshaped into a batch of vectors, multiply by a weight matrix, add a
# bias, and apply a ReLU.
W_fc1 = weight_variable([n_feature_map4*n_spin/2*n_spin/2*n_spin/2, n_fully_connected_neuron])
b_fc1 = bias_variable([n_fully_connected_neuron])

h_conv4_flat = tf.reshape(h_conv4, [-1, n_feature_map4*n_spin/2*n_spin/2*n_spin/2])
h_fc1 = tf.nn.relu(tf.matmul(h_conv4_flat, W_fc1) + b_fc1)

# Dropout
# To reduce overfitting, dropout will be applied before the readout layer.
# We'll create a placeholder for the probability that a neuron's output is
# kept during dropout. This allows us to turn dropout on during training, and
# turn it off during testing. TensorFlow's tf.nn.dropout op automatically 
# handles scaling neuron outputs in addition to masking them, so droput just 
# works without any additional scaling.
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout layer
# Finally, a softmax regression layer is added.
W_fc2 = weight_variable([n_fully_connected_neuron,n_output_neuron])
b_fc2 = bias_variable([n_output_neuron]) 

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# learning rate, eta
eta = 1e-4
# Train and Evaluate the Model
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer( eta ).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Before Variables can be used within a session, they must be initialized
# using that session.
sess.run(tf.initialize_all_variables())

# Training the model can therefore be accomplished by repeatedly running 
# train_step. Each training iteration load 50 training examples. Then, the
# train_step operation can be run using feed_dict to replace the placeholder
# tensors x and y_ with the training examples. Note: any tensor in the
# computation graph can be replcaed using feed_dict.
print '\n'
print 'Initialize training...'
start_time = time.time()

with open(filename_measure, "w") as f:

    for i in range(i_training):
        batch = HSF.train.next_batch()
        if i%100 == 0 :
            train_accuracy = accuracy.eval(feed_dict={
                             x: batch[0], y_: batch[1], keep_prob: 1.0})
            test_accuracy = accuracy.eval(feed_dict={
                             x: HSF.test.images, y_: HSF.test.labels, keep_prob: 1.0})
            Cost = cross_entropy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("%.2fs, step %d, training accuracy %g, test accuracy %g, cost %g"%(time.time()-start_time,i,train_accuracy, test_accuracy, Cost))
            f.write('%d %g %g %g\n'%(i,train_accuracy,test_accuracy,Cost))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: HSF.test.images, y_: HSF.test.labels, keep_prob: 1.0}))

# Save all the variables.
save_weights_and_biases = tf.train.Saver([W_conv1, b_conv1, W_conv2, b_conv2,
                                          W_conv3, b_conv3, W_conv4, b_conv4,
                                          W_fc1, b_fc1, W_fc2, b_fc2])

# Save all the variables.
save_weights_and_biases = tf.train.Saver([W_conv1, b_conv1, W_fc1, b_fc1, W_fc2, b_fc2])
save_path = save_weights_and_biases.save(sess,filename_weight_bias)



