# Third-party libraries
import numpy as np
import tensorflow as tf
import sys

filename_trained_model = "./20160819-1545_model_U16_CNN0f_test_acc_92.1.ckpt"

#   number of spin in each of the cube dimension
n_x = 4
#   number of imaginary time dimension
L = 200

# Feature extraction layer(s)
n_feature_map1 = 32
n_feature_map2 = 16
n_feature_map3 = 8

# Classification layer
n_fully_connected_neuron = 8
n_output_neuron = 2

# Spatial filter size: filter depth, height, and width
filter_d = 2
filter_h = filter_d
filter_w = filter_d

sess = tf.InteractiveSession()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Feature extraction layer -----------------------------------------------------------

# First Convolution Layer
# The convolution will compute n features for each mxmxm block. Its weight
# tensor will have a shape of [filter_Depth, filter_height, filter_width, 
# in_channels, out_channels].
W_conv1 = weight_variable([filter_d,filter_h,filter_w,L,n_feature_map1])
b_conv1 = bias_variable([n_feature_map1])

# Second Convolution Layer
W_conv2 = weight_variable([filter_d,filter_h,filter_w,n_feature_map1,n_feature_map2])
b_conv2 = bias_variable([n_feature_map2])

# Third Convolution Layer
W_conv3 = weight_variable([filter_d,filter_h,filter_w,n_feature_map2,n_feature_map3])
b_conv3 = bias_variable([n_feature_map3])

# Classification layer ---------------------------------------------------------------

# Fully-connected Layer
# Now add a fully-connected layer with n_fully_connected_neuron neurons to 
# allow processing on the entire image. The tensor from the previous layer
# is reshaped into a batch of vectors, multiply by a weight matrix, add a
# bias, and apply a ReLU.
W_fc1 = weight_variable([n_feature_map3*(n_x)**3, n_fully_connected_neuron])
b_fc1 = bias_variable([n_fully_connected_neuron])

# Readout layer
# Finally, a softmax regression layer is added.
W_fc2 = weight_variable([n_fully_connected_neuron,n_output_neuron])
b_fc2 = bias_variable([n_output_neuron])

# Before Variables can be used within a session, they must be initialized
# using that session.
sess.run(tf.initialize_all_variables())

saver = tf.train.Saver([W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2])
save_path = saver.restore(sess, filename_trained_model)

# ------------------------------------------------------------------------------------

weight = W_conv3.eval()
weight_shape = np.array(np.shape(weight))
output_data = tf.reshape(weight,(np.prod(weight_shape[:-1]),weight_shape[-1])).eval()
np.savetxt("20160819-1545_model_U16_CNN0f_test_acc_92.1_W_conv3.dat", output_data)

w = W_fc2.eval()
w_shape = np.array(np.shape(w))
output_data = tf.reshape(w,(w_shape[0],w_shape[-1])).eval()
#np.savetxt("20160819-1545_model_U16_CNN0f_test_acc_92.1_W_fc2.dat", output_data)

