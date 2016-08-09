# Custom library
import data_reader

# Standard libraries
import datetime
import os
import sys
import time

# Third-party libraries
import numpy as np
import tensorflow as tf

sess = tf.InteractiveSession()

# Short form of Boolean value
T, F = True, False

# Training/ classification setting ---------------------------------------------------

# This code can be used to train the neural network by setting train_neural_network
# to (T) or it can also be used for performing classification using labelled data
# by setting train_neural_network to (F) and perform_classification_with_label to
# (T) or it can also be used for performing classification using raw unlabelled
# data by setting train_neural_network to (F) and perform_classification_with_label
# to (F).
train_neural_network = T
continue_training_using_trained_model = T

# Select (T) to use one U for training or (F) to use data set with 2 U for training.
use_single_U = True

# Number of training epoch
epochs = 100
# Size of training batch
batch_size = 200
# Threshold of difference between train_accuracy and test_accuracy
delta_accuracy_threshold = 0.025
# Threshold for stopping training when overtraining is encountered.
overtraining_threshold = 10

# Classification can be performed on labelled or raw data. Set
# perform_classification_with_label to (F) to perform classification on labelled
# data or (T) to perform classification on raw data. When train_neural_network is
# set to (T), perform_classification_with_label is set to (T) automatically and
# clasification will be done on labelled data.
perform_classification_with_label = T

# System and file information --------------------------------------------------------

if use_single_U :
    # Potential energy
    U = 9
else :
    # Potential energy 1 
    U1 = 4
    # Potential energy 2
    U2 = 20

# System size
#   number of spin in each of the cube dimension
n_x = 4
#   number of imaginary time dimension
L = 200
#   Volume of tesseract
V4d = L*(n_x)**3

# Critical temperature
# Tc = 0.36

# Number of data per file
ndata_per_temp = 1000

# Number of classification data (raw unlabelled data) to be used for classification. 
classification_data_per_temp = 250

# String of current date and time
dt = datetime.datetime.now()
year, month, day, hour, minute = '%.2d' % dt.year, '%.2d' % dt.month, '%.2d' % dt.day, '%.2d' % dt.hour, '%.2d' % dt.minute
start_date_time = '%s%s%s-%s%s' % (year, month, day, hour, minute)

if use_single_U :
    # Input labelled and shuffled filename for training and performaing classification
    # with labels.
    filename = './N%dx%dx%d_L%d_U%d_Mu0_T_shuffled' % (n_x,n_x,n_x,L,U) + '_%.2d.dat'

    # Input raw filename for performing classification without labels.
    rawdata_filename       = './N%dx%dx%d_L%d_U%d_Mu0_T' % (n_x,n_x,n_x,L,U) + '%s.HSF.stream'
else :
    # Input labelled and shuffled filename for training and performaing classification
    # with labels.
    if U1 < U2 :
        filename = './N%dx%dx%d_L%d_U%d+U%d_Mu0_T_shuffled' % (n_x,n_x,n_x,L,U1,U2) + '_%.2d.dat'
    else :
        filename = './N%dx%dx%d_L%d_U%d+U%d_Mu0_T_shuffled' % (n_x,n_x,n_x,L,U2,U1) + '_%.2d.dat'

# Trained model
filename_trained_model = "./model.ckpt" 

name_output_file_by_date_first = F
if name_output_file_by_date_first == False : 
    # Output model filename
    filename_weight_bias   = "./model_" + start_date_time + ".ckpt"

    # Output of training measurements filename
    filename_measure       = "./measurements_" + start_date_time + ".dat"

    # Output of classification result with labels
    filename_result        = "./result_" + start_date_time + ".dat"

    # Output of classification result from raw data (without labels)
    filename_classified    = "./classified_" + start_date_time + ".dat"
else :
    # Output model filename
    filename_weight_bias   = "./" + start_date_time + "_model.ckpt"

    # Output of training measurements filename
    filename_measure       = "./" + start_date_time + "_measurements.dat"

    # Output of classification result with labels
    filename_result        = "./" + start_date_time + "_result.dat"

    # Output of classification result from raw data (without labels)
    filename_classified    = "./" + start_date_time + "_classified.dat"

# Neural network architecture settings -----------------------------------------------

# Feature extraction layer(s)
n_feature_map1 = 64

# Classification layer
n_fully_connected_neuron = 64
n_output_neuron = 2

# Spatial filter size: filter depth, height, and width
filter_d = 2
filter_h = filter_d
filter_w = filter_d

# Optimizer learning rate
eta = 1e-4






# Don't change any of the following variables.
if use_single_U :
  perform_classification = T
else :
  perform_classification = F

if train_neural_network == T :
  print 'Process: training.'

  if not(use_single_U) :
    print 'Using data set from two Us for training.'

  # xxxxx Don't change the following variable. xxxxx
  # Perform classification on test data after training.
  perform_classification_with_label = T
  # xxxxx Don't change the following variable. xxxxx
  # When load_test_data_only is set to False, both training and test data are
  # to be loaded.
  load_test_data_only = F

else :
  # Set continue_training_if_model_not_found to T to proceed with training when no 
  # previously trained model is found in the directory or set it to F to input the
  # appropriate model filename.
  continue_training_if_model_not_found = F

  # xxxxx Don't change the following variable. xxxxx
  if perform_classification_with_label == T :
    print 'Process: classification with label.'
  else :
    print 'Process: classification without label.'
    print 'Using %d data per temperature.' % classification_data_per_temp
  # xxxxx Don't change the following variable. xxxxx
  # When load_test_data_only is set to True, only test data will be loaded.
  load_test_data_only = T

if perform_classification_with_label == True and perform_classification == True :
  # Get temperature and save them to a file.
  os.system("ls -l N%dx%dx%d_L%d_U%d_Mu0_T*.HSF.stream | awk '{print $9}' | sed -e s/N%dx%dx%d_L%d_U%d_Mu0_T//g -e s/.HSF.stream//g > dtau.dat" %(n_x,n_x,n_x,L,U,n_x,n_x,n_x,L,U))
  dtau = np.genfromtxt("dtau.dat")
  os.remove("dtau.dat")
  # Array of shuffled file's file number 
  filenumber = np.arange(1,len(dtau)+1,1)
  # Provide file information to the data_reader module.
  HSF = data_reader.insert_file_info(filename,filenumber, load_test_data_only=load_test_data_only)
  # Load and catogorize data into either training data, test data, validation data, or 
  # all of them. If validation data is needed, set include_validation_data to (T)
  # in the insert_file_info() module above.
  HSF = HSF.categorize_data()

elif perform_classification_with_label == False and perform_classification == True :
  # Get temperature and save them to a file.
  os.system("ls -l N%dx%dx%d_L%d_U%d_Mu0_T*.HSF.stream | awk '{print $9}' | sed -e s/N%dx%dx%d_L%d_U%d_Mu0_T//g -e s/.HSF.stream//g > dtau.dat" %(n_x,n_x,n_x,L,U,n_x,n_x,n_x,L,U))
  # Load temperature into a list of string
  dtau = np.genfromtxt("dtau.dat",dtype='str')
  # The number of lines to skip at the beginning of the file if not all of the data is
  # to be loaded.
  sh = ndata_per_temp  - classification_data_per_temp
  while ( ndata_per_temp - sh - classification_data_per_temp ) < 0 :
    print 'Sum of classification data per temperature and the number of lines skip at the beginning of the file must be equal to number of data per temnperature.'
    print 'Number of data per temnperature                  : %d' % ndata_per_temp
    print 'Classification data used per temperature         : %d' % classification_data_per_temp
    print 'Number of lines skip at the beginning of the file: %d' % sh
    classification_data_per_temp = input('Input new classification data used per temperature: ')
    sh = input('Input number of lines skip at the beginning of the file: ')
  # Provide file information to the data_reader module.
  HSF = data_reader.insert_file_info(rawdata_filename,dtau)
  # Load classification data.
  HSF = HSF.load_classification_data(nrows=ndata_per_temp, ncols=n_x*n_x*n_x*L, 
        SkipHeader=sh, load_ndata_per_file=classification_data_per_temp)

# For training data with two different Us
if perform_classification == False :
  # Get file numbers.
  if U1 < U2 :
    os.system("ls -l N%dx%dx%d_L%d_U%d+U%d_Mu0_T_shuffled_*.dat | awk '{print $9}' | sed -e s/N%dx%dx%d_L%d_U%d+U%d_Mu0_T_shuffled_//g -e s/.dat//g > filenumber.dat" %(n_x,n_x,n_x,L,U1,U2,n_x,n_x,n_x,L,U1,U2))
  else :
    os.system("ls -l N%dx%dx%d_L%d_U%d+U%d_Mu0_T_shuffled_*.dat | awk '{print $9}' | sed -e s/N%dx%dx%d_L%d_U%d+U%d_Mu0_T_shuffled_//g -e s/.dat//g > filenumber.dat" %(n_x,n_x,n_x,L,U2,U1,n_x,n_x,n_x,L,U2,U1))

  # Array of shuffled file's file number 
  filenumber = np.genfromtxt("filenumber.dat")
  if np.mod(len(filenumber),2) == 1:
    print 'Attention! When using data set with two Us for training, make sure that there are EVEN number of files. Exiting...'
    sys.exit() 
  os.remove("filenumber.dat")
  # Array of shuffled file's file number 
  #filenumber = np.arange(1,len(dtau)+1,1)
  # Provide file information to the data_reader module.
  HSF = data_reader.insert_file_info(filename,filenumber)
  # Load and catogorize data into either training data, test data, validation data, or 
  # all of them. If validation data is needed, set include_validation_data to (T)
  # in the insert_file_info() module above.
  HSF = HSF.categorize_data()





if train_neural_network == T or not(load_test_data_only) :
  n_train_data = len(HSF.train.labels)

  while np.modf(float(n_train_data)/batch_size)[0] > 0.0 :
    print 'Warning! Number of data/ batch size must be an integer.'
    print 'Number of data: %d' % n_train_data
    print 'Batch size    : %d' % batch_size
    batch_size = int(input('Input new batch size: '))

  # Number of training cycle per training epoch
  iteration_per_epoch=n_train_data/batch_size

  print 'Number of training data: %d' % n_train_data
elif train_neural_network == F and perform_classification_with_label :
  n_train_data = len(dtau)*ndata_per_temp - len(HSF.test.labels)
  print 'Number of training data: %d' % n_train_data
else :
  n_train_data = -1
  while n_train_data < 0 :
    n_train_data = input('Number of training data used:' )





# x is a 2D-tensor and None means that a dimesion can be of any length,
# but in this case, it corresponds to the batch size. To start building 
# the computation graph, we'll create nodes for input images and target 
# output classes. The target output classes y_ will consist of a 2D 
# tensor, where each row is a one-hot (one-hot refers to a groups of 
# bits among which only one is (1), the opposite is called one-cold) 2
# -dimensional vector vector indicating which digit class the 
# corresponding HSF data belongs to.
x = tf.placeholder(tf.float32, [None, n_x*n_x*n_x * L])
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

def conv3d(x, W, pad='VALID'):
    # The convolutions uses a stride of one and are zero padded so that
    # the output is the same size as the input.
    # tf.nn.conv3d(input, filter, strides, padding, name=None)
    return tf.nn.conv3d(x, W, strides = [1,1,1,1,1], padding=pad)

def max_pool_2x2x2(x, pad='SAME'):
    # Max pooling over 2x2 blocks.
    # tf.nn.max_pool(input, ksize, strides, padding, 
    #                data_format='NHWC', name=None)
    # input   : shape [batch, depth, rows, cols, channels]
    # ksize   : The size of the max pool window for each dimension of the
    #           input tensor
    # strides : The stride of the sliding window for each dimension of
    #           the input tnesor.
    return tf.nn.max_pool3d(x, ksize=[1,2,2,2,1], 
                          strides=[1,2,2,2,1], padding=pad)

# Feature extraction layer -----------------------------------------------------------

# First Convolution Layer
# The convolution will compute n features for each mxmxm block. Its weight
# tensor will have a shape of [filter_Depth, filter_height, filter_width, 
# in_channels, out_channels].
W_conv1 = weight_variable([filter_d,filter_h,filter_w,L,n_feature_map1])
b_conv1 = bias_variable([n_feature_map1])

# To apply the layer, first reshape x to a 4D tensor, with the second and
# third dimensions correspondings to image width and height, and the final
# dimension corresponding to the number of color channels.
x_image = tf.reshape(x, [-1,n_x,n_x,n_x,L])

# Then convolve x_image with the weight tensor, add the bias, apply the
# ReLU function. Since no padding is used in conv3d, i.e. padding = 'VALID',
# the output size : n_feature_map1 x n_x-1 x n_x-1 x n_x-1
h_conv1 = tf.nn.relu(conv3d(x_image, W_conv1) + b_conv1)

# Classification layer ---------------------------------------------------------------

# Fully-connected Layer
# Now add a fully-connected layer with n_fully_connected_neuron neurons to 
# allow processing on the entire image. The tensor from the previous layer
# is reshaped into a batch of vectors, multiply by a weight matrix, add a
# bias, and apply a ReLU.
W_fc1 = weight_variable([n_feature_map1*(n_x-1)**3, n_fully_connected_neuron])
b_fc1 = bias_variable([n_fully_connected_neuron])

h_conv1_flat = tf.reshape(h_conv1, [-1, n_feature_map1*(n_x-1)**3])
h_fc1 = tf.nn.relu(tf.matmul(h_conv1_flat, W_fc1) + b_fc1)


# Dropout
# To reduce overfitting, dropout will be applied before the readout layer.
# We'll create a placeholder for the probability that a neuron's output is
# kept during dropout. This allows us tro turn dropout on during training, and
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

# Train and Evaluate the Model
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(eta).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Before Variables can be used within a session, they must be initialized
# using that session.
sess.run(tf.initialize_all_variables())





# When performing classification, check if the trained model checkpoint file is 
# located in the current file directory before restoring.
if train_neural_network == False :
  if os.path.isfile(filename_trained_model) == False and continue_training_if_model_not_found :
    print '%s is not found in the current directory, starting training...' % filename_trained_model.replace('./','')
    train_neural_network = True
    continue_training_using_trained_model = False
  else :
    while not(os.path.isfile(filename_trained_model)) :
      print '%s is not found in the current directory.' % filename_trained_model.replace('./','')
      filename_trained_model = raw_input('Input trained model filename: ')
      filename_trained_model = './' + filename_trained_model
    train_neural_network = False

# Training ---------------------------------------------------------------------------

# Training the model can be accomplished by repeatedly running train_step. Each
# training iteration load n training examples. Then, the train_step operation can be
# run using feed_dict to replace the placeholder tensors x and y_ with the training
# examples. Note: any tensor in the computation graph can be replcaed using feed_dict.

start_time = time.time()

if train_neural_network :

  # Check if the trained model checkpoint file is located in the current file directory
  # before restoring.
  if continue_training_using_trained_model :
    skip = False
    file_exist = os.path.isfile(filename_trained_model)
    while (not(file_exist) and not(skip)) :
      print '%s is not found in the current directory.' % filename_trained_model.replace('./','')
      skip = raw_input('Select T to start training from scratch or F to enter the filename of another trained model: ')
      while skip not in ['T','F']:
        skip = raw_input('Select T or F: ')
      if skip == 'T' :
        skip = True
      else :
        skip = False
      if skip :
        file_exist = False
      else :
        filename_trained_model = raw_input('Input trained model filename: ')
        while not(os.path.isfile(filename_trained_model)) :
          print '%s is not found in the current directory.'% filename_trained_model.replace('./','')
          filename_trained_model = raw_input('Input trained model filename: ')
          filename_trained_model = './' + filename_trained_model
        if os.path.isfile(filename_trained_model) :
          skip = True

    if file_exist :
      print 'Continue training using %s.' % filename_trained_model.replace('./','')
      saver = tf.train.Saver([W_conv1, b_conv1, W_fc1, b_fc1, W_fc2, b_fc2])
      # Restore trained model.
      save_path = saver.restore(sess, filename_trained_model)

  # Initialize best test accuracy. 
  best_test_accuracy = 0.85

  # Calculate the number of data to collect for the whole training cycle.
  ndata_collect_per_epoch = round(float(n_train_data)/batch_size/100)
  if ndata_collect_per_epoch > 1 :
    ndata_collect = int(ndata_collect_per_epoch*epochs)
  else :
    ndata_collect = int(epochs)
  
  # Initialise data table.
  # First column : Training epochs
  # Second column: Training accuracy
  # Third column : Testing accuracy
  # Fourth column: Cost
  Table_measure = np.zeros(( ndata_collect, 4))

  # Initialise the counter for number of data collected.
  n = 0
  fractional_epoch = batch_size*100/float(n_train_data)

  print 'Total number of training epochs: %.1f' % (ndata_collect*fractional_epoch)

  # Initialise counter for checking overtraining/ overfitting.
  n_overtraining_counter = 0
  m = 0
  Overtraining = False

  for j in range(epochs):
    # Break out of the training epoch loop if overtraining is encountered.
    if Overtraining :
      break
    for i in range(iteration_per_epoch):
      batch = HSF.train.next_batch(batch_size)
      if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
                         x: batch[0], y_: batch[1], keep_prob: 1.0})
        test_accuracy = accuracy.eval(feed_dict={
                         x: HSF.test.images, y_: HSF.test.labels, keep_prob: 1.0})
        Cost = cross_entropy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print '%.2fs, epoch %.2f, training accuracy %g, test accuracy %g, cost %g' % (time.time()-start_time,(n+1)*fractional_epoch, train_accuracy, test_accuracy, Cost)
        Table_measure[n,0] = n*fractional_epoch
        Table_measure[n,1] = train_accuracy
        Table_measure[n,2] = test_accuracy
        Table_measure[n,3] = Cost
        # To avoid multiple training, the model is saved when the difference between testing 
        # accuracy and training accuracy doesn't exceed a set value (it is set to 0.05 here)
        # and if the current testing accuracy is higher than the previous. 
        delta_accuracy = train_accuracy - test_accuracy
        if (test_accuracy > best_test_accuracy) and (delta_accuracy <= delta_accuracy_threshold) and (delta_accuracy > 0) :
          # Update the best test accuracy
          best_test_accuracy = test_accuracy
          # Save the best model thus far if the above two criteria are met.
          print 'Saving model %s and measurements %s.' % (filename_weight_bias.replace('./',''), filename_measure.replace('./',''))
          saver = tf.train.Saver([W_conv1, b_conv1, W_fc1, b_fc1, W_fc2, b_fc2])
          save_path = saver.save(sess, filename_weight_bias)
          check_model = tf.reduce_mean(W_conv1).eval()
          best_epoch = n*fractional_epoch
          np.savetxt(filename_measure, Table_measure[:n+1,:])
        # Check for overtraining/ overfitting. If so, stop training and break out of the
        # training iteration per epoch loop.
        if train_accuracy > test_accuracy and test_accuracy > 0.8 :
          if m == 0 :
            # If training accuracy is greater than test accuracy on first account, set
            # counter to 1.
            n_overtraining_counter = 1
            m = n
          elif (m+1) == n :
            # If training accuracy is greater than test accuracy consecutively, increase
            # counter.
            n_overtraining_counter += 1
            m = n
          elif (m+1) < n :
            # Reset counter otherwise. 
            n_overtraining_counter = 0
            m = 0
        if n_overtraining_counter >= overtraining_threshold or np.isnan(Cost):
          print 'Overtraining encountered. Stopping training.'
          Table_measure = Table_measure[:n+1,:]
          Overtraining = True
          break
        else :
          Overtraining = False
        n += 1
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

  if not(Overtraining) :
    # Final check to save the best model.
    train_accuracy = accuracy.eval(feed_dict={
      x: batch[0], y_: batch[1], keep_prob: 1.0})
    test_accuracy = accuracy.eval(feed_dict={
      x: HSF.test.images, y_: HSF.test.labels, keep_prob: 1.0})
    Cost = cross_entropy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})

    delta_accuracy = abs(train_accuracy - test_accuracy)
    if test_accuracy > best_test_accuracy :
      if (delta_accuracy <= delta_accuracy_threshold) and delta_accuracy > 0 :
        print 'Saving model...' 
        saver = tf.train.Saver([W_conv1, b_conv1, W_fc1, b_fc1, W_fc2, b_fc2])
        save_path = saver.save(sess, filename_weight_bias)
        check_model = tf.reduce_mean(W_conv1).eval()
        best_epoch = ndata_collect*fractional_epoch

    print '%.2fs, epoch %.2f, training accuracy %g, test accuracy %g, cost %g' % (time.time()-start_time,(n+1)*fractional_epoch, train_accuracy, test_accuracy, Cost)

  print 'Best training epoch: %g' % best_epoch

  print 'Model saved in file: ', save_path

  # To proceed, load the best (saved) model instead of the last training model.
  saver.restore(sess, filename_weight_bias)

  # Check if the saved model and the restored model are the same.
  if check_model != tf.reduce_mean(W_conv1).eval() :
    print 'Warning! Best training model and the restored model is incompatible. Exiting...'
    sys.exit()

  # Save the measurements:
  # First column : Training epochs
  # Second column: Training accuracy
  # Third column : Testing accuracy
  # Fourth column: Cost
  np.savetxt(filename_measure, Table_measure)

# Classification ---------------------------------------------------------------------

else :
  saver = tf.train.Saver([W_conv1, b_conv1, W_fc1, b_fc1, W_fc2, b_fc2])
  # To proceed, load the trained model.
  saver.restore(sess, filename_trained_model)

if perform_classification :
  print 'Performing classification using %s.' % filename_trained_model.replace('./','')

  # Classification with labels ---------------------------------------------------------

  if perform_classification_with_label == True :

    # First column : Temperature
    # Second column: Average classified output of the second neuron
    # Third column : Average classified output of the first neuron
    # Fourth column: Classification accuracy
    # Fifth column : Number of data used
    Table = np.zeros(( len(dtau), 5))
    Table[:,0] = dtau

    for i in range(len(HSF.test.temps)) :
      # Output of neural net vs temperature
      Table[HSF.test.temps[i],1] += np.argmax(y_conv.eval(feed_dict={x: HSF.test.images[i,:].reshape(1,V4d), keep_prob: 1.0}))
      # Accuracy vs temperature
      Table[HSF.test.temps[i],3] += accuracy.eval(feed_dict={x: HSF.test.images[i,:].reshape(1,V4d), y_: HSF.test.labels[i,:].reshape(1,n_output_neuron), keep_prob: 1.0})
      Table[HSF.test.temps[i],-1] += 1

      #data = HSF.test.images[i,:]
      #layer1 = h_conv1.eval(feed_dict={x: HSF.test.images[i,:].reshape(1,V4d)})
      #layer1 = layer1.reshape(1,64*27)
      #layer2 = h_fc1.eval(feed_dict={x: HSF.test.images[i,:].reshape(1,V4d)}) 
      #output = y_conv.eval(feed_dict={x: HSF.test.images[i,:].reshape(1,V4d), keep_prob: 1.0})
      #np.savetxt("input_layer.dat", data, fmt='%d')
      #np.savetxt("feature_extraction_layer.dat", layer1)
      #np.savetxt("fully_connected_layer1.dat", layer2)
      #np.savetxt("output.dat", output)
      #sys.exit()

    # Normalize the output of the second neuron
    Table[:,1] = Table[:,1]/Table[:,-1].astype('float')
    # Normalized output of the first neuron
    Table[:,2] = 1.0-Table[:,1]
    # Normalize the classification accuracy
    Table[:,3] = Table[:,3]/Table[:,-1].astype('float')

    np.savetxt(filename_result, Table)
    print 'Result saved as %s.' % filename_result

  # Classification on raw data --------------------------------------------------------

  else :

    # First column : Temperature
    # Second column: Average classified output of the second neuron
    # Third column : Average classified output of the first neuron
    # Fourth column: Number of data used 
    Table = np.zeros(( len(dtau), 4))
    Table[:,0] = dtau
    Table[:,-1] = classification_data_per_temp

    n=0 
    for j in range(len(dtau)) :
      for i in range(classification_data_per_temp) :
        # Output of neural net vs temperature
        Table[j,1] += np.argmax(y_conv.eval(feed_dict={x: HSF.classification.images[n,:].reshape(1,V4d), keep_prob: 1.0}))
        n+=1

    # Normalize the output of the second neuron
    Table[:,1] = Table[:,1]/Table[:,-1].astype('float')
    # Normalized output of the first neuron
    Table[:,2] = 1.0-Table[:,1]

    np.savetxt(filename_classified, Table)
    print 'Classified result saved as %s.' % filename_classified
