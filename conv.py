import tensorflow as tf
import sys
import time
import os.path

# Set train_nn to True for training the neural network or False for performing recognition. 
train_nn = True

if train_nn == False :
  print 'Process: recognition.'
  # If the following is set to True, training will start if no checkpoint is found in the
  # current directory.
  countine_training_if_ckpt_not_found = True
else :
  print 'Process: training.'
  continue_training_using_previous_model = True
  if continue_training_using_previous_model == True :
    print 'Continue training the neural network using prexisiting model.'

sess = tf.InteractiveSession()

L=200
lx=4 #=int(raw_input('lx'))
V4d=lx*lx*lx*L # 4d volume

training=2000  #=int(raw_input('training'))
bsize=50 #=int(raw_input('bsize'))

# how does the data look like
Ntemp=41 #int(raw_input('Ntemp'))   #20 # number of different temperatures used in the simulation
samples_per_T=500  #int(raw_input('samples_per_T'))  #250 # number of samples per temperature value
samples_per_T_test=500 # int(raw_input('samples_per_T'))  #250 # number of samples per temperature value

numberlabels=2

# Set the following to True for using Juan's input_data.py or False for using Kelvin's data_reader.py.
use_input_data_py = False
if use_input_data_py :
    import input_data
    mnist = input_data.read_data_sets(numberlabels,lx,L,'txt', one_hot=True)

else :
    import data_reader
    import numpy as np
    filename = './N4x4x4_L200_U9_Mu0_T_shuffled_%.2d.HSF.stream'
    filenumber = np.arange(1,41,1)
    HSF = data_reader.insert_file_info(filename,filenumber)
    mnist = HSF.categorize_data()
    #mnist = HSF.categorize_dose_of_data()
    dtau = np.array([0.060, 0.075, 0.090, 0.105, 0.120, 0.135, 0.150, 0.165, \
                 0.180, 0.195, 0.210, 0.225, 0.240, 0.255, 0.270, 0.285, \
                 0.300, 0.315, 0.330, 0.345, 0.510, 0.660, 0.810, \
                 0.960, 1.110, 1.260, 1.410, 1.560, 1.710, 1.860, 2.010, \
                 2.160, 2.310, 2.460, 2.610, 2.760, 2.910, 3.060, 3.210, \
                 3.360])

n_train_data = float(len(mnist.train.labels))

print "reading sets ok"

#sys.exit("pare aqui")

# defining weighs and initlizatinon
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# defining the convolutional and max pool layers
def conv3d(x, W):
  return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='VALID')

# defining the model

x = tf.placeholder("float", shape=[None, (lx)*(lx)*(lx)*L]) # placeholder for the spin configurations
#x = tf.placeholder("float", shape=[None, lx*lx*2]) #with padding and no PBC conv net
y_ = tf.placeholder("float", shape=[None, numberlabels])


#first layer 
# convolutional layer # 2x2x2 patch size, 2 channel (2 color), 64 feature maps computed
nmaps1=64
spatial_filter_size=2
W_conv1 = weight_variable([spatial_filter_size, spatial_filter_size, spatial_filter_size,L,nmaps1])
# bias for each of the feature maps
b_conv1 = bias_variable([nmaps1])

# applying a reshape of the data to get the two dimensional structure back
#x_image = tf.reshape(x, [-1,lx,lx,2]) # #with padding and no PBC conv net
x_image = tf.reshape(x, [-1,lx,lx,lx,L]) # with PBC 

#We then convolve x_image with the weight tensor, add the bias, apply the ReLU function, and finally max pool.

h_conv1 = tf.nn.relu(conv3d(x_image, W_conv1) + b_conv1)

h_pool1=h_conv1

#In order to build a deep network, we stack several layers of this type. The second layer will have 8 features for each 5x5 patch. 

# weights and bias of the fully connected (fc) layer. Ihn this case everything looks one dimensiona because it is fully connected
nmaps2=64

#W_fc1 = weight_variable([(lx/2) * (lx/2) * nmaps1,nmaps2 ]) # with maxpool
W_fc1 = weight_variable([(lx-1) * (lx-1)*(lx-1)*nmaps1,nmaps2 ]) # no maxpool images remain the same size after conv

b_fc1 = bias_variable([nmaps2])

# first we reshape the outcome h_pool2 to a vector
#h_pool1_flat = tf.reshape(h_pool1, [-1, (lx/2)*(lx/2)*nmaps1]) # with maxpool

h_pool1_flat = tf.reshape(h_pool1, [-1, (lx-1)*(lx-1)*(lx-1)*nmaps1]) # no maxpool
# then apply the ReLU with the fully connected weights and biases.
h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)

# Dropout: To reduce overfitting, we will apply dropout before the readout layer. We create a placeholder for the probability that a neuron's output is kept during dropout. This allows us to turn dropout on during training, and turn it off during testing. TensorFlow's tf.nn.dropout op automatically handles scaling neuron outputs in addition to masking them, so dropout just works without any additional scaling.

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout layer. Finally, we add a softmax layer, just like for the one layer softmax regression above.

# weights and bias
W_fc2 = weight_variable([nmaps2, numberlabels])
b_fc2 = bias_variable([numberlabels])

# apply a softmax layer
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


#Train and Evaluate the Model
# cost function to minimize
if use_input_data_py :
  #cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
  cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0)))
else :
  cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

#sess = tf.Session()
sess.run(tf.initialize_all_variables())

filename_weight_bias = "./model_new.ckpt"

# Check to see if the checkpoint is located in the current file directory before restoring.
if train_nn == False :
  if os.path.isfile(filename_weight_bias) == False and countine_training_if_ckpt_not_found :
    print '%s is not found in the current directory, starting training...' % filename_weight_bias
    train_nn = True
  else : 
    while not(os.path.isfile(filename_weight_bias)) :
      print '%s is not found in the current directory.' %filename_weight_bias
      filename_weight_bias = raw_input('Input checkpoint model: ')
      filename_weight_bias = './' + filename_weight_bias
    train_nn = False

start_time = time.time()

if train_nn :

  if continue_training_using_previous_model == True :
    skip = 'n'
    file_exist = os.path.isfile(filename_weight_bias)
    while ( not(file_exist) and skip == 'n' ) :
      print '%s is not found in the current directory.' %filename_weight_bias
      skip = raw_input('Hit y to input checkpoint model filename or n to start training from scratch: ')
      while skip not in ['y','n'] :
        skip = raw_input('Hit y to input checkpoint model filename or n to start training from scratch: ')
      if skip == 'y' :
        file_exist = False
      else :
        filename_weight_bias = raw_input('Input checkpoint model: ')
        while not(os.path.isfile(filename_weight_bias)) :
          print '%s is not found in the current directory.' %filename_weight_bias
          filename_weight_bias = raw_input('Input checkpoint model: ')
          filename_weight_bias = './' + filename_weight_bias     
        file_exist = True
        if file_exist :
          skip = 'y'
    if file_exist :
      saver = tf.train.Saver([W_conv1, b_conv1, W_fc1,b_fc1,W_fc2,b_fc2])
      saver.restore(sess, filename_weight_bias)

  print 'Total number of training epochs: %g' %(training/n_train_data)
  start_time = time.time()

  test_accuracy_tmp = 0
  filename_measure = "./HSF_measure.dat"
  Table_measure = np.zeros(( int(training/100), 4))

  for i in range(training):
    batch = mnist.train.next_batch(bsize)
    if i%100 == 0:
      train_accuracy = sess.run(accuracy,feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
      test_accuracy = sess.run(accuracy, feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
      Cost = sess.run(cross_entropy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
      print "%.2fs, epoch %.2f, training accuracy %g, test accuracy %g, cost %g"%(time.time()-start_time,i/n_train_data, train_accuracy, test_accuracy, Cost)
      Table_measure[i/100,0] = i/n_train_data
      Table_measure[i/100,1] = train_accuracy
      Table_measure[i/100,2] = test_accuracy
      Table_measure[i/100,3] = Cost
      # To avoid multiple training, the model is saved when the difference between testing 
      # accuracy and training accuracy doesn't exceed a set value (it is set to 0.05 here)
      # and if the current testing accuracy is higher than the previous. 
      delta_accuracy = abs(train_accuracy - test_accuracy)
      if test_accuracy > test_accuracy_tmp :
        test_accuracy_tmp = test_accuracy
        if delta_accuracy <= 0.05 :
          saver = tf.train.Saver([W_conv1, b_conv1, W_fc1,b_fc1,W_fc2,b_fc2])
          save_path = saver.save(sess, filename_weight_bias)
          check_model = tf.reduce_mean(W_conv1).eval()
          best_epoch = i/n_train_data
       #print "test accuracy %g"%sess.run(accuracy, feed_dict={
       #x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}) 
       #print "test Trick accuracy %g"%sess.run(accuracy, feed_dict={
       #x: mnist.test_Trick.images, y_: mnist.test_Trick.labels, keep_prob: 1.0})  
    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

  # Final check to save the best model.
  train_accuracy = sess.run(accuracy,feed_dict={
      x:batch[0], y_: batch[1], keep_prob: 1.0})
  test_accuracy = sess.run(accuracy, feed_dict={
      x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
  Cost = sess.run(cross_entropy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
  delta_accuracy = abs(train_accuracy - test_accuracy)
  if test_accuracy > test_accuracy_tmp :
    if delta_accuracy <= 0.05 :
      saver = tf.train.Saver([W_conv1, b_conv1, W_fc1,b_fc1,W_fc2,b_fc2])
      save_path = saver.save(sess, filename_weight_bias)
      check_model = tf.reduce_mean(W_conv1).eval()
      best_epoch = training/n_train_data

  print "epoch %.2f, training accuracy %g, test accuracy %g, cost %g"%(i/n_train_data, train_accuracy, test_accuracy, Cost)

  print 'Best training epoch: %g'%best_epoch

  print "Model saved in file: ", save_path

  # To proceed, load the best (saved) model instead of the last training model.
  saver.restore(sess, filename_weight_bias)

  # Check if the saved model and the restored model are the same.
  if check_model != tf.reduce_mean(W_conv1).eval() :
    print 'Warning! Best training model and the restored model is incompatible. Exiting...'
    sys.exit()

  # Save the measurements:
  # first column : Training epochs
  # second column: Training accuracy
  # third column : Testing accuracy
  # fourth column: Cost
  np.savetxt(filename_measure, Table_measure)

else :
  saver = tf.train.Saver([W_conv1, b_conv1, W_fc1,b_fc1,W_fc2,b_fc2])

  # To proceed, load the best (saved) model instead of the last training model.
  saver.restore(sess, filename_weight_bias)

print 'Performing recognition...'
if use_input_data_py :

    #producing data to get the plots we like

    f = open('nnout.dat', 'w')

    #output of neural net
    ii=0
    for i in range(Ntemp):
      av=0.0
      for j in range(samples_per_T_test):
            batch=(mnist.test.images[ii,:].reshape(1,lx*lx*lx*L),mnist.test.labels[ii,:].reshape((1,numberlabels)))
            res=sess.run(y_conv,feed_dict={x: batch[0], y_: batch[1],keep_prob: 1.0})
            av=av+res
            #print ii, res
            ii=ii+1
      av=av/samples_per_T_test
      f.write(str(i)+' '+str(av[0,0])+' '+str(av[0,1])+"\n") 
    f.close() 


    f = open('acc.dat', 'w')

    # accuracy vs temperature
    for ii in range(Ntemp):
      batch=(mnist.test.images[ii*samples_per_T_test:ii*samples_per_T_test+samples_per_T_test,:].reshape(samples_per_T_test,L*lx*lx*lx), mnist.test.labels[ii*samples_per_T_test:ii*samples_per_T_test+samples_per_T_test,:].reshape((samples_per_T_test,numberlabels)) )
      train_accuracy = sess.run(accuracy,feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
      f.write(str(ii)+' '+str(train_accuracy)+"\n")
    f.close()
  
else :

    # Both the output of neural network and accuracy will be saved in one single file. The first
    # column has the temperature, the second holds the output of the second output neuron,
    # the third holds the output of the first neuron, the fourth holds the accuracy, and the last
    # holds the number of test data used for each temperature.
    Table = np.zeros(( len(dtau), 5))
    Table[:,0] = dtau

    for i in range(len(mnist.test.temps)) :
        # Output of neural net vs temperature
        Table[mnist.test.temps[i],1] += np.argmax(sess.run(y_conv, feed_dict={x: mnist.test.images[i,:].reshape(1,V4d), keep_prob: 1.0}))
        # Accuracy vs temperature
        Table[mnist.test.temps[i],3] += sess.run(accuracy, feed_dict={x: mnist.test.images[i,:].reshape(1,V4d), y_: mnist.test.labels[i,:].reshape(1,numberlabels), keep_prob: 1.0})
        Table[mnist.test.temps[i],-1] += 1

    Table[:,1] = Table[:,1]/Table[:,-1].astype('float')
    Table[:,2] = 1.0-Table[:,1]
    Table[:,3] = Table[:,3]/Table[:,-1].astype('float')

    filename_result = "./result.dat"
    np.savetxt(filename_result, Table)
    print "Result saved in file: ", filename_result

#producing data to get the plots we like

#f = open('nnoutTrick.dat', 'w')

#output of neural net
#ii=0
#for i in range(Ntemp):
#  av=0.0
#  for j in range(samples_per_T_test):
#        batch=(mnist.test_Trick.images[ii,:].reshape((1,2*lx*lx)),mnist.test_Trick.labels[ii,:].reshape((1,numberlabels)))
#        res=sess.run(y_conv,feed_dict={x: batch[0], y_: batch[1],keep_prob: 1.0})
#        av=av+res
#        #print ii, res
#        ii=ii+1
#  av=av/samples_per_T_test
#  f.write(str(i)+' '+str(av[0,0])+' '+str(av[0,1])+"\n")
#f.close()


#f = open('accTrick.dat', 'w')

# accuracy vs temperature
#for ii in range(Ntemp):
#  batch=(mnist.test_Trick.images[ii*samples_per_T_test:ii*samples_per_T_test+samples_per_T_test,:].reshape(samples_per_T_test,2*lx*lx), mnist.test_Trick.labels[ii*samples_per_T_test:ii*samples_per_T_test+samples_per_T_test,:].reshape((samples_per_T_test,numberlabels)) )
#  train_accuracy = sess.run(accuracy,feed_dict={
#        x:batch[0], y_: batch[1], keep_prob: 1.0})
#  f.write(str(ii)+' '+str(train_accuracy)+"\n")
#f.close()
  
