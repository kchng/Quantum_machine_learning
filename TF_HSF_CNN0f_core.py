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

# Short form of Boolean value
T, F = True, False

class optimise_hyperparameters :

    def __init__(self, use_single_U = True, U = 5, U1 = 5, U2 = 20, name_output_file_by_date_first = True): 
        self.U = U
        self.U1 = U1
        self.U2 = U2
        self.use_single_U = use_single_U
        self.name_output_file_by_date_first = name_output_file_by_date_first

        if use_single_U :
          print 'Training using U = %d' % U
        else :
          print 'Training using U = %d and U = %d.' % (U1,U2) 

        # System size
        #   number of spin in each of the cube dimension
        self.n_x = 4
        n_x = self.n_x
        #   number of imaginary time dimension
        self.L = 200
        L = self.L
        #   Volume of tesseract
        self.V4d = L*(n_x)**3

        # Maximum number of data file to be used for training and testing.
        Max_nfile = 100
        # Offset to the file index (to load)
        File_index_offset = 0

        if use_single_U :
            # Input labelled and shuffled filename for training and performaing classification
            # with labels.
            filename = './N%dx%dx%d_L%d_U%d_Mu0_T_shuffled' % (n_x,n_x,n_x,L,U) + '_%.2d.dat'

            # Input raw filename for performing classification without labels.
            rawdata_filename = './N%dx%dx%d_L%d_U%d_Mu0_T' % (n_x,n_x,n_x,L,U) + '%s.HSF.stream'
        else :
            # Input labelled and shuffled filename for training and performaing classification
            # with labels.
            if U1 < U2 :
                filename = './N%dx%dx%d_L%d_U%d+U%d_Mu0_T_shuffled' % (n_x,n_x,n_x,L,U1,U2) + '_%.2d.dat'
            else :
                filename = './N%dx%dx%d_L%d_U%d+U%d_Mu0_T_shuffled' % (n_x,n_x,n_x,L,U2,U1) + '_%.2d.dat'

        # Load data
        if use_single_U :
          # Get temperature and save them to a file.
          os.system("ls -l N%dx%dx%d_L%d_U%d_Mu0_T*.HSF.stream | awk '{print $9}' | sed -e s/N%dx%dx%d_L%d_U%d_Mu0_T//g -e s/.HSF.stream//g > dtau.dat" %(n_x,n_x,n_x,L,U,n_x,n_x,n_x,L,U))
          self.dtau = np.genfromtxt("dtau.dat")
          os.remove("dtau.dat")
        # Array of shuffled file's file number 
        else :
          # Get temperature and save them to a file.
          os.system("ls -l N%dx%dx%d_L%d_U%d_Mu0_T*.HSF.stream | awk '{print $9}' | sed -e s/N%dx%dx%d_L%d_U%d_Mu0_T//g -e s/.HSF.stream//g > dtau1.dat" %(n_x,n_x,n_x,L,U1,n_x,n_x,n_x,L,U1))
          dtau1 = np.genfromtxt("dtau1.dat")
          # Get temperature and save them to a file.
          os.system("ls -l N%dx%dx%d_L%d_U%d_Mu0_T*.HSF.stream | awk '{print $9}' | sed -e s/N%dx%dx%d_L%d_U%d_Mu0_T//g -e s/.HSF.stream//g > dtau2.dat" %(n_x,n_x,n_x,L,U2,n_x,n_x,n_x,L,U2))
          dtau2 = np.genfromtxt("dtau2.dat")
          self.dtau = np.hstack((dtau1,dtau2))
          os.remove("dtau1.dat")
          os.remove("dtau2.dat")

        # Array of shuffled file's file number 
        filenumber = np.arange(1+File_index_offset,len(self.dtau)+1,1)
        if len(filenumber) > Max_nfile :
          filenumber = filenumber[:Max_nfile]

        # Provide file information to the data_reader module.
        HSF = data_reader.insert_file_info(filename,filenumber, load_test_data_only=False)
        # Load and catogorize data into either training data, test data, validation data, or 
        # all of them. If validation data is needed, set include_validation_data to (T)
        # in the insert_file_info() module above.
        self.HSF = HSF.categorize_data()

    def insert_hyperparameters(self, n_feature_map1, n_feature_map2, n_feature_map3, n_fully_connected_neuron, 
        continue_training_using_trained_model = False, filename_trained_model = "./model.ckpt" ):
        HSF = self.HSF
        # System size
        #   number of spin in each of the cube dimension
        n_x = self.n_x
        #   number of imaginary time dimension
        L   = self.L
        #   Volume of tesseract
        V4d = self.V4d
        U   = self.U
        U1  = self.U1
        U2  = self.U2
        dtau = self.dtau
        
        name_output_file_by_date_first = self.name_output_file_by_date_first        
 
        # Code name of the neural network
        NNetwork = 'CNN0f'

        sess = tf.InteractiveSession()

        use_single_U = self.use_single_U

        print "(Conv + ReLu) 1: %d, (Conv + ReLu) 2: %d, (Conv + ReLu) 3: %d, fc 1: %d" %(n_feature_map1, n_feature_map2, n_feature_map3, n_fully_connected_neuron)

        # Number of training epoch
        epochs = 500
        # Size of training batch
        batch_size = 100
        # Threshold of difference between train_accuracy and test_accuracy
        delta_accuracy_threshold = 0.025
        # Threshold for stopping training when overtraining is encountered.
        overtraining_threshold = 10
        # Initialize best test accuracy. The minimum test_accuracy for model and measurements
        # to be saved.
        best_test_accuracy = 0.5

        # String of current date and time
        dt = datetime.datetime.now()
        year, month, day, hour, minute = '%.2d' % dt.year, '%.2d' % dt.month, '%.2d' % dt.day, '%.2d' % dt.hour, '%.2d' % dt.minute
        start_date_time = '%s%s%s-%s%s' % (year, month, day, hour, minute)

        if name_output_file_by_date_first == False :
            if use_single_U :
                # Output model filename
                filename_weight_bias = "./model_U%d_" % U + NNetwork + "_CR1%d_CR2%d_CR3%d_fc%d" % (n_feature_map1, n_feature_map2, n_feature_map3, n_fully_connected_neuron) + "_test_acc_%.1f_" + start_date_time + ".ckpt"

                # Output of training measurements filename
                filename_measure     = "./measurements_U%d_" % U + NNetwork + "_CR1%d_CR2%d_CR3%d_fc%d" % (n_feature_map1, n_feature_map2, n_feature_map3, n_fully_connected_neuron) + "_test_acc_%.1f_" + start_date_time + ".dat"

                # Output of classification result with labels
                filename_result      = "./result_U%d_" % U + NNetwork + "_CR1%d_CR2%d_CR3%d_fc%d" % (n_feature_map1, n_feature_map2, n_feature_map3, n_fully_connected_neuron) + "_test_acc_%.1f_" + start_date_time + ".dat"

            else :
                # Output model filename
                filename_weight_bias = "./model_U%d+U%d_" % (U1,U2) + NNetwork + "_CR1%d_CR2%d_CR3%d_fc%d" % (n_feature_map1, n_feature_map2, n_feature_map3, n_fully_connected_neuron) + "_test_acc_%.1f_" + start_date_time + ".ckpt"

                # Output of training measurements filename
                filename_measure     = "./measurements_U%d+U%d_" % (U1,U2) + NNetwork + "_CR1%d_CR2%d_CR3%d_fc%d" % (n_feature_map1, n_feature_map2, n_feature_map3, n_fully_connected_neuron) + "_test_acc_%.1f_" + start_date_time + ".dat"

                # Output of classification result with labels
                filename_result      = "./result_U%d+U%d_" % (U1,U2) + NNetwork + "_CR1%d_CR2%d_CR3%d_fc%d" % (n_feature_map1, n_feature_map2, n_feature_map3, n_fully_connected_neuron) + "_test_acc_%.1f_" + start_date_time + ".dat"

        else :
            if use_single_U :
                # Output model filename
                filename_weight_bias = "./" + start_date_time + "_model_U%d_" % U + NNetwork + "_CR1%d_CR2%d_CR3%d_fc%d" % (n_feature_map1, n_feature_map2, n_feature_map3, n_fully_connected_neuron) + "_test_acc_%.1f.ckpt"

                # Output of training measurements filename
                filename_measure     = "./" + start_date_time + "_measurements_U%d_" % U + NNetwork + "_CR1%d_CR2%d_CR3%d_fc%d" % (n_feature_map1, n_feature_map2, n_feature_map3, n_fully_connected_neuron) + "_test_acc_%.1f.dat"
        
                # Output of classification result with labels
                filename_result      = "./" + start_date_time + "_result_U%d_" % U + NNetwork + "_CR1%d_CR2%d_CR3%d_fc%d" % (n_feature_map1, n_feature_map2, n_feature_map3, n_fully_connected_neuron) + "_test_acc_%.1f.dat"

            else :
                # Output model filename
                filename_weight_bias = "./" + start_date_time + "_model_U%d+U%d_" % (U1,U2) + NNetwork + "_CR1%d_CR2%d_CR3%d_fc%d" % (n_feature_map1, n_feature_map2, n_feature_map3, n_fully_connected_neuron) + "_test_acc_%.1f.ckpt"

                # Output of training measurements filename
                filename_measure     = "./" + start_date_time + "_measurements_U%d+U%d_" % (U1,U2) + NNetwork + "_CR1%d_CR2%d_CR3%d_fc%d" % (n_feature_map1, n_feature_map2, n_feature_map3, n_fully_connected_neuron) + "_test_acc_%.1f.dat"

                # Output of classification result with labels
                filename_result      = "./" + start_date_time + "_result_U%d+U%d_" % (U1,U2) + NNetwork + "_CR1%d_CR2%d_CR3%d_fc%d" % (n_feature_map1, n_feature_map2, n_feature_map3, n_fully_connected_neuron) + "_test_acc_%.1f.dat"

        # Neural network architecture settings -----------------------------------------------
        n_output_neuron = 2

        # Spatial filter size: filter depth, height, and width
        filter_d = 2
        filter_h = filter_d
        filter_w = filter_d

        # Adaptive learning rate is used. As the training goes on, the learning rate is
        # lowered progressively using exponential decay function.
        #   Optimizer initial learning rate
        eta0 = 1e-3

        #   decay rate
        decay_rate = 0.925





        n_train_data = len(HSF.train.labels)

        while np.modf(float(n_train_data)/batch_size)[0] > 0.0 :
          print 'Warning! Number of data/ batch size must be an integer.'
          print 'Number of data: %d' % n_train_data
          print 'Batch size    : %d' % batch_size
          batch_size = int(input('Input new batch size: '))

        # Number of training cycle per training epoch
        iteration_per_epoch=n_train_data/batch_size

        print 'Number of training data: %d' % n_train_data

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
        # ReLU function. Zero padding is used in conv3d, i.e. padding = 'SAME',
        # the output size : n_feature_map1 x n_x x n_x x n_x
        h_conv1 = tf.nn.relu(conv3d(x_image, W_conv1, pad='SAME') + b_conv1)


        # Second Convolution Layer
        W_conv2 = weight_variable([filter_d,filter_h,filter_w,n_feature_map1,n_feature_map2])
        b_conv2 = bias_variable([n_feature_map2])

        h_conv2 = tf.nn.relu(conv3d(h_conv1, W_conv2, pad='SAME') + b_conv2)


        # Third Convolution Layer
        W_conv3 = weight_variable([filter_d,filter_h,filter_w,n_feature_map2,n_feature_map3])
        b_conv3 = bias_variable([n_feature_map3])

        h_conv3 = tf.nn.relu(conv3d(h_conv2, W_conv3, pad='SAME') + b_conv3)


        # Classification layer ---------------------------------------------------------------

        # Fully-connected Layer
        # Now add a fully-connected layer with n_fully_connected_neuron neurons to 
        # allow processing on the entire image. The tensor from the previous layer
        # is reshaped into a batch of vectors, multiply by a weight matrix, add a
        # bias, and apply a ReLU.
        W_fc1 = weight_variable([n_feature_map3*(n_x)**3, n_fully_connected_neuron])
        b_fc1 = bias_variable([n_fully_connected_neuron])

        h_conv1_flat = tf.reshape(h_conv3, [-1, n_feature_map3*(n_x)**3])
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
        # tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=False)
        # Use adaptive learning rate
        global_step = tf.Variable(0, trainable=False)
        eta = tf.train.exponential_decay(eta0, global_step*batch_size, n_train_data, decay_rate)
        train_step = tf.train.AdamOptimizer(eta).minimize(cross_entropy, global_step=global_step)
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # Before Variables can be used within a session, they must be initialized
        # using that session.
        sess.run(tf.initialize_all_variables())





        # Training ---------------------------------------------------------------------------

        # Training the model can be accomplished by repeatedly running train_step. Each
        # training iteration load n training examples. Then, the train_step operation can be
        # run using feed_dict to replace the placeholder tensors x and y_ with the training
        # examples. Note: any tensor in the computation graph can be replcaed using feed_dict.

        start_time = time.time()

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
            saver = tf.train.Saver([W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2])
            # Restore trained model.
            save_path = saver.restore(sess, filename_trained_model)

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
        slow_learning = False
        best_epoch   = 0
        file_save_counter = 0        

        for j in range(epochs):
          # Break out of the training epoch loop if overtraining is encountered.
          if Overtraining :
            break
          if best_test_accuracy <= 0.6 and j >= 4 :
            slow_learning = True
            print 'Slow learning. Exiting...'
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
                print 'Saving model %s and measurements %s.' % ((filename_weight_bias%(best_test_accuracy*100)).replace('./',''), (filename_measure%(best_test_accuracy*100)).replace('./',''))
                saver = tf.train.Saver([W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2])
                best_epoch = (n+1)*fractional_epoch
                if file_save_counter == 0 :
                  filename_weight_bias_tmp = (filename_weight_bias%(best_test_accuracy*100))
                  filename_measure_tmp = (filename_measure%(best_test_accuracy*100))
                  save_path = saver.save(sess, filename_weight_bias_tmp )
                  np.savetxt( filename_measure_tmp, Table_measure[:n+1,:])
                  check_model = tf.reduce_mean(W_conv1).eval()
                  file_save_counter += 1
                else :
                  os.remove( filename_weight_bias_tmp )
                  os.remove( filename_weight_bias_tmp + '.meta' )
                  os.remove( filename_measure_tmp )
                  filename_weight_bias_tmp = (filename_weight_bias%(best_test_accuracy*100))
                  filename_measure_tmp = (filename_measure%(best_test_accuracy*100))
                  save_path = saver.save(sess, filename_weight_bias_tmp )
                  np.savetxt( filename_measure_tmp, Table_measure[:n+1,:])
                  check_model = tf.reduce_mean(W_conv1).eval()
                  file_save_counter += 1 
              # Check for overtraining/ overfitting. If so, stop training and break out of the
              # training iteration per epoch loop.
              if train_accuracy > test_accuracy :
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
          if test_accuracy <= 0.55 and n*fractional_epoch >= 12 :
            slow_learning = True
            print 'Slow learning. Exiting...'
            break

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
              # Update the best test accuracy
              best_test_accuracy = test_accuracy
              print 'Saving model and measurements...' 
              saver = tf.train.Saver([W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2])
              check_model = tf.reduce_mean(W_conv1).eval()

              best_epoch = ndata_collect*fractional_epoch
              os.remove( filename_weight_bias_tmp )
              os.remove( filename_weight_bias_tmp + '.meta' )
              os.remove( filename_measure_tmp )
              filename_weight_bias_tmp = (filename_weight_bias%(best_test_accuracy*100))
              filename_measure_tmp = (filename_measure%(best_test_accuracy*100))
              save_path = saver.save(sess, filename_weight_bias_tmp )
              np.savetxt(filename_measure_tmp, Table_measure)

          print '%.2fs, epoch %.2f, training accuracy %g, test accuracy %g, cost %g' % (time.time()-start_time,(n+1)*fractional_epoch, train_accuracy, test_accuracy, Cost)
        else :
          os.remove( filename_measure_tmp )
          filename_measure_tmp = (filename_measure%(best_test_accuracy*100))
          print 'Saving measurements %s.' % (filename_measure%(best_test_accuracy*100)).replace('./','')
          # Save the measurements:
          # First column : Training epochs
          # Second column: Training accuracy
          # Third column : Testing accuracy
          # Fourth column: Cost
          np.savetxt(filename_measure_tmp, Table_measure[:n+1,:])

        if best_epoch == 0 :
          print 'Training model is not saved as saving criteria are not met. Classification will not be performed.'
          saver.restore(sess, filename_trained_model)
          model_saving_criteria_not_met = True 
        else :
          print 'Best training epoch: %g' % best_epoch
          print 'Model saved in file: ', save_path

          # To proceed, load the best (saved) model instead of the last training model.i
          filename_trained_model = (filename_weight_bias%(best_test_accuracy*100))
          saver.restore(sess, filename_trained_model)

          # Check if the saved model and the restored model are the same.
          if check_model != tf.reduce_mean(W_conv1).eval() :
            print 'Warning! Best training model and the restored model is incompatible. Exiting...'
            sys.exit()
          model_saving_criteria_not_met = False
        
        # Classification ---------------------------------------------------------------------

        if not(slow_learning) or not(model_saving_criteria_not_met):

          print 'Performing classification using %s.' % filename_trained_model.replace('./','')

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

          # Normalize the output of the second neuron
          Table[:,1] = Table[:,1]/Table[:,-1].astype('float')
          # Normalized output of the first neuron
          Table[:,2] = 1.0-Table[:,1]
          # Normalize the classification accuracy
          Table[:,3] = Table[:,3]/Table[:,-1].astype('float')

          np.savetxt((filename_result%(best_test_accuracy*100)), Table)
          print 'Result saved as %s.' % (filename_result%(best_test_accuracy*100))

