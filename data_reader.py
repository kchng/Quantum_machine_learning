# Author: Kelvin Chng
# (c) 2016
# San Jose State University

import numpy as np
import random
import time
import sys

class insert_file_info :
    
    def __init__(self, full_file_path, filenumber, batch_size = 50, 
        use_random_seed = False ) :
        """ full_file_path : full file path of the shuffled data
            filenumber     : An array of file number """
        self.filename         = full_file_path.rsplit('\\', 1)[-1]
        self.filename         = self.filename.rsplit('/', 1)[-1]
        self.full_file_path   = full_file_path
        self.nrows            = 0
        self.ncols            = 0
        self.nfile            = len(filenumber)
        self.batch_size       = batch_size
        self.current_index    = 0

    class DataSet(object) :
        file_info = None
    
        def __init__(self, images, labels, nrows, nfile_train, 
                     nfile_test, nfile_val, data_type = 'unknown') :
            #self.file_into = insert_file_info()
        
            #super(DataSet,self).__init__()
            #self.insert_file_info = insert_file_info

            self.data_type = data_type
            self.batch_size = 0
            self._ndata     = 0
            self._images = images
            self._labels = labels
            self._epochs_completed = 0
            self._index_in_epoch = 0
            self._index_in_datafile = 0
            self._file_index = 1
            self.nrows = nrows
        
            if self.data_type == 'train' :
                self.start_file_index   = 1
                self.end_file_index i   = nfile_train
                self._ndata             = nfile_train*self.nrows
                self.convert_to_one_hot = True
            elif self.data_type == 'test' :
                self.start_file_index   = nfile_train + 1
                self.end_file_index     = nfile_train + nfile_test
                self._ndata             = nfile_test*self.nrows
                self.convert_to_one_hot = True
            elif self.data_type == 'validation' :
                self.start_file_index   = nfile_train + nfile_test + 1
                self.end_file_index     = nfile_train + nfile_test + nfile_val
                self._ndata             = nfile_val*self.nrows
                self.convert_to_one_hot = False

        #@staticmethod
        #def feed_self(self, batch_size, nrows) :
        #    self.batch_size = batch_size
        #    self.nrows      = nrows
            #print self.batch_size, self.nrows

        @property
        def images(self):
            return self._images
        
        @property
        def labels(self):
            return self._labels
    
        @property
        def ndata(self):
            return self._ndata
    
        @property
        def epochs_completed(self):
            return self._epochs_completed
            
        def next_batch(self, batch_size = 50) :
            
            start = self._index_in_epoch
            if ( self._epochs_completed == 0 ) and ( start == 0 ) :
                self.batch_size = batch_size
                while np.modf(float(self.nrows)/self.batch_size)[0] > 0.0 :
                     print 'Warning! Number of data per file/ batch size must be an integer.'
                     self.batch_size = int(input('Input new batch size: ')
                print 'batch size : %d'    % self.batch_size
                print 'number of data: %d' % self._ndata

            self._index_in_epoch += self.batch_size
            if self._index_in_epoch > self._ndata :
                # Number of training epochs completed
                self._epochs_completed += 1
                # Shuffle data
                random.shuffle(self._images)
                random.shuffle(self._labels)
                # Reinitialize conunter
                start = 0
                self._index_in_epoch = self.batch_size
                assert self.batch_size <= self._ndata
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]

        def next_dose(self, batch_size = 50) :

            def convert_to_one_hot( label ) :
                label_one_hot = np.zeros((len(label),2))
                for i in range(len(label)) :
                    label_one_hot[i,label[i]] = 1
                return label_one_hot

            start = self._index_in_datafile 
            if ( self._file_index == self.start_file_index ) and ( start == 0 ) :
                self.batch_size = batch_size
                while np.modf(float(self.nrows)/self.batch_size)[0] > 0.0 :
                     print 'Warning! Number of data per file/ dose size must be an integer.'
                     self.batch_size = int(input('Input new dose size: ')
                print 'dose size : %d'    % self.batch_size
                print 'number of data: %d' % self._ndata

            self._index_in_datafile += self.batch_size
            if self._index_in_datafile > self.nrows :
                self._file_index += 1
                start = 0
                self._index_in_datafile = self.batch_size
                assert self.batch_size <= self.nrows

            if self._file_index > self.end_file_index :
                # Number of training epochs completed
                self._epochs_completed += 1
                self._file_index = self.start_file_index
                # Shufle data
                random.shuffle(self._images)
                random.shuffle(self._labels)
                # Reinitialize conunter
                start = 0
                self._index_in_datafile = self.batch_size

            end = self._index_in_datafile

            # Read in small dosage of data
            data = np.genfromtxt(self.full_file_path%(self._file_index) ,dtype=int,
                   skip_header=start, skip_footer=self.nrows-end)
            self._images = data[:,:-1].astype('int')
            labels = data[:,-1:].astype('int')
            if self.convert_to_one_hot :
                self._labels = convert_to_one_hot(labels)

            return self._images, self._labels

    def categorize_data(self, convert_test_labels_to_one_hot = True) :
        class DataSets(object):
            pass
        data_sets = DataSets()
        
        def convert_to_one_hot( label ) :
            label_one_hot = np.zeros((len(label),2))
            for i in range(len(label)) :
                label_one_hot[i,label[i]] = 1
            return label_one_hot
        
        data = np.loadtxt(self.full_file_path%1)
        self.nrows, self.ncols = np.shape(data)
        self.nrows, self.ncols = int(self.nrows), int(self.ncols)
        
        if np.modf(float(self.nrows)/self.batch_size)[0] > 0.0 :
            self.batch_size = int(float(self.nrows)/20)    
            
        # Use 10% of the data each for testing and validating, the remaining for
        # training    
        nfile_train = int(self.nfile*.8)
        nfile_test  = int(self.nfile*.1)
        nfile_val   = nfile_test
    
        n_data_check = self.nfile - ( nfile_train + nfile_test + nfile_val )
        if n_data_check > 0 :
            nfile_train += n_data_check
        elif n_data_check < 0 :
            nfile_train -= n_data_check
   
        #self.ndata = nfile_train*self.nrows     
        start_time = time.time()
    
        TRAIN_DATA = np.zeros((nfile_train*self.nrows,self.ncols))
        train_images = np.zeros((nfile_train*self.nrows,self.ncols-1))
        train_labels = np.zeros((nfile_train*self.nrows,1))
        print 'Loading %d/%d files for training data...' % (nfile_train,self.nfile)
        for i in range(nfile_train) :
            print '%.1fs. Loading file %d.' % (time.time()-start_time, i+1)
            TRAIN_DATA[i*self.nrows:(i+1)*self.nrows,:] = np.loadtxt(self.full_file_path%(i+1))
        train_images = TRAIN_DATA[:,:-1].astype('int')
        train_labels = TRAIN_DATA[:,-1].astype('int')
        train_labels = convert_to_one_hot(train_labels)
        
        print 'Loading %d/%d files for test data...' % (nfile_test,self.nfile)
        TEST_DATA = np.zeros((nfile_test*self.nrows,self.ncols))
        test_images = np.zeros((nfile_test*self.nrows,self.ncols-1))
        test_labels = np.zeros((nfile_test*self.nrows,1))
        for i in range(nfile_test) :
            print '%.1fs. Loading file %d.' % (time.time()-start_time, i+1)
            TEST_DATA[i*self.nrows:(i+1)*self.nrows,:] = np.loadtxt(self.full_file_path%(i+1+nfile_train))
        test_images = TEST_DATA[:,:-1].astype('int')
        test_labels = TEST_DATA[:,-1].astype('int')
        if convert_test_labels_to_one_hot :
            test_labels = convert_to_one_hot(test_labels)
    
        print 'Loading %d/%d files for validation data...' % (nfile_val,self.nfile)
        VALIDATION_DATA = np.zeros((nfile_val*self.nrows,self.ncols))
        validation_images = np.zeros((nfile_val*self.nrows,self.ncols-1))
        validation_labels = np.zeros((nfile_val*self.nrows,1))
        for i in range(nfile_test) :
            print '%.1fs. Loading file %d.' % (time.time()-start_time, i+1)
            VALIDATION_DATA[i*self.nrows:(i+1)*self.nrows,:] = np.loadtxt(self.full_file_path%(i+1+nfile_train+nfile_test))
        validation_images = VALIDATION_DATA[:,:-1].astype('int')
        validation_labels = VALIDATION_DATA[:,-1].astype('int')

        data_sets.train      = insert_file_info.DataSet(train_images, train_labels,
                               self.nrows, nfile_train, nfile_test, nfile_val,
                               data_type = 'train')
        data_sets.test       = insert_file_info.DataSet(test_images, test_labels,
                               self.nrows, nfile_train, nfile_test, nfile_val,
                               data_type = 'test')
        data_sets.validation = insert_file_info.DataSet(validation_images,
                               validation_labels, self.nrows, nfile_train,
                               nfile_test, nfile_val, data_type = 'validation')

        return data_sets
 
    def categorize_dose_of_data(self) :
        class DataSets(object):
            pass
        data_sets = DataSets()

        data = np.loadtxt(self.full_file_path%1)
        self.nrows, self.ncols = np.shape(data)
        self.nrows, self.ncols = int(self.nrows), int(self.ncols)

        if np.modf(float(self.nrows)/self.batch_size)[0] > 0.0 :
            self.batch_size = int(float(self.nrows)/20)

        # Use 10% of the data each for testing and validating, the remaining for
        # training    
        nfile_train = int(self.nfile*.8)
        nfile_test  = int(self.nfile*.1)
        nfile_val   = nfile_test

        n_data_check = self.nfile - ( nfile_train + nfile_test + nfile_val )
        if n_data_check > 0 :
            nfile_train += n_data_check
        elif n_data_check < 0 :
            nfile_train -= n_data_check
        
        train_images = np.array([]).astype('int')
        train_labels = np.array([]).astype('int')

        test_images = np.array([]).astype('int')
        test_labels = np.array([]).astype('int') 

        validation_images = np.array([]).astype('int')
        validation_labels = np.array([]).astype('int')
         
        data_sets.train      = insert_file_info.DataSet(train_images, train_labels, 
                               self.nrows, nfile_train, nfile_test, nfile_val, 
                               data_type = 'train')
        data_sets.test       = insert_file_info.DataSet(test_images, test_labels, 
                               self.nrows, nfile_train, nfile_test, nfile_val, 
                               data_type = 'test')
        data_sets.validation = insert_file_info.DataSet(validation_images, 
                               validation_labels, self.nrows, nfile_train, 
                               nfile_test, nfile_val, data_type = 'validation')
        
        return data_sets

