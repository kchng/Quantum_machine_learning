import numpy as np
import os.path
import random
import string
import sys
import time

# print os.path.isfile('Tempfile0.txt')

class insert_file_info :
    
    def __init__(self, full_file_path, filenumber, boundary, nrows = 1000,
        ncols = 12800, use_random_seed = True ) :
        """ filenumber : An array of file number """
        self.filename         = full_file_path.rsplit('\\', 1)[-1]
        self.filename         = self.filename.rsplit('/', 1)[-1]
        self.newfilename      = string.replace(self.filename, '%.3f', '_shuffled_%.2d')
        self.full_file_path   = full_file_path
        self.filenumber       = filenumber
        self.filenumber_below = filenumber[filenumber<boundary]
        self.filenumber_above = filenumber[filenumber>boundary]
        self.nfile            = len(filenumber)
        self.nrows            = nrows
        self.ncols            = ncols
        self.delimiter        = [1 for i in xrange(self.ncols)]
        
        if not(use_random_seed) :
            print 'Using fixed seed.'
            random.seed(500)
        else :
            print 'Using random seed.'      
        
        print '%d files below and %d files above the boundary.' % (len(self.filenumber_below), len(self.filenumber_above))

        #if np.mod(self.nfile,2) > 0 :
        #    print ('Warning! Make sure there are same number of files above and below the boundary. Exiting...')
        #    sys.exit()

        print 'File shape: %d x %d' % (self.nrows, self.ncols)
 
    def randomize_data(self, memory_size = 'medium', shuffle_data = True) :
        print 'Memory setting: %s'%memory_size
        print 'Use ''medium'' setting if RAM is < 8 GB; otherwise, set it to high for speed.'
        """ Import and randomize data, then export to new files """
        nfile_below = len(self.filenumber_below)
        ndata_below = int(float(nfile_below)/self.nfile*float(self.nrows))
        nfile_above = len(self.filenumber_above)
        ndata_above = int(float(nfile_above)/self.nfile*float(self.nrows))

        if memory_size == 'medium' :
       
            # The second to last column is the label and the last column holds the information for temperature.
            shuffled_data = np.zeros((nfile_below*self.nrows, self.ncols+2))
            shuffled_indices = np.arange(0,nfile_below*self.nrows,1)
            if shuffle_data :
                random.shuffle(shuffled_indices)
                random.shuffle(shuffled_indices)
 
            start_time = time.time()
            for i in range(nfile_below) :
                print '%.1f' % (time.time()-start_time),'s. Opening', self.filename % self.filenumber_below[i], '...'
                shuffled_data[i*self.nrows:(i+1)*self.nrows,:self.ncols] = np.genfromtxt( 
                        self.full_file_path % self.filenumber_below[i], dtype = int, 
                        delimiter=self.delimiter, skip_header=0, skip_footer=0)
                shuffled_data[i*self.nrows:(i+1)*self.nrows,-1:] = i
            print '\nShuffling data...\n'
            shuffled_data = shuffled_data[shuffled_indices]
            
            for i in range(self.nfile) :
                print '%.1f' % (time.time()-start_time),'s. Saving shuffled data %d.'%(i+1)
                with open(self.newfilename%(i+1),'w') as f:
                    #f.write('\n\n')
                    np.savetxt(f, shuffled_data[i*ndata_below:(i+1)*ndata_below,:], fmt='%d')

            # The second to last column is the label and the last column holds the information for temperature.
            shuffled_data = np.zeros((nfile_above*self.nrows, self.ncols+2))
            shuffled_indices = np.arange(0,nfile_above*self.nrows,1)
            if shuffle_data :
                random.shuffle(shuffled_indices)
                random.shuffle(shuffled_indices)

            for i in range(nfile_above) :
                print '%.1f' % (time.time()-start_time),'s. Opening', self.filename % self.filenumber_above[i], '...'
                shuffled_data[i*self.nrows:(i+1)*self.nrows,:self.ncols] = np.genfromtxt( 
                        self.full_file_path % self.filenumber_above[i], dtype = int, 
                        delimiter=self.delimiter, skip_header=0, skip_footer=0)
                shuffled_data[i*self.nrows:(i+1)*self.nrows,-1:] = i+nfile_below
            print '\nShuffling data...\n'
            shuffled_data = shuffled_data[shuffled_indices]
            shuffled_data[:,-2:-1] = 1
                    
            for i in range(self.nfile) :
                print '%.1f' % (time.time()-start_time),'s. Saving shuffled data %d.'%(i+1)
                with open(self.newfilename%(i+1),'a') as f:
                    #f.write('\n\n')
                    np.savetxt(f, shuffled_data[i*ndata_above:(i+1)*ndata_above,:], fmt='%d')

            shuffled_indices = np.arange(0,self.nrows,1)
            for i in range(self.nfile) :
                print '%.1f' % (time.time()-start_time),'s. Reshuffling', self.newfilename%(i+1),'.'
                if shuffle_data :
                    random.shuffle(shuffled_indices)
                    random.shuffle(shuffled_indices)
                data = np.loadtxt(self.newfilename%(i+1))
                data = data[shuffled_indices]
                with open(self.newfilename%(i+1),'w') as f:
                    np.savetxt(f, data, fmt='%d')        
                        
        if memory_size == 'high' :

            # The second to last column is the label and the last column holds the information for temperature.
            shuffled_data = np.zeros((self.nfile*self.nrows, self.ncols+2))
            shuffled_indices = np.arange(0,self.nfile*self.nrows,1)
            if shuffle_data :
                random.shuffle(shuffled_indices)
                random.shuffle(shuffled_indices)

            start_time = time.time() 
            for i in range(self.nfile) :
                print '%.1f' % (time.time()-start_time),'s. Opening', self.filename % self.filenumber[i], '...'
                shuffled_data[i*self.nrows:(i+1)*self.nrows,:self.ncols] = np.genfromtxt( 
                        self.full_file_path % self.filenumber[i], dtype = int, 
                        delimiter=self.delimiter, skip_header=0, skip_footer=0)
                shuffled_data[i*self.nrows:(i+1)*self.nrows,-1:] = i
    
            print '\nShuffling data...\n'
            shuffled_data[nfile_below*self.nrows:,-2:-1] = 1
            shuffled_data = shuffled_data[shuffled_indices]

            for i in range(self.nfile) :
                print '%.1f' % (time.time()-start_time),'s. Saving shuffled data %d.'%(i+1)
                with open(self.newfilename%(i+1),'w') as f:
                    #f.write('\n\n')
                    np.savetxt(f, shuffled_data[i*self.nrows:(i+1)*self.nrows,:], fmt='%d')
            
        print 'Done.'
