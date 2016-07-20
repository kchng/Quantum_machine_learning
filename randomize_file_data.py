import numpy as np
import os.path
import random
import string
import sys
import time

# print os.path.isfile('Tempfile0.txt')

class insert_file_info :
    
    def __init__(self, full_file_path, filenumber, boundary, nrows = 1000, 
        ncols = 12800, use_random_seed = False ) :
        """ filenumber : An array of file number """
        self.filename         = full_file_path.rsplit('\\', 1)[-1]
        self.filename         = self.filename.rsplit('/', 1)[-1]
        self.newfilename      = string.replace(self.filename, '%.3f', '_shuffled_%.2d')
        self.full_file_path   = full_file_path
        self.filenumber_below = filenumber[filenumber<boundary]
        self.filenumber_above = filenumber[filenumber>boundary]
        self.nrows            = nrows
        self.ncols            = ncols
        self.nfile            = len(filenumber)
        self.Delimiter = [1 for i in xrange(self.ncols)]
        
        if not(use_random_seed) :
            print 'Using fixed seed.'
            random.seed(500)
        else :
            print 'Using random seed.'      
        
        if np.mod(self.nfile,2) > 0 :
            print ('Warning! Make sure there are same number of files above and below the boundary. Exiting...')
            sys.exit()
        
    def randomize_data(self, memory_size = 'medium') :
        """ Import and randomize data, then export to new files """
        half_nfile = int(self.nfile/2.)
        half_nrows = int(self.nrows/2.)
        shuffled_data = np.zeros((half_nfile*self.nrows, self.ncols+1))
        shuffled_indices = np.arange(0,half_nfile*self.nrows,1)
        random.shuffle(shuffled_indices)
        random.shuffle(shuffled_indices)

        start_time = time.time()
        if memory_size == 'medium' :
        
            for i in range(half_nfile) :
                print '%.1f' % (time.time()-start_time),'s. Opening', self.filename % self.filenumber_below[i], '...'
                shuffled_data[i*self.nrows:(i+1)*self.nrows,:self.ncols] = np.genfromtxt( 
                        self.full_file_path % self.filenumber_below[i], dtype = int, 
                        delimiter=self.Delimiter, skip_header=0, skip_footer=0)
            print '\nShuffling data...\n'
            shuffled_data = shuffled_data[shuffled_indices]
            
            for i in range(self.nfile) :
                print 'Saving shuffled data %d.'%(i+1)
                with open(self.newfilename%(i+1),'w') as f:
                    #f.write('\n\n')
                    np.savetxt(f, shuffled_data[i*half_nrows:(i+1)*half_nrows,:], fmt='%d')
                                
            for i in range(half_nfile) :
                print '%.1f' % (time.time()-start_time),'s. Opening', self.filename % self.filenumber_above[i], '...'
                shuffled_data[i*self.nrows:(i+1)*self.nrows,:self.ncols] = np.genfromtxt( 
                        self.full_file_path % self.filenumber_above[i], dtype = int, 
                        delimiter=self.Delimiter, skip_header=0, skip_footer=0)
            print '\nShuffling data...\n'
            shuffled_data = shuffled_data[shuffled_indices]
            shuffled_data[:,-1:] = 1
                    
            for i in range(self.nfile) :
                print 'Saving shuffled data %d.'%(i+1)
                with open(self.newfilename%(i+1),'a') as f:
                    #f.write('\n\n')
                    np.savetxt(f, shuffled_data[i*half_nrows:(i+1)*half_nrows,:], fmt='%d')

            shuffled_indices = np.arange(0,self.nrows,1)
            for i in range(self.nfile) :
                print '%.1f' % (time.time()-start_time),'s. Reshuffling', self.newfilename%(i+1),'.'
                random.shuffle(shuffled_indices)
                random.shuffle(shuffled_indices)
                data = np.loadtxt(self.newfilename%(i+1))
                data = data[shuffled_indices]
                with open(self.newfilename%(i+1),'w') as f:
                    np.savetxt(f, data, fmt='%d')        
                        
        if memory_size == 'high' :
        
            for i in range(half_nfile) :
                print '%.1f' % (time.time()-start_time),'s. Opening', self.filename % self.filenumber_below[i], '...'
                shuffled_data[i*self.nrows:(i+1)*self.nrows,:self.ncols] = np.genfromtxt( 
                        self.full_file_path % self.filenumber_below[i], dtype = int, 
                        delimiter=self.Delimiter, skip_header=0, skip_footer=0)
            print '\nShuffling data...\n'
    
            for i in range(half_nfile) :
                print '%.1f' % (time.time()-start_time),'s. Opening', self.filename % self.filenumber_above[i], '...'
                shuffled_data[(i+half_nfile)*self.nrows:(i+half_nfile+1)*self.nrows,:self.ncols] = np.genfromtxt(
                        self.full_file_path % self.filenumber_above[i], dtype = int,
                        delimiter=self.Delimiter, skip_header=0, skip_footer=0)
            print '\nShuffling data...\n'
            shuffled_data[half_nfile*self.nrows:,-1:] = 1
            shuffled_data = shuffled_data[shuffled_indices]
            
            for i in range(self.nfile) :
                print 'Saving shuffled data %d.'%(i+1)
                with open(self.newfilename%(i+1),'w') as f:
                    #f.write('\n\n')
                    np.savetxt(f, shuffled_data[i*self.nrows:(i+1)*self.nrows,:], fmt='%d')
            
        print 'Done.'