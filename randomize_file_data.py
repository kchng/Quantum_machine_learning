import numpy as np
import os.path
import random
import string
import sys
import time
import scipy.stats

class insert_file_info :
    
    def __init__(self, full_file_path, filenumber, boundary, nrows = 1000,
        ncols = 12800, use_random_seed = True ) :
        """ filenumber : An array of file number """
        self.filename         = full_file_path.rsplit('\\', 1)[-1]
        self.filename         = self.filename.rsplit('/', 1)[-1]
        self.newfilename      = string.replace(self.filename, '%.3f.HSF.stream', '_shuffled_%.2d.dat')
        self.boundary         = boundary
        self.full_file_path   = full_file_path
        self.filenumber       = filenumber
        self.filenumber_below = filenumber[filenumber<self.boundary]
        self.filenumber_above = filenumber[filenumber>self.boundary]
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

        if abs( len(self.filenumber_below) - len(self.filenumber_above) ) > 0 :
            print ('Warning! Make sure there are same number of files above and below the boundary. Exiting...')
            sys.exit()
        
        if boundary == filenumber[filenumber==boundary] :
            self.boundary_file_exist = True
            print '\nBoundary file data exists.\n'
        else :
            self.boundary_file_exist = False
            print '\nBoundary file data not found.\n'

        print 'File shape: %d x %d' % (self.nrows, self.ncols)
 
    def randomize_data(self, memory_size = 'medium') :
        """ Import and randomize data, then export to new files """

        def shuffle_( indices ) :
            random.shuffle(indices)
            random.shuffle(indices)
            random.shuffle(indices)
            random.shuffle(indices)
            random.shuffle(indices)
            random.shuffle(indices)
            random.shuffle(indices)
            return indices

        def shuffle_normal( indices, nfile, nrows ) :
            Reshaped_indices = np.reshape(indices,(nrows,nfile)).T
            for i in range(nfile) :
                Reshaped_indices[i,:] = shuffle_( Reshaped_indices[i,:] )
            indices = np.reshape(Reshaped_indices,nfile*nrows)
            return indices

        print 'Memory setting: %s'%memory_size
        print 'Use ''medium'' setting if RAM is < 8 GB; otherwise, set it to high for speed.'

        nfile_below = len(self.filenumber_below)
        nfile_above = len(self.filenumber_above)

        if self.boundary_file_exist == True :
            n = 0.5
        else :
            n = 0.0
        ndata_below = int((nfile_below+n)/self.nfile*float(self.nrows))
        ndata_above = int((nfile_above+n)/self.nfile*float(self.nrows))
        nrows_half  = int(self.nrows/2.)

        if memory_size == 'medium' :
 
            if self.boundary_file_exist == True :
            #    # Load and hold data file on the boundary
                print 'Preloading boundary file...'
                boundary_data =  np.genfromtxt(
                    self.full_file_path % self.filenumber[self.filenumber==self.boundary], dtype = int,
                    delimiter=self.delimiter, skip_header=0, skip_footer=0)
                print 'Shuffling boundary file...\n'
                shuffled_indices = shuffle_normal( np.arange(0,self.nrows,1), 2, nrows_half )
                boundary_data = boundary_data[shuffled_indices]
                Labelling_cutoff = nfile_below*self.nrows + nrows_half
                nlabelling_cutoff = nfile_below + 1
            else :
                # If the data for which it is at the boundary exist, label it equally as being 0 (unordered) or 1 (ordered).
                Labelling_cutoff = nfile_below*self.nrows
                nlabelling_cutoff = nfile_below 
            # The second to last column is the label and the last column holds the information for temperature.
            shuffled_data = np.zeros((Labelling_cutoff, self.ncols+2))

            start_time = time.time()
            for i in range(nfile_below) :
                print '%.1f' % (time.time()-start_time),'s. Opening', self.filename % self.filenumber_below[i], '...'
                shuffled_data[i*self.nrows:(i+1)*self.nrows,:self.ncols] = np.genfromtxt( 
                        self.full_file_path % self.filenumber_below[i], dtype = int, 
                        delimiter=self.delimiter, skip_header=0, skip_footer=0)
                shuffled_data[i*self.nrows:(i+1)*self.nrows,-1:] = i
            if self.boundary_file_exist == True :
                # Add half of the data from the boundary data file.
                shuffled_data[(i+1)*self.nrows:,:self.ncols] = boundary_data[:nrows_half,:]
                shuffled_data[(i+1)*self.nrows:,-1:] = i+1
            print '\nShuffling data...\n'
            shuffled_indices = shuffle_normal( np.arange(0,Labelling_cutoff,1), self.nfile, ndata_below )
            shuffled_data = shuffled_data[shuffled_indices]
           
            print 'Checking data...\n'
            frequency_tmp = np.zeros(self.nfile)
            frequency_tmp[:nlabelling_cutoff] = scipy.stats.itemfreq(shuffled_data[:,-1:])[:,1]
            frequency_sum = sum(frequency_tmp)
            if abs(frequency_sum - Labelling_cutoff) > 0 :
                print 'Error in shuffled data. Exiting...'
                sys.exit()

            for i in range(self.nfile) :
                print '%.1f' % (time.time()-start_time),'s. Saving shuffled data %d.'%(i+1)
                with open(self.newfilename%(i+1),'w') as f:
                   #f.write('\n\n')
                    np.savetxt(f, shuffled_data[i*ndata_below:(i+1)*ndata_below,:], fmt='%d')

            if self.boundary_file_exist == True :
                Labelling_cutoff = self.nfile*self.nrows - Labelling_cutoff
                n, m = 1, nrows_half
            else :
                # If the data for which it is at the boundary exist, label it equally as being 1 or 0.
                Labelling_cutoff = nfile_above*self.nrows
                n, m = 0, 0
            # The second to last column is the label and the last column holds the temperature indices.
            shuffled_data = np.zeros((Labelling_cutoff, self.ncols+2))

            for i in range(nfile_above) :
                print '%.1f' % (time.time()-start_time),'s. Opening', self.filename % self.filenumber_above[i], '...'
                shuffled_data[i*self.nrows+m:(i+1)*self.nrows+m,:self.ncols] = np.genfromtxt( 
                        self.full_file_path % self.filenumber_above[i], dtype = int, 
                        delimiter=self.delimiter, skip_header=0, skip_footer=0)
                shuffled_data[i*self.nrows+m:(i+1)*self.nrows+m,-1:] = i+n+nfile_below
            print '\nShuffling data...\n'
            shuffled_indices = shuffle_normal( np.arange(0,Labelling_cutoff,1), self.nfile, ndata_below )
            if self.boundary_file_exist == True :
                # Add half of the data from the boundary data file
                shuffled_data[:nrows_half,:self.ncols] = boundary_data[nrows_half:,:]
                shuffled_data[:nrows_half,-1:] = nfile_below
            shuffled_data = shuffled_data[shuffled_indices]
            shuffled_data[:,-2:-1] = 1

            print 'Checking data...\n'
            frequency = np.zeros(self.nfile)
            if self.boundary_file_exist == True : 
                frequency[nlabelling_cutoff-1:] = scipy.stats.itemfreq(shuffled_data[:,-1:])[:,1]
            else :
                frequency[nlabelling_cutoff:] = scipy.stats.itemfreq(shuffled_data[:,-1:])[:,1]
            frequency += frequency_tmp
            frequency_checker = np.zeros(self.nfile)
            frequency_checker += self.nrows
            if not(np.array_equal(frequency,frequency_checker)) or not( sum(shuffled_data[:,-2:-1]) == Labelling_cutoff ) :
                print 'Error in shuffled data. Exiting...'
                sys.exit()
        
            for i in range(self.nfile) :
                print '%.1f' % (time.time()-start_time),'s. Saving shuffled data %d.'%(i+1)
                with open(self.newfilename%(i+1),'a') as f:
                    #f.write('\n\n')
                    np.savetxt(f, shuffled_data[i*ndata_above:(i+1)*ndata_above,:], fmt='%d')

            for i in range(self.nfile) :
                print '%.1f' % (time.time()-start_time),'s. Reshuffling', self.newfilename%(i+1),'.'
                shuffled_indices = shuffle_( np.arange(0,self.nrows,1) )
                data = np.loadtxt(self.newfilename%(i+1))
                data = data[shuffled_indices]
                with open(self.newfilename%(i+1),'w') as f:
                    np.savetxt(f, data, fmt='%d')        
                        
        if memory_size == 'high' :

            # The second to last column is the label and the last column holds the temperature indices.
            shuffled_data = np.zeros((self.nfile*self.nrows, self.ncols+2))

            start_time = time.time()
            if self.boundary_file_exist == True :

                for i in range(self.nfile) :
                    print '%.1f' % (time.time()-start_time),'s. Opening', self.filename % self.filenumber[i], '...'
                    shuffled_data[i*self.nrows:(i+1)*self.nrows,:self.ncols] = np.genfromtxt(
                        self.full_file_path % self.filenumber[i], dtype = int,
                        delimiter=self.delimiter, skip_header=0, skip_footer=0)
                    # If the boundary data file exist, shuffle the data.
                    if i == nfile_below :
                        shuffled_indices = shuffle_normal( np.arange(0,self.nrows,1), 2, self.nrows/2 )
                        boundary_data = shuffled_data[i*self.nrows:(i+1)*self.nrows,:self.ncols]
                        shuffled_data[i*self.nrows:(i+1)*self.nrows,:self.ncols] = boundary_data[shuffled_indices]
                    shuffled_data[i*self.nrows:(i+1)*self.nrows,-1:] = i
                n, m = 1, nrows_half

            else :

                for i in range(self.nfile) :
                    print '%.1f' % (time.time()-start_time),'s. Opening', self.filename % self.filenumber[i], '...'
                    shuffled_data[i*self.nrows:(i+1)*self.nrows,:self.ncols] = np.genfromtxt( 
                        self.full_file_path % self.filenumber[i], dtype = int, 
                        delimiter=self.delimiter, skip_header=0, skip_footer=0)
                    shuffled_data[i*self.nrows:(i+1)*self.nrows,-1:] = i
                n, m = 0, 0
 
            print '\nLabelling data...\n'
            shuffled_data[nfile_below*self.nrows+m:,-2:-1] = 1
  
            print 'Shuffling data...\n'
            shuffled_indices = shuffle_normal( np.arange(0,self.nfile*self.nrows,1), self.nfile, self.nrows )
            shuffled_data = shuffled_data[shuffled_indices]           

            print 'Checking data...\n'
            frequency = scipy.stats.itemfreq(shuffled_data[:,-1:])[:,1]
            frequency_checker = np.zeros(self.nfile)
            frequency_checker += self.nrows
            if not(np.array_equal(frequency,frequency_checker)) :
                print 'Error in shuffled data. Exiting...'
                sys.exit()

            for i in range(self.nfile) :
            	print '%.1f' % (time.time()-start_time),'s. Saving shuffled data %d.'%(i+1)
                with open(self.newfilename%(i+1),'w') as f:
                    #f.write('\n\n')
                    np.savetxt(f, shuffled_data[i*self.nrows:(i+1)*self.nrows,:], fmt='%d')
            
        print 'Done.'
