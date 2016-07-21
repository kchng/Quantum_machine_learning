import data_reader
import randomize_file_data
import numpy as np
import time
import sys

shuffle_data = True
Option1      = True

# First, shuffle the data :

# Provide the full file patch below and make sure to modify the number formatting accordingly. Here it is %.3f.

filename = '/home/kelvin/Desktop/Theano test/HSF_N4x4x4_L200_U9_Mu0_UniformTGrid/N4x4x4_L200_U9_Mu0_T%.3f.HSF.stream'

# Then provide the full list of file number (without including the one at Tc)

dtau = np.array([0.060, 0.075, 0.090, 0.105, 0.120, 0.135, 0.150, 0.165, \
                 0.180, 0.195, 0.210, 0.225, 0.240, 0.255, 0.270, 0.285, \
                 0.300, 0.315, 0.330, 0.345, 0.510, 0.660, 0.810, \
                 0.960, 1.110, 1.260, 1.410, 1.560, 1.710, 1.860, 2.010, \
                 2.160, 2.310, 2.460, 2.610, 2.760, 2.910, 3.060, 3.210, \
                 3.360])

if shuffle_data :

    # Initilize the python module by giving it the file information
    initialize = randomize_file_data.insert_file_info( filename, dtau, boundary = 0.36 )

    # Start randomizing data. Sit back and have a cup of coffee, it needs a bit of time.
    initialize.randomize_data()



# Second, run your neural network with the data_reader.py. Depending on the amount of memory you have, you can choose
# to read the data in one go (option 1) or read the data in small dosage (option 2).

# Again, give the full file path below and make sure to modify the number formatting to %.2d.
filename = '/home/kelvin/Desktop/Theano test/HSF_N4x4x4_L200_U9_Mu0_UniformTGrid/N4x4x4_L200_U9_Mu0_T_shuffled_%.2d.HSF.stream'

filenumber = np.arange(1,41,1)

HSF = data_reader.insert_file_info( filename, filenumber )

if Option1 :
    # Option 1

    HSF1 = HSF.categorize_data()

    for j in range(2) :
        for i in range(640) :
            batch = HSF1.train.next_batch()
            print i, (batch[1][:,0]).sum(), (batch[1][:,1]).sum()

else :
    # Option 2
    HSF2 = HSF.categorize_dose_of_data()

    HSF2.train.initialize_file_info_for_dose_of_data( filename, filenumber )

    for j in range(2) :
        for i in range(640) :
            batch = HSF2.train.next_dose()
            print i, (batch[1][:,0]).sum(), (batch[1][:,1]).sum()
