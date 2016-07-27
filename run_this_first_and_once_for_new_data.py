import randomize_file_data
import numpy as np
import time
import sys

# If you are in the directory where the data file are located, make sure to modify the number formatting accordingly. Here it is %.3f. If not, give the full file path and modify the number formatting accordingly. Make sure the data file have consisten number formatting, i.e. 3 decimal places.

filename = './N4x4x4_L200_U9_Mu0_T%.3f.HSF.stream'

# Then provide the full list of file number (WITHOUT including the one at T_c)

dtau = np.array([0.060, 0.075, 0.090, 0.105, 0.120, 0.135, 0.150, 0.165, \
                 0.180, 0.195, 0.210, 0.225, 0.240, 0.255, 0.270, 0.285, \
                 0.300, 0.315, 0.330, 0.345, 0.510, 0.660, 0.810, \
                 0.960, 1.110, 1.260, 1.410, 1.560, 1.710, 1.860, 2.010, \
                 2.160, 2.310, 2.460, 2.610, 2.760, 2.910, 3.060, 3.210, \
                 3.360])

# Initilize the python module by giving it the file information
initialize = randomize_file_data.insert_file_info( filename, dtau, boundary = 0.36 )

# Start randomizing data. Sit back and have a cup of coffee, it needs a bit of time. On an i5 3.1 GHz, it takes about 24 minutes on the medium setting. If your computer have more than 8 GB of memory, set it to high and you should expect a 30% reduction in time on a comparable processor.
initialize.randomize_data(memory_size = 'medium',shuffle_data=True)

