import randomize_file_data
import numpy as np
import os
import time
import sys

# If you are in the directory where the data file are located, make sure to modify the number formatting accordingly. Here it is %.3f. If not, give the full file path and modify the number formatting accordingly. Make sure the data file have consisten number formatting, i.e. 3 decimal places.

# Potential energy
U = 9
# System size
n_x = 4
# Imaginary time
L = 200
# Critical temperature
Tc = 0.36
# Memory setting
memory_setting = 'medium'

filename = './N%dx%dx%d_L200_U%d_Mu0_T' % (n_x,n_x,n_x,U) + '%.3f.HSF.stream'

# Get temperature and save them to a file.
os.system("ls -l N%dx%dx%d_L%d_U%d_Mu0_T*.HSF.stream | awk '{print $9}' | sed -e s/N%dx%dx%d_L%d_U%d_Mu0_T//g -e s/.HSF.stream//g > dtau.dat" %(n_x,n_x,n_x,L,U,n_x,n_x,n_x,L,U))

# Load temperature into a list of string
dtau = np.genfromtxt("dtau.dat",dtype='str')

# Rename files for consistent formating
for i in range(len(dtau)):
   os.system("mv N%dx%dx%d_L%d_U%d_Mu0_T%s.HSF.stream N%dx%dx%d_L%d_U%d_Mu0_T%.3f.HSF.stream" % (n_x,n_x,n_x,L,U,dtau[i],n_x,n_x,n_x,L,U,float(dtau[i])) )

# Get reformatted temperature and save them to a file.
os.system("ls -l N%dx%dx%d_L%d_U%d_Mu0_T*.HSF.stream | awk '{print $9}' | sed -e s/N%dx%dx%d_L%d_U%d_Mu0_T//g -e s/.HSF.stream//g > dtau.dat" % (n_x,n_x,n_x,L,U,n_x,n_x,n_x,L,U) )

dtau = np.genfromtxt("dtau.dat")

# Initilize the python module by giving it the file information
initialize = randomize_file_data.insert_file_info( filename, dtau, boundary = Tc )

# Start randomizing data. Sit back and have a cup of coffee, it needs a bit of time. On an i5 3.1 GHz, it takes about 24 minutes on the medium setting. If your computer have more than 8 GB of memory, set it to high and you should expect a 30% reduction in time on a comparable processor.
initialize.randomize_data(memory_size = memory_setting)
