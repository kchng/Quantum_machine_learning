import randomize_file_data_tmp
import numpy as np
import os
import time
import sys

# If you are in the directory where the data file are located, make sure to modify the number formatting accordingly. Here it is %.3f. If not, give the full file path and modify the number formatting accordingly. Make sure the data file have consisten number formatting, i.e. 3 decimal places.

# Potential energy
U = 9
# System size
nspin = 4
# Critical temperature
Tc = 0.36
# Memory setting
memory_setting = 'medium'

filename = './N%dx%dx%d_L200_U%d_Mu0_T' % (nspin,nspin,nspin,U) + '%.3f.HSF.stream'

# Get temperature
os.system("ls -l N%dx%dx%d_L200_U%d_Mu0_T*.HSF.stream | awk '{print $9}' | sed -e s/N%dx%dx%d_L200_U%d_Mu0_T//g -e s/.HSF.stream//g > dtau.dat" %(nspin,nspin,nspin,U,nspin,nspin,nspin,U))

# Load temperature into a list of string
dtau = np.genfromtxt("dtau.dat",dtype='str')

# Rename files for consistent formating
for i in range(len(dtau)):
   os.system("mv N%dx%dx%d_L200_U%d_Mu0_T%s.HSF.stream N%dx%dx%d_L200_U%d_Mu0_T%.3f.HSF.stream" % (nspin,nspin,nspin,U,dtau[i],nspin,nspin,nspin,U,float(dtau[i])) )

# Get temperature (with consistent formating) 
os.system("ls -l N%dx%dx%d_L200_U%d_Mu0_T*.HSF.stream | awk '{print $9}' | sed -e s/N%dx%dx%d_L200_U%d_Mu0_T//g -e s/.HSF.stream//g > dtau.dat" % (nspin,nspin,nspin,U,nspin,nspin,nspin,U) )

dtau = np.genfromtxt("dtau.dat")

# Initilize the python module by giving it the file information
initialize = randomize_file_data_tmp.insert_file_info( filename, dtau, boundary = Tc )

# Start randomizing data. Sit back and have a cup of coffee, it needs a bit of time. On an i5 3.1 GHz, it takes about 24 minutes on the medium setting. If your computer have more than 8 GB of memory, set it to high and you should expect a 30% reduction in time on a comparable processor.
initialize.randomize_data(memory_size = memory_setting)
