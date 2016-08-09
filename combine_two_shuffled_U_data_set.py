import numpy as np
import os
import time
import sys

# This algorithm rename shuffled data files from two separate U.

# Potential energy of first collection of file
U1 = 4
# Potential energy of second collection of file
U2 = 20
# System size
n_x = 4
# Imaginary time
L = 200
# Memory setting
memory_setting = 'medium'

# Get filenumbers and save it to file.
os.system("ls -l N%dx%dx%d_L%d_U%d_Mu0_T_shuffled_*.dat | awk '{print $9}' | sed -e s/N%dx%dx%d_L%d_U%d_Mu0_T_shuffled_//g -e s/.dat//g > filenumber1.dat" %(n_x,n_x,n_x,L,U1,n_x,n_x,n_x,L,U1))
os.system("ls -l N%dx%dx%d_L%d_U%d_Mu0_T_shuffled_*.dat | awk '{print $9}' | sed -e s/N%dx%dx%d_L%d_U%d_Mu0_T_shuffled_//g -e s/.dat//g > filenumber2.dat" %(n_x,n_x,n_x,L,U2,n_x,n_x,n_x,L,U2))

# Load filenumber
filenumber1 = np.genfromtxt("filenumber1.dat")
filenumber2 = np.genfromtxt("filenumber2.dat")

if len(filenumber1) < len(filenumber2) :
    half_nfile = len(filenumber1)
elif len(filenumber1) > len(filenumber2) :
    half_nfile = len(filenumber2)
else :
    half_nfile = len(filenumber1)

# Rename files
if U1 < U2 :
    for i in range(half_nfile):
       os.system("mv N%dx%dx%d_L%.2d_U%d_Mu0_T_shuffled_%.2d.dat N%dx%dx%d_L%d_U%d+U%d_Mu0_T_shuffled_%.2d.dat" % (n_x,n_x,n_x,L,U1,filenumber1[i],n_x,n_x,n_x,L,U1,U2,(2*i+1)))
       os.system("mv N%dx%dx%d_L%.2d_U%d_Mu0_T_shuffled_%.2d.dat N%dx%dx%d_L%d_U%d+U%d_Mu0_T_shuffled_%.2d.dat" % (n_x,n_x,n_x,L,U2,filenumber1[i],n_x,n_x,n_x,L,U1,U2,2*(i+1)))
else :
    for i in range(half_nfile):
       os.system("mv N%dx%dx%d_L%.2d_U%d_Mu0_T_shuffled_%.2d.dat N%dx%dx%d_L%d_U%d+U%d_Mu0_T_shuffled_%.2d.dat" % (n_x,n_x,n_x,L,U2,filenumber1[i],n_x,n_x,n_x,L,U2,U1,(2*i+1)))
       os.system("mv N%dx%dx%d_L%.2d_U%d_Mu0_T_shuffled_%.2d.dat N%dx%dx%d_L%d_U%d+U%d_Mu0_T_shuffled_%.2d.dat" % (n_x,n_x,n_x,L,U1,filenumber1[i],n_x,n_x,n_x,L,U2,U1,2*(i+1)))

os.remove("filenumber1.dat")
os.remove("filenumber2.dat")

print "Done."
