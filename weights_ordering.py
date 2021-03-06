import numpy as np
import glob

Absolute_file_path = "/Users/kelvinchng/Desktop/20160814-2323_model_U5_CNN0f_test_acc_84.6_W_*.dat"
Absolute_file_path = "/Users/kelvinchng/Desktop/20160819-1545_model_U16_CNN0f_test_acc_92.1_W_*.dat"
all_data_filenames = glob.glob(Absolute_file_path)

# Find the maximum and minimum values in the filter
W_mag_max = []
for i in range(len(all_data_filenames)) :
    data_tmp = np.loadtxt(all_data_filenames[i])
    print(i,abs(data_tmp).max(),np.shape(data_tmp))
    W_mag_max.append(abs(data_tmp).max())

ith_filter_layer_to_draw = 1

# Read data
W_conv1 = np.loadtxt(all_data_filenames[ith_filter_layer_to_draw-1])

print all_data_filenames[ith_filter_layer_to_draw-1]

n_cols, n_rows = np.shape(W_conv1)

# Spatial filter size: filter depth, height, and width
filter_d = 2
filter_h = filter_d
filter_w = filter_d

n_cubes = n_cols/8*n_rows

W_arranged = np.zeros((n_cubes,filter_d**3+1))
n=0
for i in range(n_rows) :
    for j in range(n_cols/8) :
	W_arranged[n,:8] = W_conv1[j*8:(j+1)*8,i]
        W_arranged[n,-1] = sum(np.abs(W_conv1[j*8:(j+1)*8,i]))
	n += 1

sort_index = W_arranged[:,-1].argsort()[::-1]
W_arranged[:,:8] = W_arranged[:,:8][sort_index]
W_arranged[:,-1] = W_arranged[:,-1][sort_index]

output_data = np.reshape(W_arranged[:,:8],(n_rows,n_cols)).T

print np.shape(output_data)

#print np.savetxt("20160814-2323_model_U5_CNN0f_test_acc_84.6_W_conv1a.dat",output_data)
#print np.savetxt("20160819-1545_model_U16_CNN0f_test_acc_92.1_W_conv3a.dat",output_data)
