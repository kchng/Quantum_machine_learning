import bpy
import glob
import numpy as np
import sys
import time

def makeMaterial(name, diffuse, specular=(1,1,1), alpha=1, transparent=False, transparency=0.5):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = diffuse
    mat.diffuse_shader = 'LAMBERT'
    mat.diffuse_intensity = 1.0
    mat.specular_intensity = 0.5
    mat.alpha = alpha
    mat.ambient = 1
    if transparent :
        mat.use_transparency = True
        mat.transparency_method = 'Z_TRANSPARENCY'
        mat.alpha = transparency
    return mat

def setMaterial(ob, mat):
    me = ob.data
    me.materials.append(mat)

def makecubes(ncube, cube_size, positions_x, positions_y, positions_z, input_configurations, three_dimensions=True) :
    add_cube = bpy.ops.mesh.primitive_cube_add
    offset = (ncube-1.0)*cube_size/2.
    cube_scale = cube_size/2.
    n = 0
    if three_dimensions :
        ncube_x = ncube
        offset_x = offset
    else :
        ncube = 1
        offset = 0   
    for m in range(len(positions_z)) :
        for l in range(len(positions_x)) :
            for k in range(ncube) :
                for j in range(ncube) :
                    for i in range(ncube) :
                        loc_x = i*cube_size+positions_x[l]-offset
                        loc_y = j*cube_size+positions_y[l]-offset
                        loc_z = (ncube-k-1)*cube_size+positions_z[m]-offset
                        
                        add_cube(location=(loc_x,loc_y,loc_z))
                        bpy.context.object.scale = (cube_scale,cube_scale,cube_scale)
                        if input_configurations[n] > 0 :
                            # If the weight value is negative, colour it red
                            setMaterial(bpy.context.object, makeMaterial('Red', (1-input_configurations[n],0,0)))
                            # If the weight value is positive, colour it blue
                        else :
                            setMaterial(bpy.context.object, makeMaterial('Blue', (0,0,1+input_configurations[n])))
                        n += 1
                                      
def generate_neuron_position(n_feature_to_map, delta_position) :
    # position of end feature map cube
    if np.mod(n_feature_to_map ,2) == 0 : 
        end_pos = float(n_feature_to_map-1)*delta_position/2.
    else :
        end_pos = float(n_feature_to_map)*delta_position/2.    
    if end_pos < 0:
        delta_position = -delta_position   
    # return position of the cubes in an array
    return np.arange(-end_pos,end_pos+delta_position,delta_position)

################################################ End of Function ################################################

three_dimensions=True
#input_data_filename = "C:\\Users\\phoni\\Desktop\\20160827-1803_model_U5+U16_CNN0i_test_acc_87.9_W_conv1.dat"
#input_data_filename = "/home/kelvin/Desktop/HSF Tensor Flow/Blender/20160827-1803_model_U5+U16_CNN0i_test_acc_87.9_W_conv4.dat"
#all_data_filenames = glob.glob("/Users/kelvinchng/Desktop/20160827-1803_model_U5+U16_CNN0i_test_acc_87.9_W_*.dat")
all_data_filenames = glob.glob("/home/kelvin/Desktop/HSF Tensor Flow/Blender/20160819-1545_model_U16_CNN0f_test_acc_92.1_W_*.dat")


# Find the maximum and minimum values in the filter
W_mag_max = []
W_mag_min = []
for i in range(len(all_data_filenames)) :
    data_tmp = np.loadtxt(all_data_filenames[i])
    W_mag_max.append(data_tmp.max())
    W_mag_min.append(data_tmp.min())

# Delete the initial cube
for ob in bpy.context.scene.objects:
    ob.select = ob.type == 'MESH' and ob.name.startswith("Cube")
bpy.ops.object.delete()

# Spatial filter size: filter depth, height, and width
filter_d = 2
filter_h = filter_d
filter_w = filter_d

# Size of cube
cube_size = 0.5
         
# Read data
W_conv1 = np.loadtxt(all_data_filenames[2])

print(all_data_filenames)
print(np.shape(W_conv1))

# Find the largest weight magnitude
W_mag_max = np.array([max(W_mag_max),np.abs(min(W_mag_min))]).max()

# Normalise data
W_conv1[W_conv1>0] = W_conv1[W_conv1>0]/W_mag_max
W_conv1[W_conv1<0] = W_conv1[W_conv1<0]/W_mag_max

n_row, n_col = np.shape(W_conv1)
if three_dimensions :
    n_y_cube, n_z_cube = n_row/(filter_d)**3, n_col
    # Separation
    delta_x_cube = (filter_d+1)*cube_size
else :
    n_y_cube, n_z_cube = n_row, n_col
    # Separation
    delta_x_cube = cube_size

delta_x_layer = 0
# y position of the cubes
y_conv1 = generate_neuron_position(n_y_cube, delta_x_cube)
# z position of the cubes
z_conv1 = -generate_neuron_position(n_z_cube, delta_x_cube)
# x position of the cubes
x_conv1 = np.zeros(np.shape(y_conv1))+delta_x_layer

W_conv1 = np.reshape(W_conv1.T,(np.prod(np.array(np.shape(W_conv1))),1))

start = time.clock()

# Draw the cubes
makecubes(filter_d,
          cube_size,
          x_conv1,
          y_conv1,
          z_conv1,
          W_conv1,
          three_dimensions)

print('time elapsed: %.2g s'%(time.clock()-start))
