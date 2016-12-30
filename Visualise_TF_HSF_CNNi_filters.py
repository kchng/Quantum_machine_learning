import bpy
import glob
import numpy as np
import sys
import time

############################################### User control area ###############################################

# Note: If you would like to look at the filters in the 3D convolutional layers, change three_dimensions to True
# or other to False if you would like to look at the weights in the fully-connected layers instead.
three_dimensions=True
# Make sure you place all of the filter files in the same directory and key-in the absolute file path here. Use
# the asterisk symbol * appropriate so that all of the relevant files will be loaded. This is so that all of the
# values will be normalized by the greatest value.
# Windows
# Absolute_file_pathv = "C:\\Users\\phoni\\Desktop\\20160827-1803_model_U5+U16_CNN0i_test_acc_87.9_W_*.dat"
# MacOS
# Absolute_file_path = "/Users/kelvinchng/Desktop/20160827-1803_model_U5+U16_CNN0i_test_acc_87.9_W_*.dat"
# Linux
Absolute_file_path = "/home/kelvin/Desktop/HSF Tensor Flow/Blender/20160819-1545_model_U16_CNN0f_test_acc_92.1_W_*.dat"
Absolute_file_path1 = "/Users/kelvinchng/Desktop/20160814-2323_model_U5_CNN0f_test_acc_84.6_W_*.dat"
Absolute_file_path2 = "/Users/kelvinchng/Desktop/20160819-1545_model_U16_CNN0f_test_acc_92.1_W_*.dat"
all_data_filenames = glob.glob(Absolute_file_path1)


# Give the layer index that you want to look at.
# For N = 4**3, there are 3 convolutional layers and make sure three_dimensions is set to True and there are two
# fully-connected layers and make sure to set three_dimensions to False. For N = 8**3, there are 4 convolutional
# layers and 2 fullly-connected layers.
ith_filter_layer_to_draw = 7

#################################################################################################################

def makeMaterial(name, diffuse, specular=(1,1,1), alpha=1, transparent=False, transparency=0.5):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = diffuse
    mat.diffuse_shader = 'LAMBERT'
    mat.diffuse_intensity = 1.0
    mat.specular_intensity = 0.0
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
    nn = 4096
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
                        if input_configurations[n] < 0 :
                            # If the weight value is negative, colour it red
                            setMaterial(bpy.context.object, makeMaterial('Red', (1,1+input_configurations[n],1+input_configurations[n])))
                        else :
                            # If the weight value is positive, colour it blue
                            setMaterial(bpy.context.object, makeMaterial('Blue', (1-input_configurations[n],1-input_configurations[n],1)))
                        n += 1
                        if n == nn :
                            break
                    if n == nn :
                        break                                      
                if n == nn :
                    break
            if n == nn :
                break
        if n == nn :
            break
                        
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

def Find_max_val(absolute_file_path1,absolute_file_path2) :
    data_filenames = glob.glob(Absolute_file_path1) + glob.glob(Absolute_file_path2)

    # Find the maximum and minimum values in the filter
    W_mag_max = []
    for i in range(len(data_filenames)) :
        data_tmp = np.loadtxt(data_filenames[i])
        W_mag_max.append(abs(data_tmp).max())

    print('Global maximum value:', max(W_mag_max))
    return max(W_mag_max)

def deg_to_rad( deg ):
    return deg*np.pi/180

################################################ End of Function ################################################

# Delete all the cubes
for ob in bpy.context.scene.objects:
    ob.select = ob.type == 'MESH' and ob.name.startswith("Cube")
bpy.ops.object.delete()

# Set background colour
bpy.data.worlds["World"].horizon_color = (0,0,0)

# Spatial filter size: filter depth, height, and width
filter_d = 2
filter_h = filter_d
filter_w = filter_d

# Size of cube
cube_size = 0.5

# Read data
W_conv1 = np.loadtxt(all_data_filenames[ith_filter_layer_to_draw-1])

print('Current layer maximum value:', W_conv1.max())

# Plot the first 100 largest filters
W_conv1 = np.reshape(W_conv1.T.ravel()[:800],(10,80)).T

# Find the maximum and minimum values in the filter
W_mag_max = Find_max_val(Absolute_file_path1,Absolute_file_path2)

# Normalise data
W_conv1 = W_conv1/W_mag_max

print('Normalised current layer maximum value:', W_conv1.max())

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

W_conv1 = W_conv1.T.ravel()
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

# Set camera resolution, location, and angle.
scene = bpy.data.scenes["Scene"]
scene.render.resolution_x = 1080
scene.render.resolution_y = 1080 
scene.render.resolution_percentage = 200
scene.camera.rotation_mode = 'XYZ'
bpy.data.cameras["Camera"].clip_end = 100000

# Reposition camera
scene.camera.location.x = 19.0
scene.camera.location.y = 0
scene.camera.location.z = 0
scene.camera.rotation_euler[0] = deg_to_rad(-90.)
scene.camera.rotation_euler[1] = deg_to_rad(180.)
scene.camera.rotation_euler[2] = deg_to_rad(-90)

# Change lamp to Hemi
lamp = bpy.data.lamps["Lamp"]
lamp_obj = bpy.data.objects["Lamp"]
lamp.type = 'HEMI'
lamp_obj.location = (100.0,0.0,0.0)
lamp_obj.scale = (10.,10.,10.)
lamp_obj.rotation_euler[0] = deg_to_rad(90.0)
lamp_obj.rotation_euler[1] = deg_to_rad(0.0)
lamp_obj.rotation_euler[2] = deg_to_rad(90.0)

