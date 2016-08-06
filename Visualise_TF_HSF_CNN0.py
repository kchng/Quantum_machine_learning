import bpy
import numpy as np

def makeMaterial(name, diffuse, specular, alpha, transparent=False, t_alpha = 0.5):
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
        mat.alpha = t_alpha
    return mat

def setMaterial(ob, mat):
    me = ob.data
    me.materials.append(mat)

def makecubes( ncube, cube_size, positions_x, positions_y, configurations ) :
    black = makeMaterial('Black', (0,0,0),(1,1,1),1)
    white = makeMaterial('White', (1,1,1),(1,1,1),1)
    add_cube = bpy.ops.mesh.primitive_cube_add
    offset = ncube*cube_size/2.-cube_size/2.
    n = 0
    for l in range(len(positions_x)) :
        for i in range(ncube) :
            for j in range(ncube) :
                for k in range(ncube) :
                    if j == 0 or k == (ncube-1) or (i==(ncube-1) and j>0) or (i==(ncube-1) and k>0):
                        add_cube(location=(i*cube_size+positions_x[l]-offset,j*cube_size+positions_y[l]-offset,k*cube_size-offset))
                        bpy.context.object.scale = (cube_size/2,cube_size/2,cube_size/2)
                        setMaterial(bpy.context.object, makeMaterial('Color', (configurations[n],configurations[n],configurations[n]),(1,1,1),1))
                    n += 1

def makeplane( length, width, x, y, z, colour ):
    add_cube = bpy.ops.mesh.primitive_cube_add
    add_cube(location=(x,y,z))
    bpy.context.object.scale = (width/2.,length/2.,0.)
    setMaterial(bpy.context.object, makeMaterial('Color', colour,(1,1,1),1))

def makecircles( r, d, positions_x, positions_y, probability, colour, prob_colour=0.0 ):
    add_cylinder = bpy.ops.mesh.primitive_cylinder_add
    n = 0
    for l in range(len(positions_x)) :
        add_cylinder(location=(positions_x[l],positions_y[l],0.0))
        bpy.context.object.scale = (r,r,d)
        setMaterial(bpy.context.object, makeMaterial('Color', colour,(1,1,1),1, transparent=False))
        n += 1

    if probability.any() > 0 and prob_colour == 0.0 :
        for l in range(len(positions_x)) :
            add_cylinder(location=(positions_x[l],positions_y[l],probability[l]))
            bpy.context.object.scale = (r/2.,r/2.,probability[l])
            setMaterial(bpy.context.object, makeMaterial('Color', colour,(1,1,1),1, transparent=True))
            n += 1
    elif probability.any() > 0 and prob_colour != 0.0 :
        for l in range(len(positions_x)) :
            add_cylinder(location=(positions_x[l],positions_y[l],probability[l]))
            bpy.context.object.scale = (r/2.,r/2.,probability[l])
            setMaterial(bpy.context.object, makeMaterial('Color', prob_colour[l],(1,1,1),1, transparent=True))
            n += 1

def calc_position_of_ellipses( a0, a1 ) :
    middle = a0+abs(a0-a1)/2.
    delta_position = abs(a0-middle)/2.
    return np.linspace(a0+delta_position,a0+delta_position*3,3)

def make_neuron_connections(x_right, y_right, x_left_array, y_left_array):
    delta_x = x_right-x_left_array
    delta_y = y_right-y_left_array
    x_pivot = delta_x/2.+x_left_array
    y_pivot = -delta_y/2.+y_right
    length = np.sqrt(delta_x**2. + delta_y**2.)
    angle = -np.arctan(delta_y/delta_x)#*180./np.pi
    
    add_cylinder = bpy.ops.mesh.primitive_cylinder_add
    radius = 0.05
    for i in range(len(x_pivot)) :
        add_cylinder(rotation=(angle[i],np.pi/2.,0), location=(x_pivot[i],y_pivot[i],0.0))
        bpy.context.object.scale = (radius,radius,length[i]/2.)
        setMaterial(bpy.context.object, makeMaterial('Color', (0.25,0.25,0.25),(1,1,1),1, transparent=True, t_alpha = 0.25))

scene = bpy.data.scenes["Scene"]
scene.camera.resolution_x = 1920
scene.camera.resolution_x = 720
scene.camera.location.x = 73.53608
scene.camera.location.y = -46.89505
scene.camera.location.z = 32.10664
scene.camera.rotation_mode = 'XYZ'
scene.camera.rotation_euler[0] = 64.8*np.pi/180
scene.camera.rotation_euler[1] = 0*np.pi/180
scene.camera.rotation_euler[2] = 57.8*np.pi/180

lamp_data = bpy.data.lamps.new(name="Light source", type='HEMI')
lamp_object = bpy.data.objects.new(name="Light source", object_data=lamp_data)
scene.objects.link(lamp_object)
lamp_object.location = (22.5,0.,50.)
lamp_object.scale = (10.,10.,10.)
lamp_object.select = True
scene.objects.active = lamp_object


# System size
#   number of spin in each of the cube dimension
nspin = 4
#   number of "time" dimension
n_time_dimension = 200
#   Volume of cube
V_cube = (nspin)**3
#   Volume of tesseract
V4d = n_time_dimension*V_cube

# Input layer ------------------------------------------------------------------------
input_layer_filename = "/home/kelvin/Desktop/HSF Tensor Flow/HSF_N4x4x4_L200_U9_Mu0_UniformTGrid/input_layer.dat" 
configurations = np.loadtxt(input_layer_filename, unpack = True)
# Reduce n_time_dimension to 25
configurations = np.delete(configurations, np.arange(V_cube,V4d-V_cube*24))


cube_size = 0.25
# Offset from the origin
offset = cube_size
# y position of the cubes
input_layer_pos_y = np.arange(-39.,42.,3)-offset
# Remove the second and third cube
input_layer_pos_y = input_layer_pos_y[input_layer_pos_y!=input_layer_pos_y[1]]
input_layer_pos_y = input_layer_pos_y[input_layer_pos_y!=input_layer_pos_y[1]]
# x position of the cubes
input_layer_pos_x = np.zeros(np.shape(input_layer_pos_y))

# Draw the cubes
makecubes( nspin, cube_size, input_layer_pos_x, input_layer_pos_y, configurations )


input_layer_ellipses_pos_y = calc_position_of_ellipses( input_layer_pos_y[0], input_layer_pos_y[1] )
# x position of the ellipses
input_layer_ellipses_pos_x = np.zeros(np.shape(input_layer_ellipses_pos_y))
# Draw ellipses
makecircles( 0.25, 0, input_layer_ellipses_pos_x, input_layer_ellipses_pos_y, np.array([0.]), (0,0,0))


offset = cube_size
# Feature extraction layer -----------------------------------------------------------
feature_extraction_layer_filename = "/home/kelvin/Desktop/HSF Tensor Flow/HSF_N4x4x4_L200_U9_Mu0_UniformTGrid/feature_extraction_layer.dat" 
feature_maps = np.loadtxt(feature_extraction_layer_filename, unpack = True)
V_feature_map_cube = (nspin-1)**3
n_feature_map1 = 64
V4d_feature_map = n_feature_map1*V_feature_map_cube
# Reduce n_feature_map1 to 8
feature_maps = np.delete(feature_maps, np.arange(V_feature_map_cube, V4d_feature_map-V_feature_map_cube*24))
# Normalize the value of feature_maps for use as an input to greyscale colour.
feature_maps = feature_maps/feature_maps.max()

# y position of the cubes
feature_map1_pos_y = np.arange(-13.5,16.5,3)
# Remove the second and third cube
feature_map1_pos_y = feature_map1_pos_y[feature_map1_pos_y!=feature_map1_pos_y[1]]
feature_map1_pos_y = feature_map1_pos_y[feature_map1_pos_y!=feature_map1_pos_y[1]]
print(feature_map1_pos_y)
# x position of the cubes
feature_map1_pos_x = np.zeros(np.shape(feature_map1_pos_y))+15

# Draw the cubes
makecubes( 3, cube_size, feature_map1_pos_x, feature_map1_pos_y, feature_maps )

feature_map1_ellipses_pos_y = calc_position_of_ellipses( feature_map1_pos_y[0], feature_map1_pos_y[1] )
# x position of the ellipses
feature_map1_ellipses_pos_x = np.zeros(np.shape(feature_map1_ellipses_pos_y))+15
# Draw ellipses
makecircles( 0.25, 0, feature_map1_ellipses_pos_x, feature_map1_ellipses_pos_y, np.array([0.]), (0,0,0))

# Draw the connections
for i in range(len(feature_map1_pos_x)):
    make_neuron_connections(feature_map1_pos_x[i], feature_map1_pos_y[i], input_layer_pos_x, input_layer_pos_y)

makeplane( abs(feature_map1_pos_y[-1]-feature_map1_pos_y[0])+6, 10., 15, 0, -(cube_size*3)/2., (0.90, 0.60, 0.10))

# Fully connected layer --------------------------------------------------------------
output_filename = "/home/kelvin/Desktop/HSF Tensor Flow/HSF_N4x4x4_L200_U9_Mu0_UniformTGrid/fully_connected_layer1.dat" 
output = np.loadtxt(output_filename, unpack = True)
# Normalize the output of neurons
output = np.delete(output, np.arange(1,64-7))/output.max()*2
fc1_pos_y = np.arange(-13.5,16.5,3)
fc1_pos_y = fc1_pos_y[fc1_pos_y!=fc1_pos_y[1]]
fc1_pos_y = fc1_pos_y[fc1_pos_y!=fc1_pos_y[1]]
fc1_pos_x = np.zeros(np.shape(fc1_pos_y))+30
# Radius of circle
r = 0.75
# Height of cylinder
d = 0.1

# Draw the output of fully-connected neurons and it's output
makecircles( r, d, fc1_pos_x, fc1_pos_y, output, (0.25,0.25,0.25))

fc1_pos_ellipses_y = calc_position_of_ellipses( fc1_pos_y[0], fc1_pos_y[1] )
# x position of the ellipses
fc1_pos_ellipses_x = np.zeros(np.shape(fc1_pos_ellipses_y))+30
# Draw ellipses
makecircles( 0.25, 0, fc1_pos_ellipses_x, fc1_pos_ellipses_y, np.array([0.]), (0,0,0))

# Draw the connections
for i in range(len(fc1_pos_x)):
    make_neuron_connections(fc1_pos_x[i], fc1_pos_y[i], feature_map1_pos_x, feature_map1_pos_y)


# Output layer -----------------------------------------------------------------------
output_filename = "/home/kelvin/Desktop/HSF Tensor Flow/HSF_N4x4x4_L200_U9_Mu0_UniformTGrid/output.dat" 
probability = np.loadtxt(output_filename, unpack = True)*2
output_pos_y = np.arange(-1.5,4.5,3)
output_pos_x = np.zeros(np.shape(output_pos_y))+45

output_prob_colour = [ [0.95, 0.35, 0.00], [0.20, 0.60, 0.90] ]

# Draw the output neurons and it's output probability
makecircles( r, d, output_pos_x, output_pos_y, probability, (0.25,0.25,0.25), output_prob_colour)

# Draw the connections
for i in range(len(output_pos_x)):
    make_neuron_connections(output_pos_x[i], output_pos_y[i], fc1_pos_x, fc1_pos_y)

cylinder_size = 0.10
makeplane( abs(fc1_pos_y[-1]-fc1_pos_y[0])+6, 25., 37.5, 0, -(cylinder_size*3)/2., (0.55, 0.90, 0.25))
