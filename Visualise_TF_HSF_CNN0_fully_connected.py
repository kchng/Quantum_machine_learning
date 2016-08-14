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
            setMaterial(bpy.context.object, makeMaterial('Color', colour,(1,1,1),1, transparent=False))
            n += 1
    elif probability.any() > 0 and prob_colour != 0.0 :
        for l in range(len(positions_x)) :
            add_cylinder(location=(positions_x[l],positions_y[l],probability[l]))
            bpy.context.object.scale = (r/2.,r/2.,probability[l])
            setMaterial(bpy.context.object, makeMaterial('Color', prob_colour[l],(1,1,1),1, transparent=False))
            n += 1
            
def maketorus( r, d, positions_x, positions_y, colour=[1,1,1]):
    add_torus = bpy.ops.mesh.primitive_torus_add
    n = 0
    if len(colour) == 1 :
        for l in range(len(positions_x)) :
            add_torus(location=(positions_x[l],positions_y[l],0.0))
            bpy.context.object.scale = (r,r,d)
            setMaterial(bpy.context.object, makeMaterial('Color', colour,(1,1,1),1, transparent=False))
            n += 1
    else :
        for l in range(len(positions_x)) :
            add_torus(location=(positions_x[l],positions_y[l],0.0))
            bpy.context.object.scale = (r,r,d)
            setMaterial(bpy.context.object, makeMaterial('Color', colour[l],(1,1,1),1, transparent=False))
            n += 1

def calc_position_of_ellipses( a0, a1 ) :
    middle = a0+abs(a0-a1)/2.
    delta_position = abs(a0-middle)/2.
    return np.linspace(a0+delta_position,a0+delta_position*3,3)

def make_neuron_connections(x_right, y_right, x_left_array, y_left_array, z_right = 0.0, z_left_array=0.0):
    if len(np.hstack((z_left_array,[]))) == 1 :
        z_left_array = np.zeros( len(np.hstack((x_left_array,[]))) )
    delta_x = x_right-x_left_array
    delta_y = y_right-y_left_array
    delta_z = z_right-z_left_array
    x_pivot = delta_x/2.+x_left_array
    y_pivot = -delta_y/2.+y_right
    z_pivot = delta_z/2.+z_left_array
    length = np.sqrt(delta_x**2. + delta_y**2.+delta_z**2.)
    angle = -np.arctan(delta_y/delta_x)
    phi = -np.arctan(delta_z/np.sqrt(delta_x**2. + delta_y**2.))
    
    add_cylinder = bpy.ops.mesh.primitive_cylinder_add
    radius = 0.05
    for i in range(len(x_pivot)) :
        add_cylinder(rotation=(angle[i],np.pi/2.+phi[i],0), location=(x_pivot[i],y_pivot[i],z_pivot[i]))
        bpy.context.object.scale = (radius,radius,length[i]/2.)
        setMaterial(bpy.context.object, makeMaterial('Color', (0.25,0.25,0.25),(1,1,1),1, transparent=True, t_alpha = 0.50))

def deg_to_rad( deg ):
    return deg*np.pi/180

def calc_feature_map_neuron_loc( n_feature_map, ncube, cube_size, positions_x, positions_y ) :

    offset = ncube*cube_size/2.-cube_size/2.
    n_neuron = n_feature_map*((ncube**2)+(ncube-1)**2+ncube*(ncube-1))
    n = 0
    neuron_loc_x = np.zeros(n_neuron)
    neuron_loc_y = np.zeros(n_neuron)
    neuron_loc_z = np.zeros(n_neuron)
    for l in range(len(positions_x)) :
        for i in range(ncube) :
            for j in range(ncube) :
                for k in range(ncube) :
                    if j == 0 or k == (ncube-1) or (i==(ncube-1) and j>0) or (i==(ncube-1) and k>0):
                        neuron_loc_x[n] = i*cube_size+positions_x[l]-offset
                        neuron_loc_y[n] = j*cube_size+positions_y[l]-offset
                        neuron_loc_z[n] = k*cube_size-offset
                        n+=1
    return neuron_loc_x, neuron_loc_y, neuron_loc_z

# Change lamp to Hemi
lamp = bpy.data.lamps["Lamp"]
lamp_obj = bpy.data.objects["Lamp"]
lamp.type = 'HEMI'
lamp_obj.location = (22.5,0.,50.)
lamp_obj.scale = (10.,10.,10.)
lamp_obj.rotation_euler[0] = deg_to_rad(37.261)
lamp_obj.rotation_euler[1] = deg_to_rad(3.164)
lamp_obj.rotation_euler[2] = deg_to_rad(106.936)

# Add new light source
#lamp_data = bpy.data.lamps.new(name="Light source", type='HEMI')
#lamp_object = bpy.data.objects.new(name="Light source", object_data=lamp_data)
#scene.objects.link(lamp_object)
#lamp_object.location = (22.5,0.,50.)
#lamp_object.scale = (10.,10.,10.)
#lamp_object.select = True
#scene.objects.active = lamp_object

# Set background colour
bpy.data.worlds["World"].horizon_color = (1,1,1)

#bpy.ops.render.render(use_viewport=True)


# System size
#   number of spin in each of the cube dimension
nspin = 4
#   number of imaginary time dimension
n_time_dimension = 200
#   Volume of cube
V_cube = (nspin)**3
#   Volume of tesseract
V4d = n_time_dimension*V_cube
# Number of input "neuron" to be drawn
draw_n_input = 10
# Distance between cube
delta_x_cube = 3
# Size of cube
cube_size = 0.5
# Separation between layer
delta_x_layer = 5.

def gen_neuron_pos_y( end_pos_y, delta_x ) :
    return np.arange(-end_pos_y,end_pos_y+delta_x,delta_x)




'''
# position of end fully-connected neuron
fc1_end_pos_y = 6
fc1_pos_y = gen_neuron_pos_y(fc1_end_pos_y, delta_x_cube)
#fc1_pos_y = fc1_pos_y[fc1_pos_y!=fc1_pos_y[1]]
#fc1_pos_y = fc1_pos_y[fc1_pos_y!=fc1_pos_y[1]]
fc1_pos_x = np.zeros(np.shape(fc1_pos_y))+0
# Radius of circle
r = 0.75
# Height of cylinder
d = 0.1

#output_prob_colour  = [[0.40, 0.20, 0.95]]*draw_n_fc1 

# Draw the output of fully-connected neurons and it's output
makecircles( r, d, fc1_pos_x, fc1_pos_y, np.array([0.]), (0.25,0.25,0.25) )

#fc1_pos_ellipses_y = calc_position_of_ellipses( fc1_pos_y[0], fc1_pos_y[1] )
# x position of the ellipses
#fc1_pos_ellipses_x = np.zeros(np.shape(fc1_pos_ellipses_y))+delta_x_layer
# Draw ellipses
#makecircles( 0.25, 0, fc1_pos_ellipses_x, fc1_pos_ellipses_y, np.array([0.]), (0,0,0))

# position of end fully-connected neuron
fc2_end_pos_y = 4.5
fc2_pos_y = gen_neuron_pos_y(fc2_end_pos_y, delta_x_cube)
#fc2_pos_y = fc2_pos_y[fc2_pos_y!=fc2_pos_y[1]]
#fc2_pos_y = fc2_pos_y[fc2_pos_y!=fc2_pos_y[1]]
fc2_pos_x = np.zeros(np.shape(fc2_pos_y))+delta_x_layer
# Radius of circle
r = 0.75
# Height of cylinder
d = 0.1

#output_prob_colour  = [[0.40, 0.20, 0.95]]*draw_n_fc2 

# Draw the output of fully-connected neurons and it's output
makecircles( r, d, fc2_pos_x, fc2_pos_y, np.array([0.]), (0.25,0.25,0.25) )

#fc2_pos_ellipses_y = calc_position_of_ellipses( fc2_pos_y[0], fc2_pos_y[1] )
# x position of the ellipses
#fc2_pos_ellipses_x = np.zeros(np.shape(fc2_pos_ellipses_y))+delta_x_layer*2
# Draw ellipses
#makecircles( 0.25, 0, fc2_pos_ellipses_x, fc2_pos_ellipses_y, np.array([0.]), (0,0,0))

# Draw the connections
for i in range(len(fc2_pos_x)):
    make_neuron_connections(fc2_pos_x[i], fc2_pos_y[i], fc1_pos_x, fc1_pos_y)

# Draw all the connections
#for i in range(len(fc1_pos_x)):
#    make_neuron_connections(fc1_pos_x[i], fc1_pos_y[i], feature_map1_neuron_x, feature_map1_neuron_y, z_left_array = feature_map1_neuron_z)

# position of end fully-connected neuron
fc3_end_pos_y = 1.5
fc3_pos_y = gen_neuron_pos_y(fc3_end_pos_y, delta_x_cube)
#fc3_pos_y = fc3_pos_y[fc3_pos_y!=fc3_pos_y[1]]
#fc3_pos_y = fc3_pos_y[fc3_pos_y!=fc3_pos_y[1]]
fc3_pos_x = np.zeros(np.shape(fc3_pos_y))+delta_x_layer*2
# Radius of circle
r = 0.75
# Height of cylinder
d = 0.1

#output_prob_colour  = [[0.40, 0.20, 0.95]]*draw_n_fc3 

# Draw the output of fully-connected neurons and it's output
makecircles( r, d, fc3_pos_x, fc3_pos_y, np.array([0.]), (0.25,0.25,0.25) )

#fc3_pos_ellipses_y = calc_position_of_ellipses( fc3_pos_y[0], fc3_pos_y[1] )
# x position of the ellipses
#fc3_pos_ellipses_x = np.zeros(np.shape(fc3_pos_ellipses_y))+delta_x_layer*2
# Draw ellipses
#makecircles( 0.25, 0, fc3_pos_ellipses_x, fc3_pos_ellipses_y, np.array([0.]), (0,0,0))

# Draw the connections
for i in range(len(fc3_pos_x)):
    make_neuron_connections(fc3_pos_x[i], fc3_pos_y[i], fc2_pos_x, fc2_pos_y)
'''

'''
# Input layer ------------------------------------------------------------------------
input_layer_filename = "/home/kelvin/Desktop/HSF Tensor Flow/HSF_N4x4x4_L200_U9_Mu0_UniformTGrid/input_layer.dat" 
configurations = np.loadtxt(input_layer_filename, unpack = True)
# Reduce n_time_dimension to draw_n_input
configurations = np.delete(configurations, np.arange(V_cube,V4d-V_cube*(draw_n_input-1)))

# position of end cube
input_layer_end_pos_y = float(draw_n_input+1)*delta_x_cube/2.
# y position of the cubes
input_layer_pos_y = gen_neuron_pos_y(input_layer_end_pos_y, delta_x_cube)
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

makeplane( abs(input_layer_pos_y [-1]-input_layer_pos_y [0])+6, delta_x_layer*2/3., 0, 0, -(cube_size*4)/2., (0.20, 0.60, 0.95))

# Feature extraction layer -----------------------------------------------------------
feature_extraction_layer_filename = "/home/kelvin/Desktop/HSF Tensor Flow/HSF_N4x4x4_L200_U9_Mu0_UniformTGrid/feature_extraction_layer.dat" 
feature_maps = np.loadtxt(feature_extraction_layer_filename, unpack = True)
V_feature_map_cube = (nspin-1)**3
n_feature_map1 = 64
# Number of feature map 1 to be drawn
draw_n_feature_map1 = round(float(draw_n_input)/n_time_dimension*float(n_feature_map1))
V4d_feature_map = n_feature_map1*V_feature_map_cube
# Reduce n_feature_map1 to 8
feature_maps = np.delete(feature_maps, np.arange(V_feature_map_cube, V4d_feature_map-V_feature_map_cube*(draw_n_feature_map1-1)))
# Normalize the value of feature_maps for use as an input to greyscale colour.
feature_maps = feature_maps/feature_maps.max()

# position of end feature map cube
feature_map1_end_pos_y = float(draw_n_feature_map1+1)*delta_x_cube/2.
# y position of the cubes
feature_map1_pos_y = gen_neuron_pos_y(feature_map1_end_pos_y, delta_x_cube)
# Remove the second and third cube
feature_map1_pos_y = feature_map1_pos_y[feature_map1_pos_y!=feature_map1_pos_y[1]]
feature_map1_pos_y = feature_map1_pos_y[feature_map1_pos_y!=feature_map1_pos_y[1]]

# x position of the cubes
feature_map1_pos_x = np.zeros(np.shape(feature_map1_pos_y))+delta_x_layer

feature_map1_neuron_x, feature_map1_neuron_y, feature_map1_neuron_z = calc_feature_map_neuron_loc( 3, draw_n_feature_map1, cube_size, feature_map1_pos_x, feature_map1_pos_y )

# Draw the cubes
makecubes( 3, cube_size, feature_map1_pos_x, feature_map1_pos_y, feature_maps )

feature_map1_ellipses_pos_y = calc_position_of_ellipses( feature_map1_pos_y[0], feature_map1_pos_y[1] )
# x position of the ellipses
feature_map1_ellipses_pos_x = np.zeros(np.shape(feature_map1_ellipses_pos_y))+delta_x_layer
# Draw ellipses
makecircles( 0.25, 0, feature_map1_ellipses_pos_x, feature_map1_ellipses_pos_y, np.array([0.]), (0,0,0))

# Draw the connections
for i in range(len(feature_map1_pos_x)):
    make_neuron_connections(feature_map1_pos_x[i], feature_map1_pos_y[i], input_layer_pos_x, input_layer_pos_y)

makeplane( abs(feature_map1_pos_y[-1]-feature_map1_pos_y[0])+6, delta_x_layer*2/3., delta_x_layer, 0, -(cube_size*3)/2., (0.20, 0.60, 0.60)) #(0.90, 0.60, 0.10))


# Fully connected layer --------------------------------------------------------------
output_filename = "/home/kelvin/Desktop/HSF Tensor Flow/HSF_N4x4x4_L200_U9_Mu0_UniformTGrid/fully_connected_layer1.dat" 
output = np.loadtxt(output_filename, unpack = True)
# Number of fully-connected_neurons
n_fc1 = 64
# Number of fully-connected_neurons to be drawn
draw_n_fc1 = round(float(draw_n_input)/n_time_dimension*float(n_fc1))
# Normalize the output of neurons
output = np.delete(output, np.arange(1,n_fc1-(draw_n_fc1-1)))/output.max()*2

# position of end fully-connected neuron
fc1_end_pos_y = float(draw_n_fc1+1)*delta_x_cube/2.
fc1_pos_y = gen_neuron_pos_y(fc1_end_pos_y, delta_x_cube)
fc1_pos_y = fc1_pos_y[fc1_pos_y!=fc1_pos_y[1]]
fc1_pos_y = fc1_pos_y[fc1_pos_y!=fc1_pos_y[1]]
fc1_pos_x = np.zeros(np.shape(fc1_pos_y))+delta_x_layer*2
# Radius of circle
r = 0.75
# Height of cylinder
d = 0.1

output_prob_colour  = [[0.40, 0.20, 0.95]]*draw_n_fc1 

# Draw the output of fully-connected neurons and it's output
makecircles( r, d, fc1_pos_x, fc1_pos_y, output, (0.25,0.25,0.25), output_prob_colour )

fc1_pos_ellipses_y = calc_position_of_ellipses( fc1_pos_y[0], fc1_pos_y[1] )
# x position of the ellipses
fc1_pos_ellipses_x = np.zeros(np.shape(fc1_pos_ellipses_y))+delta_x_layer*2
# Draw ellipses
makecircles( 0.25, 0, fc1_pos_ellipses_x, fc1_pos_ellipses_y, np.array([0.]), (0,0,0))

# Draw the connections
for i in range(len(fc1_pos_x)):
    make_neuron_connections(fc1_pos_x[i], fc1_pos_y[i], feature_map1_pos_x, feature_map1_pos_y)

# Draw all the connections
#for i in range(len(fc1_pos_x)):
#    make_neuron_connections(fc1_pos_x[i], fc1_pos_y[i], feature_map1_neuron_x, feature_map1_neuron_y, z_left_array = feature_map1_neuron_z)


# Output layer -----------------------------------------------------------------------
output_filename = "/home/kelvin/Desktop/HSF Tensor Flow/HSF_N4x4x4_L200_U9_Mu0_UniformTGrid/output.dat" 
probability = np.loadtxt(output_filename, unpack = True)*2
output_pos_y = np.arange(-1.5,4.5,3)
output_pos_x = np.zeros(np.shape(output_pos_y))+delta_x_layer*3

output_prob_colour = [ [0.95, 0.35, 0.00], [0.20, 0.60, 0.90] ]

# Draw the output neurons and it's output probability
makecircles( r, d, output_pos_x, output_pos_y, probability, (0.25,0.25,0.25), output_prob_colour)

colour = [ [0.90, 0.25, 0.35], [0.20, 0.60, 0.90] ]
maketorus( 1.20, d, output_pos_x, output_pos_y, colour )

# Draw the connections
for i in range(len(output_pos_x)):
    make_neuron_connections(output_pos_x[i], output_pos_y[i], fc1_pos_x, fc1_pos_y)

cylinder_size = 0.10
makeplane( abs(fc1_pos_y[-1]-fc1_pos_y[0])+6, delta_x_layer*(1+2/3.), delta_x_layer*2.5, 0, -(cylinder_size*3)/2., (0.55, 0.90, 0.25))
'''



# Set camera resolution, location, and angle.
scene = bpy.data.scenes["Scene"]
scene.render.resolution_x = 1920
scene.render.resolution_y = 1440
scene.render.resolution_percentage = 200
scene.camera.rotation_mode = 'XYZ'
bpy.data.cameras["Camera"].clip_end = 100000
#scene.camera.location.x = 73.53608
#scene.camera.location.y = -46.89505
#scene.camera.location.z = 32.10664
#scene.camera.rotation_euler[0] = deg_to_rad(64.8)
#scene.camera.rotation_euler[1] = deg_to_rad(0)
#scene.camera.rotation_euler[2] = deg_to_rad(57.8)

radius = 22
theta = deg_to_rad( np.linspace(0.,45.,10) )
phi = deg_to_rad( np.linspace(0.,80.,9) )
camera_x_offset = delta_x_layer#*(3+1/3.)/2.

frame_num = 0
n=0
#scene.camera.location.x = camera_x_offset + radius*np.sin( theta[i])*np.cos( phi[j] )
#scene.camera.location.y = radius*np.sin( theta[i])*np.sin( phi[j] )
#scene.camera.location.z = radius*np.cos( theta[i])
#scene.camera.rotation_euler[0] = theta[i]
#scene.camera.rotation_euler[1] = phi[j]
#scene.camera.rotation_euler[2] = deg_to_rad(0)
scene.camera.location.x =0
scene.camera.location.y = 0
scene.camera.location.z = radius
scene.camera.rotation_euler[0] = 0
scene.camera.rotation_euler[1] = 0
scene.camera.rotation_euler[2] = 0


bpy.ops.object.empty_add(type='CIRCLE', location=(camera_x_offset,0,0))
camera_rig = bpy.data.objects["Empty"]
camera_obj = bpy.data.objects["Camera"]
camera_obj.parent = camera_rig

'''
n=0
for j in range(len(phi)) :
    for i in range(len(theta)) :
        bpy.context.scene.frame_set(n)
        camera_rig.rotation_euler = [theta[i], phi[j], 0]
        camera_rig.keyframe_insert(data_path="rotation_euler")
        scene.camera.rotation_euler[2] = phi[j]#+5*np.pi/180.
        scene.camera.keyframe_insert(data_path="rotation_euler")
        if n == 45 :
            print (theta[i],phi[j])
        #scene.render.filepath = '/home/kelvin/Desktop/HSF Tensor Flow/HSF_N4x4x4_L200_U9_Mu0_UniformTGrid/TF_HSF_CNN0_%.2d.png' %n
        #bpy.ops.render.render(write_still=True)
        n+=1
'''
'''
for j in range(len(phi)) :
    for i in range(len(theta)) :
        bpy.context.scene.frame_set(frame_num)
        scene.camera.location.x = camera_x_offset + radius*np.sin( theta[i])*np.cos( phi[j] )
        scene.camera.location.y = radius*np.sin( theta[i])*np.sin( phi[j] )
        scene.camera.location.z = radius*np.cos( theta[i])
        scene.camera.rotation_euler[0] = theta[i]
        scene.camera.rotation_euler[1] = phi[j]
        scene.camera.rotation_euler[2] = deg_to_rad(0)
        
        scene.camera.keyframe_insert(data_path="location", index=-1)
        scene.camera.keyframe_insert(data_path="rotation_euler")
        #scene.render.filepath = '/home/kelvin/Desktop/HSF Tensor Flow/HSF_N4x4x4_L200_U9_Mu0_UniformTGrid/TF_HSF_CNN0_%.2d.png' %n
        #bpy.ops.render.render(write_still=True)
        n+=1
        frame_num +=1
'''