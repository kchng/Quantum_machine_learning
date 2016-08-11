import cubic_spline_interpolation
import matplotlib.pyplot as plt
import numpy as np
import sys

filename_measurements = '20160810-0955_measurements_CNN0a.dat'
filename_result = '20160810-0955_result_CNN0a.dat'

filename_measurements = '20160810-1104_measurements_CNN0a.dat'
filename_result = '20160810-1104_result_CNN0a.dat'

#filename_measurements = '20160803-0833_measurements.dat'
#filename_result = '20160803-0833_result.dat'

title = '$\mathrm{Network:\ TF\_HSF\_CNN0a.py}$'

# temperature_U1 index_U1 closest to T_c
index_U1 = 9

# Potential energy data set 1
U1 = 4

# Critical temperature_U1
T_c_U1= 0.16

# Initial guess solution of critical temperature_U1
T_c_guess_U1 = 0.105

# temperature_U2 index_U2 closest to T_c
index_U2 = 20

# Potential energy data set 2
U2 = 20

# Critical temperature_U2
T_c_U2= 0.19

# Initial guess solution of critical temperature_U2
T_c_guess_U2 = 0.19

T_c_U1_known = False

use_single_U = False
U1_temp_len = 36

# 'equal' or 'log'
grid = 'equal'
grid = 'log'
# 'cubic' or 'linear'
interpolation = 'cubic'
interpolation = 'linear'

def quadratic1_U1( x ):
    T = temperature_U1[index_U1]
    a, b, c, d = params_a1[index_U1], params_b1[index_U1], params_c1[index_U1], params_d1[index_U1]
    return a + b*(x-T) + c*(x-T)**2. + d*(x-T)**3., b + 2.*c*(x-T) + 3.*d*(x-T)**2.

def quadratic2_U1( x ):
    T = temperature_U1[index_U1]
    a, b, c, d = params_a2[index_U1], params_b2[index_U1], params_c2[index_U1], params_d2[index_U1]
    return a + b*(x-T) + c*(x-T)**2. + d*(x-T)**3., b + 2.*c*(x-T) + 3.*d*(x-T)**2.

def quadratic1_U2( x ):
    T = temperature_U2[index_U2]
    a, b, c, d = params_a1[index_U2], params_b1[index_U2], params_c1[index_U2], params_d1[index_U2]
    return a + b*(x-T) + c*(x-T)**2. + d*(x-T)**3., b + 2.*c*(x-T) + 3.*d*(x-T)**2.

def quadratic2_U2( x ):
    T = temperature_U2[index_U2]
    a, b, c, d = params_a2[index_U2], params_b2[index_U2], params_c2[index_U2], params_d2[index_U2]
    return a + b*(x-T) + c*(x-T)**2. + d*(x-T)**3., b + 2.*c*(x-T) + 3.*d*(x-T)**2.    
            
def linear1_U1( x ):
    delta_y = (output_neuron1_U1[index_U1+1]-output_neuron1_U1[index_U1])
    delta_x = (temperature_U1[index_U1+1]-temperature_U1[index_U1])
    b = output_neuron1_U1[index_U1] - delta_y*temperature_U1[index_U1]/delta_x
    return delta_y*x/delta_x+b, delta_y/delta_x

def linear2_U1( x ):
    delta_y = (output_neuron2_U1[index_U1+1]-output_neuron2_U1[index_U1])
    delta_x = (temperature_U1[index_U1+1]-temperature_U1[index_U1])
    b = output_neuron2_U1[index_U1] - delta_y*temperature_U1[index_U1]/delta_x
    return delta_y*x/delta_x+b, delta_y/delta_x
    
def linear1_U2( x ):
    delta_y = (output_neuron1_U2[index_U2+1]-output_neuron1_U2[index_U2])
    delta_x = (temperature_U2[index_U2+1]-temperature_U2[index_U2])
    b = output_neuron1_U2[index_U2] - delta_y*temperature_U2[index_U2]/delta_x
    return delta_y*x/delta_x+b, delta_y/delta_x

def linear2_U2( x ):
    delta_y = (output_neuron2_U2[index_U2+1]-output_neuron2_U2[index_U2])
    delta_x = (temperature_U2[index_U2+1]-temperature_U2[index_U2])
    b = output_neuron2_U2[index_U2] - delta_y*temperature_U2[index_U2]/delta_x
    return delta_y*x/delta_x+b, delta_y/delta_x
    
def dx(f, g, x):
    return abs(g(x)[0]-f(x)[0])

def newtons_method(f, g, x0, e = 10e-10):
    delta = dx(f, g, x0)
    while delta > e:
        x0 = x0 - (f(x0)[0] - g(x0)[0])/(f(x0)[1] - g(x0)[1])
        delta = dx(f, g, x0)
    return x0

date = filename_measurements.rsplit('_',2)[0]

data_measurements = np.loadtxt(filename_measurements)

training_epochs   = data_measurements[:,0]
training_accuracy = data_measurements[:,1]
test_accuracy     = data_measurements[:,2]
cost              = data_measurements[:,3]

data_result = np.loadtxt(filename_result)

if use_single_U :
    temperature_U1       = data_result[:,0]
    output_neuron2_U1    = data_result[:,1]
    output_neuron1_U1    = data_result[:,2]
    accuracy_U1          = data_result[:,3]
    
    if interpolation == 'linear' :
        
        #m1 = (output_neuron1_U1[index_U1+1]-output_neuron1_U1[index_U1])/(temperature_U1[index_U1+1]-temperature_U1[index_U1])
        #b1 = output_neuron1_U1[index_U1+1] - m1*temperature_U1[index_U1+1]
        #m2 = (output_neuron2_U1[index_U1+1]-output_neuron2_U1[index_U1])/(temperature_U1[index_U1+1]-temperature_U1[index_U1])
        #b2 = output_neuron2_U1[index_U1+1] - m2*temperature_U1[index_U1+1]
        
        #T_c_experiment_x_U1 = (b2-b1)/(m1-m2)
        
        T_c_experiment_x_U1 = newtons_method( linear1_U1, linear2_U1, T_c_guess_U1 )
        T_c_experiment_y_U1 = linear1_U1(T_c_experiment_x_U1)[0]

    if interpolation == 'cubic' :
        # d (accuracy) / d (temperature_U1)
        velocity_U1 = np.zeros( np.shape( temperature_U1 ) ) 
    
        # Get the cubic spline interpolated curve and it's parameters       
        [T_mod1_U1, Output_mod1_U1, V_mod1] = cubic_spline_interpolation.ClampedCubicSpline( temperature_U1, output_neuron1_U1, velocity_U1, 250 )
        params_a1, params_b1, params_c1, params_d1 = cubic_spline_interpolation.ClampedCubicSplineCoefficients( temperature_U1, output_neuron1_U1, velocity_U1 )
        
        [T_mod2_U1, Output_mod2, V_mod2] = cubic_spline_interpolation.ClampedCubicSpline( temperature_U1, output_neuron2_U1, velocity_U1, 250 )
        params_a2, params_b2, params_c2, params_d2 = cubic_spline_interpolation.ClampedCubicSplineCoefficients( temperature_U1, output_neuron2_U1, velocity_U1 )
            
        T_c_experiment_x_U1 = newtons_method( linear1_U1, linear2_U1, T_c_guess_U1 )
        T_c_experiment_y_U1 = linear1_U1(T_c_experiment_x_U1)[0]

    print 'T_c_U1          = %.2f' % T_c_U1
    print 'T_c, experiment = %.2f' % T_c_experiment_x_U1
    print 'Percent error   = %.2g %%' % (abs(1.-T_c_experiment_x_U1/T_c_U1)*100)
    
else :
    temperature       = data_result[:,0]
    temperature_U1    = data_result[:,0][:U1_temp_len]
    output_neuron2_U1 = data_result[:,1][:U1_temp_len]
    output_neuron1_U1 = data_result[:,2][:U1_temp_len]
    accuracy_U1       = data_result[:,3][:U1_temp_len]

    temperature_U2    = data_result[:,0][U1_temp_len:]
    output_neuron2_U2 = data_result[:,1][U1_temp_len:]
    output_neuron1_U2 = data_result[:,2][U1_temp_len:]
    accuracy_U2       = data_result[:,3][U1_temp_len:]
    
    if interpolation == 'linear' :
        
        T_c_experiment_x_U1 = newtons_method( linear1_U1, linear2_U1, T_c_guess_U1 )
        T_c_experiment_y_U1 = linear1_U1(T_c_experiment_x_U1)[0]
        
        T_c_experiment_x_U2 = newtons_method( linear1_U2, linear2_U2, T_c_guess_U2 )
        T_c_experiment_y_U2 = linear1_U2(T_c_experiment_x_U2)[0]
            
    if interpolation == 'cubic' :
        # d (accuracy) / d (temperature_U1)
        velocity_U1 = np.zeros( np.shape( temperature_U1 ) ) 
    
        # Get the cubic spline interpolated curve and it's parameters       
        [T_mod1_U1, Output_mod1_U1, V_mod1] = cubic_spline_interpolation.ClampedCubicSpline( temperature_U1, output_neuron1_U1, velocity_U1, 250 )
        params_a1, params_b1, params_c1, params_d1 = cubic_spline_interpolation.ClampedCubicSplineCoefficients( temperature_U1, output_neuron1_U1, velocity_U1 )
        
        [T_mod2_U1, Output_mod2_U1, V_mod2] = cubic_spline_interpolation.ClampedCubicSpline( temperature_U1, output_neuron2_U1, velocity_U1, 250 )
        params_a2, params_b2, params_c2, params_d2 = cubic_spline_interpolation.ClampedCubicSplineCoefficients( temperature_U1, output_neuron2_U1, velocity_U1 )
            
        T_c_experiment_x_U1 = newtons_method( quadratic1_U1, quadratic2_U1, T_c_guess_U1 )
        T_c_experiment_y_U1 = quadratic2_U1(T_c_experiment_x_U1)[0]

        # d (accuracy) / d (temperature_U2)
        velocity_U2 = np.zeros( np.shape( temperature_U2 ) ) 
    
        # Get the cubic spline interpolated curve and it's parameters       
        [T_mod1_U2, Output_mod1_U2, V_mod1] = cubic_spline_interpolation.ClampedCubicSpline( temperature_U2, output_neuron1_U2, velocity_U2, 250 )
        params_a1, params_b1, params_c1, params_d1 = cubic_spline_interpolation.ClampedCubicSplineCoefficients( temperature_U2, output_neuron1_U2, velocity_U2 )
        
        [T_mod2_U2, Output_mod2_U2, V_mod2] = cubic_spline_interpolation.ClampedCubicSpline( temperature_U2, output_neuron2_U2, velocity_U2, 250 )
        params_a2, params_b2, params_c2, params_d2 = cubic_spline_interpolation.ClampedCubicSplineCoefficients( temperature_U2, output_neuron2_U2, velocity_U2 )
            
        T_c_experiment_x_U2 = newtons_method( quadratic1_U2, quadratic2_U2, T_c_guess_U2 )
        T_c_experiment_y_U2 = quadratic2_U2(T_c_experiment_x_U2)[0]

    if T_c_U1_known :
        print 'T_c (U=%d)       = %.2f' % (U1, T_c_U1)
        print 'T_c, experiment = %.2f' % T_c_experiment_x_U1
        print 'Percent error   = %.2g %%' % (abs(1.-T_c_experiment_x_U1/T_c_U1)*100)
    else :
        print 'T_c, experiment = %.2f' % T_c_experiment_x_U1        

    print 'T_c (U=%d)      = %.2f' % (U2, T_c_U2)
    print 'T_c, experiment = %.2f' % T_c_experiment_x_U2
    print 'Percent error   = %.2g %%' % (abs(1.-T_c_experiment_x_U2/T_c_U2)*100)

plt.close('all')

# Graph properties #############################################################

# Define colours in RGB space
Color    = [ [0.90, 0.25, 0.35], [0.95, 0.35, 0.00], [0.95, 0.55, 0.00],
             [0.95, 0.75, 0.00], [0.55, 0.90, 0.25], [0.40, 0.95, 0.40],
             [0.40, 0.95, 0.45], [0.40, 0.95, 0.50], [0.40, 0.95, 0.55],
             [0.20, 0.60, 0.80], [0.20, 0.60, 0.85], [0.20, 0.60, 0.90],
             [0.20, 0.60, 0.95], [0.20, 0.40, 0.95], [0.40, 0.20, 0.95],
             [0.80, 0.20, 0.95], [0.10, 0.10, 0.10], [0.60, 0.60, 0.60]                  
           ]
           
fig = plt.figure( figsize = plt.figaspect( 1.33 ) *3.0 )

ax11 = fig.add_subplot( 3, 1, 1 )

#for i in range(len(epoch_at_which_model_saved)) :
#    ax11.plot([epoch_at_which_model_saved[i],
#    epoch_at_which_model_saved[i]],[0,1], ls='-.', 
#    label = '', color=Color[2], lw=2, alpha=0.5)
#ax11.plot([],[],ls='-.', 
#  label = '$\mathrm{Epoch\ at\ which\ model\ saved}$', color=Color[2], lw=2, 
#  alpha=0.5)

ax11.plot(training_epochs, training_accuracy, ls='-', 
  label = '$\mathrm{Training\ accuracy}$', color=Color[1], lw=2, alpha=1.0)
ax11.plot(training_epochs, test_accuracy    , ls='-', 
  label = '$\mathrm{Test\ accuracy}$', color=Color[9], lw=2, alpha=1.0)

ax11.set_xlabel('$\mathrm{Training\ epoch}$', fontsize='25')
ax11.set_ylabel('$\mathrm{Accuracy}$', fontsize='25')
#plt.xlim([0.2,10])
plt.ylim([0,1])

ax12 = ax11.twinx()
ax12.plot(training_epochs, cost, ls = '--', 
  label = '$\mathrm{Cross-entropy\ cost}$', color=Color[-1], lw=2, alpha=0.5)

ax12.set_ylabel('$\mathrm{Cost}$', fontsize='25')

lines1, labels1 = ax11.get_legend_handles_labels()
lines2, labels2 = ax12.get_legend_handles_labels()

ax12.legend(lines1+lines2, labels1+labels2, loc='center right', fontsize='15')
 
ax11.grid(True)
#plt.grid(True)   

ax21 = fig.add_subplot( 3, 1, 2 )

if use_single_U :

    ax21.plot([T_c_U1, T_c_U1], [0,1], ls='--', 
    label = '$T_{c} = %.2f$' % T_c_U1, color=Color[-1], lw=2, alpha=0.5)    
            
    ax21.plot(temperature_U1, output_neuron2_U1, color=Color[1], marker='o', 
    linestyle='None', ms=5, markeredgewidth=0.0, alpha=1.0)
    ax21.plot(temperature_U1, output_neuron1_U1, color=Color[9], marker='o', 
    linestyle='None', ms=5, markeredgewidth=0.0, alpha=1.0)

    if grid == 'equal' :
        if interpolation == 'linear' :
            ax21.plot(temperature_U1, output_neuron2_U1, color=Color[1],  
            linestyle='-', lw=2, alpha=1.0)
            ax21.plot(temperature_U1, output_neuron1_U1, color=Color[9],
            linestyle='-', lw=2, alpha=1.0)
        elif interpolation == 'cubic' :
            ax21.plot(T_mod2_U1, Output_mod2_U1, 
            ls='-', label='', color=Color[1], lw=2, alpha=1.0)               
            ax21.plot(T_mod1_U1, Output_mod1_U1, 
            ls='-', label='', color=Color[9], lw=2, alpha=1.0)     
        ax21.plot(T_c_experiment_x_U1, T_c_experiment_y_U1, 
        label='$T_{c,\ \mathrm{experiment}} = %.2f$' % T_c_experiment_x_U1, color=Color[-1], 
        marker='o', linestyle='None', ms=10, markeredgewidth=0.0, alpha=0.5)
        ax21.plot([],[], 
        label='$\mathrm{Percent\ error} = %.2g %%$'%(abs(1.-T_c_experiment_x_U1/T_c_U1)*100),
        linestyle='None')
                                            
    if grid == 'log' :
        if interpolation == 'linear' :
            ax21.semilogx(temperature_U1, output_neuron2_U1, color=Color[1],  
            linestyle='-', lw=2, alpha=1.0)
            ax21.semilogx(temperature_U1, output_neuron1_U1, color=Color[9],
            linestyle='-', lw=2, alpha=1.0)
        elif interpolation == 'cubic' :
            ax21.semilogx(T_mod2_U1, Output_mod2_U1, 
            ls='-', label='', color=Color[1], lw=2, alpha=1.0)               
            ax21.semilogx(T_mod1_U1, Output_mod1_U1, 
            ls='-', label='', color=Color[9], lw=2, alpha=1.0)                
        ax21.semilogx(T_c_experiment_x_U1, T_c_experiment_y_U1, 
        label='$T_{c,\ \mathrm{experiment}} = %.2f$' % T_c_experiment_x_U1, color=Color[-1], 
        marker='o', linestyle='None', ms=10, markeredgewidth=0.0, alpha=0.5)
        ax21.semilogx([],[], 
        label='$\mathrm{Percent\ error} = %.2g %%$'%(abs(1.-T_c_experiment_x_U1/T_c_U1)*100),
        linestyle='None')
    
else :

    if T_c_U1_known :
        ax21.plot([T_c_U1, T_c_U1], [0,1], ls='--', 
        label = '$T_{c} (U=%d) = %.2f$' % (U1,T_c_U1), color=Color[-1], lw=2, alpha=0.5) 

    ax21.plot(temperature_U1, output_neuron2_U1, color=Color[1], marker='o', 
    linestyle='None', ms=5, markeredgewidth=0.0, alpha=1.0)
    ax21.plot(temperature_U1, output_neuron1_U1, color=Color[9], marker='o', 
    linestyle='None', ms=5, markeredgewidth=0.0, alpha=1.0)
    ax21.plot(temperature_U2, output_neuron2_U2, color=Color[2], marker='o', 
    linestyle='None', ms=5, markeredgewidth=0.0, alpha=1.0)
    ax21.plot(temperature_U2, output_neuron1_U2, color=Color[4], marker='o', 
    linestyle='None', ms=5, markeredgewidth=0.0, alpha=1.0)
        
    if grid == 'equal' :
        if interpolation == 'linear' :
            ax21.plot(temperature_U1, output_neuron2_U1, color=Color[1],  
            linestyle='-', lw=2, alpha=1.0)
            ax21.plot(temperature_U1, output_neuron1_U1, color=Color[9],
            linestyle='-', lw=2, alpha=1.0)
            ax21.plot(temperature_U2, output_neuron2_U2, color=Color[2],  
            linestyle='-', lw=2, alpha=1.0)
            ax21.plot(temperature_U2, output_neuron1_U2, color=Color[4],
            linestyle='-', lw=2, alpha=1.0)
            
        elif interpolation == 'cubic' :
            ax21.plot(T_mod2_U1, Output_mod2_U1, 
            ls='-', label='', color=Color[1], lw=2, alpha=1.0)               
            ax21.plot(T_mod1_U1, Output_mod1_U1, 
            ls='-', label='', color=Color[9], lw=2, alpha=1.0)     
            ax21.plot(T_mod2_U2, Output_mod2_U2, 
            ls='-', label='', color=Color[2], lw=2, alpha=1.0)               
            ax21.plot(T_mod1_U2, Output_mod1_U2, 
            ls='-', label='', color=Color[4], lw=2, alpha=1.0) 
            
        ax21.plot(T_c_experiment_x_U1, T_c_experiment_y_U1, 
        label='$T_{c,\ \mathrm{experiment}} = %.2f$' % T_c_experiment_x_U1, color=Color[-1], 
        marker='o', linestyle='None', ms=10, markeredgewidth=0.0, alpha=0.5)
        if T_c_U1_known :
            ax21.plot([],[], 
            label='$\mathrm{Percent\ error} = %.2g %%$'%(abs(1.-T_c_experiment_x_U1/T_c_U1)*100),
            linestyle='None')
        
        ax21.plot([T_c_U2, T_c_U2], [0,1], ls='--', 
        label = '$T_{c} (U=%d) = %.2f$' % (U2, T_c_U2), color=Color[-1], lw=2, alpha=0.5)
        
        ax21.plot(T_c_experiment_x_U2, T_c_experiment_y_U2, 
        label='$T_{c,\ \mathrm{experiment}} = %.2f$' % T_c_experiment_x_U2, color=Color[-1], 
        marker='o', linestyle='None', ms=10, markeredgewidth=0.0, alpha=0.5)
        ax21.plot([],[], 
        label='$\mathrm{Percent\ error} = %.2g %%$'%(abs(1.-T_c_experiment_x_U2/T_c_U2)*100),
        linestyle='None')
    
    if grid == 'log' :
        if interpolation == 'linear' :
            ax21.semilogx(temperature_U1, output_neuron2_U1, color=Color[1],  
            linestyle='-', lw=2, alpha=1.0)
            ax21.semilogx(temperature_U1, output_neuron1_U1, color=Color[9],
            linestyle='-', lw=2, alpha=1.0)
            ax21.semilogx(temperature_U2, output_neuron2_U2, color=Color[2],  
            linestyle='-', lw=2, alpha=1.0)
            ax21.semilogx(temperature_U2, output_neuron1_U2, color=Color[4],
            linestyle='-', lw=2, alpha=1.0)
            
        elif interpolation == 'cubic' :        
            ax21.semilogx(T_mod2_U1, Output_mod2_U1, 
            ls='-', label='', color=Color[1], lw=2, alpha=1.0)               
            ax21.semilogx(T_mod1_U1, Output_mod1_U1, 
            ls='-', label='', color=Color[9], lw=2, alpha=1.0)     
            ax21.semilogx(T_mod2_U2, Output_mod2_U2, 
            ls='-', label='', color=Color[2], lw=2, alpha=1.0)               
            ax21.semilogx(T_mod1_U2, Output_mod1_U2, 
            ls='-', label='', color=Color[4], lw=2, alpha=1.0) 
    
        ax21.semilogx(T_c_experiment_x_U1, T_c_experiment_y_U1, 
        label='$T_{c,\ \mathrm{experiment}} = %.2f$' % T_c_experiment_x_U1, color=Color[-1], 
        marker='o', linestyle='None', ms=10, markeredgewidth=0.0, alpha=0.5)
        if T_c_U1_known :
            ax21.semilogx([],[], 
            label='$\mathrm{Percent\ error} = %.2g %%$'%(abs(1.-T_c_experiment_x_U1/T_c_U1)*100),
            linestyle='None')
        
        ax21.semilogx([T_c_U2, T_c_U2], [0,1], ls='--', 
        label = '$T_{c} (U=%d) = %.2f$' % (U2, T_c_U2), color=Color[-1], lw=2, alpha=0.8)
        
        ax21.semilogx(T_c_experiment_x_U2, T_c_experiment_y_U2, 
        label='$T_{c,\ \mathrm{experiment}} = %.2f$' % T_c_experiment_x_U2, color=Color[-1], 
        marker='o', linestyle='None', ms=10, markeredgewidth=0.0, alpha=0.8)
        ax21.semilogx([],[], 
        label='$\mathrm{Percent\ error} = %.2g %%$'%(abs(1.-T_c_experiment_x_U2/T_c_U2)*100),
        linestyle='None')
    
    
ax21.set_xlabel('$\mathrm{Temperature}$', fontsize='25')
ax21.set_ylabel('$\mathrm{Normalized\ output}$', fontsize='25')
plt.ylim([0,1])
plt.xlim([temperature.min(), temperature.max()])

plt.legend(loc='center right', fontsize ='15')

ax21.grid(True)

ax31 = fig.add_subplot( 3, 1, 3 )

ax31.plot(temperature_U1, accuracy_U1, color=Color[1], marker='o', 
    linestyle='None', ms=5, markeredgewidth=0.0, alpha=1.0)
ax31.plot(temperature_U1, accuracy_U1, color=Color[1], 
    label = '$U=%d$' % U1, linestyle='-', lw=2, alpha=1.0)
    
if not(use_single_U) :
    ax31.plot(temperature_U2, accuracy_U2, color=Color[9], marker='o', 
        linestyle='None', ms=5, markeredgewidth=0.0, alpha=1.0)
    ax31.plot(temperature_U2, accuracy_U2, color=Color[9],  
        label = '$U=%d$' % U2, linestyle='-', lw=2, alpha=1.0)

ax31.set_xlabel('$\mathrm{Temperature}$', fontsize='25')
ax31.set_ylabel('$\mathrm{Classfication\ accuracy}$', fontsize='25')
plt.ylim([0,1])
plt.xlim([temperature.min(), temperature.max()])

plt.legend(loc='center right', fontsize ='15')

ax31.grid(True)

# Add date as footnote
plt.figtext(.05, .02, date)
                                                                                                                                    
plt.tight_layout( )              
fig.suptitle( title, fontsize ='24', y =0.99 )
plt.subplots_adjust( top=0.94 )
plt.savefig( date + '_plot.png', dpi=300)
print 'Plot saved.'
plt.show()
mng = plt.get_current_fig_manager()
mng.window.showMaximized() 