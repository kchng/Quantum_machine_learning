import cubic_spline_interpolation
import matplotlib.pyplot as plt
import numpy as np
import sys

# Temperature index closest to T_c
index = 20

# Critical temperature
T_c = 0.36

# Initial guess solution of critical temperature
T_c_guess = 0.5

filename_measurements = '20160802-2202_measurements.dat'
filename_result = '20160802-2202_result.dat'
title = '$\mathrm{2\ (conv + ReLU),\ learning\ rate = 1e-3,\ \\lambda/n_{training\ data} = 0.0001}$'

def quadratic( x ):
    T = temperature[index]
    a, b, c, d = A[index], B[index], C[index], D[index]
    return a + b*(x-T) + c*(x-T)**2. + d*(x-T)**3., b + 2.*c*(x-T) + 3.*d*(x-T)**2.  

def quadratic1( x ):
    T = temperature[index]
    a, b, c, d = A1[index], B1[index], C1[index], D1[index]
    return a + b*(x-T) + c*(x-T)**2. + d*(x-T)**3., b + 2.*c*(x-T) + 3.*d*(x-T)**2.

def dx(f, g, x):
    return abs(g(x)[0]-f(x)[0])

def newtons_method(f, g, x0, e = 10e-10):
    delta = dx(f, g, x0)
    while delta > e:
        x0 = x0 - (f(x0)[0] - g(x0)[0])/(f(x0)[1] - g(x0)[1])
        delta = dx(f, g, x0)
    return x0

date = filename_measurements.rsplit('_',1)[0]

data_measurements = np.loadtxt(filename_measurements)

training_epochs   = data_measurements[:,0]
training_accuracy = data_measurements[:,1]
test_accuracy     = data_measurements[:,2]
cost              = data_measurements[:,3]

data_result = np.loadtxt(filename_result)

temperature       = data_result[:,0]
output_neuron2    = data_result[:,1]
output_neuron1    = data_result[:,2]
accuracy          = data_result[:,3]

# d (accuracy) / d (temperature)
velocity = np.zeros( np.shape( temperature ) ) 

[T_mod2, Output_mod2, V_mod2] = cubic_spline_interpolation.ClampedCubicSpline( temperature, output_neuron2, velocity, 250 )
A, B, C, D = cubic_spline_interpolation.ClampedCubicSplineCoefficients( temperature, output_neuron2, velocity )
[T_mod1, Output_mod1, V_mod1] = cubic_spline_interpolation.ClampedCubicSpline( temperature, output_neuron1, velocity, 250 )
A1, B1, C1, D1 = cubic_spline_interpolation.ClampedCubicSplineCoefficients( temperature, output_neuron1, velocity )

T_c_experiment = newtons_method( quadratic, quadratic1, T_c_guess )
T_c_experimenty = quadratic(T_c_experiment)[0]

print 'T_c, experiment = %.2f' % T_c_experiment

plt.close('all')

# Graph properties #############################################################

# Define colours in RGB space
Color    = [ [0.90, 0.25, 0.35], [0.95, 0.35, 0.00], [0.95, 0.55, 0.00],
             [0.95, 0.75, 0.00], [0.55, 0.90, 0.25], [0.40, 0.95, 0.45], 
             [0.20, 0.60, 0.90], [0.20, 0.40, 0.95], [0.40, 0.20, 0.95],
             [0.80, 0.20, 0.95], [0.10, 0.10, 0.10]                  
           ]
           
fig = plt.figure( figsize = plt.figaspect( 1.0 ) *3.0 )

ax11 = fig.add_subplot( 2, 1, 1 )

ax11.plot(training_epochs, training_accuracy, ls='-', 
  label = '$\mathrm{Training\ accuracy}$', color=Color[1], lw=2, alpha=1.0)
ax11.plot(training_epochs, test_accuracy    , ls='-', 
  label = '$\mathrm{Test\ accuracy}$', color=Color[6], lw=2, alpha=1.0)

ax11.set_xlabel('$\mathrm{Training\ epoch}$', fontsize='25')
ax11.set_ylabel('$\mathrm{Accuracy}$', fontsize='25')
#plt.xlim([0.2,10])
plt.ylim([0,1])

ax12 = ax11.twinx()
ax12.plot(training_epochs, cost, ls = '--', 
  label = '$\mathrm{Cross-entropy\ cost}$', color=Color[-1], lw=2, alpha=0.5)

lines1, labels1 = ax11.get_legend_handles_labels()
lines2, labels2 = ax12.get_legend_handles_labels()
ax12.legend(lines1+lines2, labels1+labels2, loc='center right', fontsize='25')
 
ax11.grid(True)
#plt.grid(True)        
                                                                                                                                              
ax21 = fig.add_subplot( 2, 1, 2 )

ax21.plot([T_c, T_c], [0,1], ls='--', 
  label = '$T_{c} = %.2f$' % T_c, color=Color[-1], lw=2, alpha=0.5)                  
ax21.semilogx(temperature, output_neuron2, color=Color[1], marker='o', 
  linestyle='None', ms=5, markeredgewidth=0.0, alpha=1.0)
ax21.semilogx(temperature, output_neuron1, color=Color[6], marker='o', 
  linestyle='None', ms=5, markeredgewidth=0.0, alpha=1.0)               
ax21.semilogx(T_mod2, Output_mod2, 
  ls='-', label='', color=Color[1], lw=2, alpha=1.0)               
ax21.semilogx(T_mod1, Output_mod1, 
  ls='-', label='', color=Color[6], lw=2, alpha=1.0)
ax21.plot(T_c_experiment, T_c_experimenty, 
  label='$T_{c,\ \mathrm{experiment}} = %.2f$' % T_c_experiment, color=Color[-1], 
  marker='o', linestyle='None', ms=10, markeredgewidth=0.0, alpha=0.5)
ax21.plot([],[], 
  label='$\mathrm{Percent\ error} = %.2g%%$'%(abs(1.-T_c_experiment/T_c)*100),
  linestyle='None')

ax21.set_xlabel('$\mathrm{Temperature}$', fontsize='25')
ax21.set_ylabel('$\mathrm{Accuracy}$', fontsize='25')
plt.ylim([0,1])
plt.xlim([temperature[0], temperature[-1]])

plt.legend(loc='center right', fontsize ='25')

ax21.grid(True)

# Add date as footnote
plt.figtext(.05, .02, date)
                                                                                                                                    
plt.tight_layout( )              
fig.suptitle( title, fontsize ='24', y =0.99 )
plt.subplots_adjust( top=0.95 )
#plt.show()
#mng = plt.get_current_fig_manager()
#mng.window.showMaximized() 
plt.savefig( date + '_plot.png', dpi=300)