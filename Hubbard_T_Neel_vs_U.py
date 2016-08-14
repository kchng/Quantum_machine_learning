import matplotlib.pyplot as plt
import numpy as np

# 3D Hubbard Model
Estimate_T_Neel_data = np.loadtxt('Estimates_TNeel_vs_U.txt')

U = Estimate_T_Neel_data[:,0]
Estimate_T_Neel = Estimate_T_Neel_data[:,1]

U_learned = np.array([5,6,8,9,10])
T_Neel_learned = np.array([0.21,0.23,0.32,0.34,0.38])

title = '$\mathrm{3D\ Hubbard\ model}\ T_\mathrm{Neel}\ \mathrm{vs}\ U$'

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
           
fig = plt.figure( figsize = plt.figaspect( 0.65 ) *1.5 )

ax11 = fig.add_subplot( 1, 1, 1 )

ax11.plot(U,Estimate_T_Neel, color=Color[1], marker='o', label='$T_\mathrm{Neel} \mathrm{(Estimate)}$',
linestyle='None', ms=5, markeredgewidth=0.0, alpha=1.0)
ax11.plot(U_learned,T_Neel_learned, color=Color[9], marker='o', label='$T_\mathrm{Neel} \mathrm{(Estimate)}$',
linestyle='None', ms=5, markeredgewidth=0.0, alpha=1.0)

ax11.set_xlabel('$U\ \mathrm{(Potential\ energy)}$', fontsize='25')
ax11.set_ylabel('$T_\mathrm{Neel}$', fontsize='25')
#plt.ylim([0,1])
#plt.xlim([temperature.min(), temperature.max()])

plt.legend(loc='center right', fontsize ='15')

plt.tight_layout( )              
fig.suptitle( title, fontsize ='24', y =0.99 )
plt.subplots_adjust( top=0.94 )
#plt.savefig( date + '_plot.png', dpi=300)
#print 'Plot saved.'
plt.show()
mng = plt.get_current_fig_manager()
mng.window.showMaximized() 