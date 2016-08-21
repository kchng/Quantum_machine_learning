import numpy as np
import random
import TF_HSF_CNN0i_core

def evenize_integer( number ):
    if np.mod(number,2):
        return number-1
    else :
        return number

n_rep       = 20
min_n_layer = 8
max_n_layer = 128

initialise = TF_HSF_CNN0i_core.optimise_hyperparameters(use_single_U = False, U1 =4, U2=20)

for i in range(n_rep) :
    n_feature_map1 = evenize_integer( random.randint(min_n_layer, max_n_layer) )
    n_feature_map2 = evenize_integer( random.randint(min_n_layer, max_n_layer) )
    n_feature_map3 = evenize_integer( random.randint(min_n_layer, max_n_layer) )
    n_feature_map4 = evenize_integer( random.randint(min_n_layer, max_n_layer) )
    n_fully_connected = evenize_integer( random.randint(min_n_layer, max_n_layer) )
    initialise.insert_hyperparameters(n_feature_map1,n_feature_map2,n_feature_map3,
    n_feature_map4,n_fully_connected)
