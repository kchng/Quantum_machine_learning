# Quantum machine learning

First, run run_this_first_and_once_for_new_data.py to label, shuffle and save data into n files.

Second, run TF_HSF_CNN0.py or TF_HSF_CNN1.py (deeper network). You can set training epochs as high as you like as the network can detect overfitting and stop training. It will also try to save the best model as the training goes on.

Finally, run TF_HSF_CNN_data_plotter.py to plot the data. You need cubic_spline_interpolation.py to be in the same directory as the plotter.
