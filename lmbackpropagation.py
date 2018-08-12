import numpy as np
from neupy import algorithms, plots

x_train = np.array([[1, 2], [3, 4]])
y_train = np.array([[1], [0]])

lmnet = algorithms.LevenbergMarquardt((2, 3, 1))
lmnet.train(x_train, y_train)

plots.error_plot(lmnet)
