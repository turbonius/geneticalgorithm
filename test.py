#!/usr/bin/python3
import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import sys

def banana(x,y,a=1,b=100):
	return (a-x)**2 + b*(y-x**2)**2

def banana_par(X):
	n_pop = X.shape[0]
	out = np.zeros(n_pop)
	for i in range(n_pop):
		out[i] = banana(X[i,0],X[i,1]);
	return out

model=ga(function= banana_par,#lambda x: banana(x[0],x[1]),
         dimension=2,
         variable_type='real',
         variable_boundaries=np.array([[-10, 10],[-10, 10]]),
         function_timeout=120,
         convergence_curve=False,
         algorithm_parameters={'max_num_iteration': 50,\
                                       'population_size':100,\
                                       'mutation_probability':0.1,\
                                       'elit_ratio': 0.10,\
                                       'crossover_probability': 0.5,\
                                       'parents_portion': 0.3,\
                                       'crossover_type':'uniform',\
                                       'max_iteration_without_improv':None},
         parallel=True)


model.run()
print(model.best_variable)
#sys.stdout.write(model.best_variable)

