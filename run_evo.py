import argparse
import evaluation as evalu
import generate
import multiprocessing as mp
import random
import numpy as np
import sys, os
import subprocess as sp
import shutil
import uuid
import pickle
from pytictoc import TicToc

#Known Bug: When resuming from pickle file with a population of sparse vectors, the sparisty abruptly dips upon resuming. Unsure why this happens.

def run_evo(data_init=True,sparse=False,lam=1e-5,num_params=1024,load_file = None,maxiter=5,out_name=None,bound=1,pop_size=120,mutation_rate=0.05,recomb_rate=0.9,norm=0):
	pickle_file = './pickle/' + out_name + '.pkl'
	trial = int(out_name[-1])
	if os.path.exists(pickle_file) == False:
		Generator = generate.Generator(num_params)
		start = 0
		learning_curve = []
		if sparse:
			adj_learning_curve = []
		max_scores = []
		max_name = 'no score'
		population = []
		init_names = []
		l0_norms = []
		if data_init:
			population, init_names = Generator.load_training_objects(pop_size,trial) #we seed the initialization per trial for fair comparisions
		else:
			for i in range(0,pop_size):
				indv = []
				for j in range(num_params):
					indv.append(np.random.normal())
				population.append(indv)
	else:
		print('Loading from pickle file')
		with open(pickle_file,'rb') as f:  # Python 3: open(..., 'rb')
			if sparse:
				population,learning_curve,adj_learning_curve,max_scores,max_name,l0_norms,init_pop,Generator = pickle.load(f)
			else:
				population,learning_curve,max_scores,max_name,l0_norms,init_pop,Generator = pickle.load(f)
		start = len(learning_curve)
		print("Start: {}".format(start))

	tictoc = TicToc()
	previous_generation = []
	num_cores = mp.cpu_count()
	if(not max_scores):
		max_score = 1e6
	else:
		max_score = max_scores[-1]


	for i in range(start,maxiter):
		tictoc.tic() 
		worker_args = []
		res = []
		current_generation = []
		for j in range(0,pop_size):

			#--- MUTATION ---------------------+
			# select three random vector index positions [0, popsize), not including current vector (j)
			candidates = list(range(0,pop_size))
			candidates.remove(j)
			random_index = random.sample(candidates, 3)

			x_1 = population[random_index[0]]
			x_2 = population[random_index[1]]
			x_3 = population[random_index[2]]
			x_t = population[j]     # target individual

			# subtract x3 from x2, and create a new vector (x_diff)
			x_diff = [x_2_i - x_3_i for x_2_i, x_3_i in zip(x_2, x_3)]

			# multiply x_diff by the mutation factor (F) and add to x_1
			v_donor = [x_1_i + mutation_rate * x_diff_i for x_1_i, x_diff_i in zip(x_1, x_diff)]
			v_donor = evalu.ensure_bounds(v_donor, (-bound,bound))

			#--- RECOMBINATION ----------------+
			v_trial = []
			for k in range(len(x_t)):
				crossover = random.random()
				if crossover <= recomb_rate:
					v_trial.append(v_donor[k])
				else:
					v_trial.append(x_t[k])
			name = "G" + str(i) + "_i" + str(j)
			points = 10*Generator.generate_return_pts(v_trial) #multiply by 10 to undo AtlasNet's normalization of PCs
			worker_args.append((name,points,v_trial,sparse,lam,norm))
                
		pool = mp.Pool(num_cores-1)
		res = pool.starmap(evalu._EllipsoidEvalFunc, worker_args) #or _EvaluationFunction for boats
		current_generation = []
		for r in res:
			if r is not None:
				current_generation.append(r)
		pool.close()
		pool.join()
		
		# sort the dictionary by "output" and take the t
		current_generation.extend(previous_generation)
		current_generation = sorted(current_generation, key=lambda k: k["output"])
		curr_scores = [k["output"] for k in current_generation]
		curr_names = [k["name"] for k in current_generation]
		curr_l0 = [k["l0"] for k in current_generation]
		if(sparse):
			adj_scores = [k["adj_score"] for k in current_generation]
			adj_learning_curve.append(np.mean(adj_scores))
		l0_norms.append(np.mean(curr_l0))
		previous_generation = current_generation[:pop_size]
		population = [genome["input"] for genome in previous_generation] 
		if i == 0:
			init_pop = population

		#adjust the scores for l_0 norm penalty for learning curve plot for direct comparisons
		# if sparse:  
		# 	adj_scores = []
		# 	for j in range(len(curr_scores)):
		# 		adj_scores.append(curr_scores[j] - lam*curr_l0[j]/num_params)

		learning_curve.append(np.mean(curr_scores))
		if(curr_scores[0] < max_score):
			max_name = curr_names[0]
			max_score = curr_scores[0]
			max_scores.append(max_score)


		#dump pickle file
		with open(pickle_file,'wb') as f:
			if sparse:
				pickle.dump([population,learning_curve,adj_learning_curve,max_scores,max_name,l0_norms,init_pop,Generator],f)
			else:
				pickle.dump([population,learning_curve,max_scores,max_name,l0_norms,init_pop,Generator],f)
		
		#print stuff
		print('GENERATION: {}'.format(i))
		print('Average Score This Generation: {}'.format(np.mean(curr_scores)))
		if sparse:
			print('Average Adjusted Score This Generation: {}'.format(np.mean(adj_scores)))
		print('Average l0-Norm This Generation: {}'.format(np.mean(curr_l0)))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--num_params', type=int, default = 1024,  help='size of latent space')
	parser.add_argument('--bound', type=int, default=1, help='bound on latent space variable size')
	parser.add_argument('--sparse', type=bool, default=False, help='Whether to add l_0 norm penalty')
	parser.add_argument('--lam', type=float, default = 0.01,  help='weight of l_0 norm penalty')
	parser.add_argument('--pop_size', type=int, help='population size', default=7)
	parser.add_argument('--mutation_rate', type=float, default = 0.5,  help='mutation rate')
	parser.add_argument('--recomb_rate', type=float, default = 0.7,  help='recombination rate')
	parser.add_argument('--maxiter', type=int, default = 50,  help='maximum number of iterations')
	parser.add_argument('--load_file', type=str, default = None,  help='your path to the trained model')
	parser.add_argument('--data_init', type=bool, default=True, help='Whether to use training data as initial latent vectors (random initialization if False)')

	args = parser.parse_args()
	print(args)

	run_evo(args.data_init,args.sparse,args.lam,args.num_params,args.maxiter,args.load_file)