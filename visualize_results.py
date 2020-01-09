import argparse
import ellipsoid_eval as ee
import generate
import os
import sys
import random
sys.path.append('./AtlasNet/auxiliary/')
import ellipsoid_dataset as ed
import numpy as np
import pickle

def visualize(lam_array,num_params,trials,load_dir):
	generator = generate.Generator(num_params)
	data_key = ['DIF','DIT']
	#data_key = ['DIT']
	sparse_key = ['SF','ST']
	#sparse_key = ['SF']
	exp_names = []
	sparse_flags = []

	for s in sparse_key:
		if s == 'SF':
			pass
			for d in data_key:
				sparse_flags.append(False)
				exp_names.append(d + '_' + s + '_')
		else:	
			d = 'DIT'
			for lam in lam_array:
				sparse_flags.append(True)
				exp_names.append(d + '_' + s + '_' + str(lam) + '_')


	exp_latents = []
	for i in range(len(exp_names)):
		exp_latents.append(generator.get_latent(exp_names[i],sparse_flags[i],trials,load_dir))

	print(exp_names)
	for i in range(len(exp_latents)):
		#Plot performances of all objects
		pops = exp_latents[i]
		name = exp_names[i]
		print(name)
		for j in range(len(pops)):
			pop = pops[j]
			#pop = np.reshape(pops,(-1,num_params))
			rxs = []
			rys = []
			performances = []
			for latent in pop:
				pts = 10*generator.generate_return_pts(latent)
				radii = ee.getMinVolEllipse(P=pts)
				rx = radii[0]
				ry = radii[1]
				rxs.append(rx)
				rys.append(ry)
				performances.append(ee.performance_from_radii(rx,ry))
			ed.plot_data(rxs,rys,performances)

		#Visualize random Objects
		# latent = random.choice(pop)
		# pts = 10*generator.generate_return_pts(latent) #unnormalize from AtlasNet
		# xpts = pts[0]
		# ypts = pts[1]
		# zpts = pts[2]
		# ed.plot_ellipsoid_points(xpts,ypts,zpts)

def visualize_training(num_samples,num_params=1024):
	generator = generate.Generator(num_params)
	for i in range(5):
		latents, names = generator.load_training_objects(num_samples,i)
		#print('Loading ' + name)
		rxs = []
		rys = []
		performances = []
		for latent in latents:
			pts = 10*generator.generate_return_pts(latent) #unnormalize from AtlasNet
			#Performance
			radii = ee.getMinVolEllipse(P=pts)
			rx = radii[0]
			ry = radii[1]
			performance = ee.performance_from_radii(rx,ry)
			rxs.append(rx)
			rys.append(ry)
			performances.append(performance)
		ed.plot_data(rxs,rys,performances)
	#Plot it
	# xpts = pts[0]
	# ypts = pts[1]
	# zpts = pts[2]
	# ed.plot_ellipsoid_points(xpts,ypts,zpts)	

def visualize_pts(lam_array,num_params,trials,load_dir):
	generator = generate.Generator(num_params)
	data_key = ['DIF','DIT']
	sparse_key = ['SF','ST']
	exp_names = []
	sparse_flags = []

	for s in sparse_key:
		if s == 'SF':
			pass
			for d in data_key:
				sparse_flags.append(False)
				exp_names.append(d + '_' + s + '_')
		else:	
			d = 'DIT'
			for lam in lam_array:
				sparse_flags.append(True)
				exp_names.append(d + '_' + s + '_' + str(lam) + '_')
	exp_latents = []
	for i in range(len(exp_names)):
		exp_latents.append(generator.get_latent(exp_names[i],sparse_flags[i],trials,load_dir))

	print(exp_names)
	#for pops in exp_latents[1]:
		#Plot performances of all objects
	pops = exp_latents
	for pop in pops:
		pop = exp_latents[1]
		pop = np.reshape(pops,(-1,num_params))
		pop = pop.tolist()
		rxs = []
		rys = []
		performances = []
		random.shuffle(pop)	
		for latent in pop[0:3]:
			latent = np.asarray(latent)
			pts = 10*generator.generate_return_pts(latent)
			xpts = pts[0]
			ypts = pts[1]
			zpts = pts[2]
			ed.plot_ellipsoid_points(xpts,ypts,zpts)




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--lam_array', type=float,nargs = '+', help='lambda values to evaluate')
	parser.add_argument('--num_params', type=int, default = 1024,  help='size of latent space')
	parser.add_argument('--load_dir', type=str, default = './pickle/',  help='Folder to load latent vectors from')
	parser.add_argument('--trials', type=int, default = 3,  help='Number of repititions of each experiments in the load dir')
	args = parser.parse_args()
	print(args)

	#visualize(args.lam_array,args.num_params,args.trials,args.load_dir)
	visualize_training(120)
	#visualize_pts(args.lam_array,args.num_params,args.trials,args.load_dir)
