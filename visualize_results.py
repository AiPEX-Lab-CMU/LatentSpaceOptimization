import argparse
import ellipsoid_eval as ee
import generate
import os
import sys
import random
sys.path.append('./AtlasNet/auxiliary/')
import ellipsoid_dataset as ed


NUM_PARAMS = 1024

def visualize(lam_array,num_params,trials,load_dir):
	out_dir = './model_samples/'
	generator = generate.Generator(num_params=NUM_PARAMS)
	lam_array = [0.1,1.0]
	data_key = ['DIF','DIT']
	sparse_key = ['ST','SF']
	exp_names = []
	sparse_flags = []
	for d in data_key:
		for s in sparse_key:
			if s == 'ST':
				for lam in lam_array:
				   sparse_flags.append(True)
				   exp_names.append(d + '_' + s + '_' + str(lam) + '_')
			else:
				sparse_flags.append(False)
				exp_names.append(d + '_' + s + '_')

	exp_latents = []
	for i in range(len(exp_names)):
		exp_latents.append(generator.get_latent(exp_names[i],sparse_flags[i],trials,load_dir))

	for pops in exp_latents:
		#Plot performances of all objects
		for pop in pops:
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
			latent = random.choice(pop)
			pts = 10*generator.generate_return_pts(latent) #unnormalize from AtlasNet
			xpts = pts[0]
			ypts = pts[1]
			zpts = pts[2]
			ed.plot_ellipsoid_points(xpts,ypts,zpts)

def visualize_training():
	generator = generate.Generator(num_params=NUM_PARAMS)
	latent, name = generator.load_training_object()
	print('Loading ' + name)
	pts = 10*generator.generate_return_pts(latent) #unnormalize from AtlasNet

	#Performance
	radii = ee.getMinVolEllipse(P=pts)
	rx = radii[0]
	ry = radii[1]
	performance = ee.performance_from_radii(rx,ry)
	print([rx,ry,performance])

	#Plot it
	xpts = pts[0]
	ypts = pts[1]
	zpts = pts[2]
	ed.plot_ellipsoid_points(xpts,ypts,zpts)	

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--lam_array', type=float,nargs = '+', help='lambda values to evaluate')
	parser.add_argument('--num_params', type=int, default = 1024,  help='size of latent space')
	parser.add_argument('--load_dir', type=str, default = './pickle_12-20-19/',  help='Folder to load latent vectors from')
	parser.add_argument('--trials', type=int, default = 1,  help='Number of repititions of each experiments in the load dir')
	args = parser.parse_args()
	print(args)

	#visualize(args.lam_array,args.num_params,args.trials,args.load_dir)
	visualize_training()