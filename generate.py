from __future__ import print_function
#from show3d_balls import *
import multiprocessing as mp
import argparse
import os
import random
import numpy as np
import pandas as pd
import pywavefront as pwf
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
import scipy.io as sio
import sys
sys.path.append('./AtlasNet/auxiliary/')
import dataset
from model import *
from utils import *
from ply import *

MODEL_PATH = './AtlasNet/trained_models/modelG_24.pth'
ATLAS_MODEL_PATH = './AtlasNet/trained_models/ae_atlasnet_ellipsoid_1024.pth'
NUM_POINTS = 2500
NB_PRIMITIVES = 1
SPHERE_FILE = "sphere2562.obj"

class Generator:
	def __init__(self, num_params):  
		self.num_params = num_params
		self.d = dataset.Ellipsoid()#dataset.ShapeNet_Boats()


		self.network = AE_AtlasNet_SPHERE(num_points = NUM_POINTS,bottleneck_size=num_params, nb_primitives = NB_PRIMITIVES)
		self.network.apply(weights_init)
		self.network.load_state_dict(torch.load(ATLAS_MODEL_PATH, map_location='cpu'))
		self.network.eval()

		sphere = pwf.Wavefront(SPHERE_FILE,collect_faces=True)
		self.points_sphere = np.array(sphere.vertices)
		self.points_sphere = Variable(torch.FloatTensor(self.points_sphere).transpose(0,1)).contiguous()

	def load_training_objects(self,num_objects,seed):
		np.random.seed(seed)
		indexes = np.random.randint(low=0,high=self.d.__len__()-1,size=num_objects)
		np.random.seed()
		latent_vectors = []
		names = []
		for i in indexes:
			point_set, name = self.d.__getitem__(i)
			latent_vector = self.network.forward_encoder(point_set.float())
			latent_vector = latent_vector.squeeze().detach().numpy()
			latent_vectors.append(latent_vector)
			names.append(name)

		if(num_objects == 1):
			return latent_vector, name
		else:
			return latent_vectors, names	

	def load_specific_training_object(self,idx):
		point_set, name = self.d.__getitem__(idx)
		latent_vector = self.network.forward_encoder(point_set)
		latent_vector = latent_vector.squeeze().detach().numpy()
		return latent_vector, name

	def generate(self,params,uuids):
		params = Variable(torch.FloatTensor(params)).unsqueeze(0)
		genPoints = self.network.forward_inference_from_latent_space(params,self.points_sphere)
		b = np.zeros((np.shape(self.triangles)[0],4)) + 3
		b[:,1:] = self.triangles
		f_name_obj = "%s.obj" % (uuids)
		self.write_obj(filename=f_name_obj,points=pd.DataFrame(genPoints.cpu().data.squeeze().numpy()),faces=pd.DataFrame(b.astype(int)))

	def generate_return_pts(self,params):
		params = Variable(torch.FloatTensor(params)).unsqueeze(0)
		genPoints = self.network.forward_inference_from_latent_space(params,self.points_sphere)
		return pd.DataFrame(genPoints.cpu().data.squeeze().numpy())

	def write_obj(self,filename,points,faces):
		with open(filename, 'w+') as ofile:
			begin = "#OBJ file\n"
			ofile.write(begin)

			points = pd.DataFrame.as_matrix(points)#points.to_numpy()#
			for v in points:
				ofile.write("v {} {} {}\n".format(v[0],v[1],v[2]))
			
			faces = pd.DataFrame.as_matrix(faces)#faces.to_numpy()
			for f in faces:
				ofile.write("f {} {} {}\n".format(f[1]+1,f[2]+1,f[3]+1))

	def get_latent(self,exp_name,sparse,trials,load_dir):
		import pickle
		latent_vecs = []
		for i in range(trials):
			load_file = load_dir + exp_name + str(i) + '.pkl'
			with open(load_file,'rb') as f: 
				if sparse:
					latent_vec,_,_,_,_,_,_,_ = pickle.load(f)
				else:
					latent_vec,_,_,_,_,_,_ = pickle.load(f)
			latent_vecs.append(latent_vec)
		#latent_vecs = np.asarray(latent_vecs)
		return latent_vecs

if __name__  == '__main__':
	trials = 5
	load_file = './pickle_100_backup/'
	out_dir = './model_samples/'
	generator = Generator(num_params=NUM_PARAMS)
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
		exp_latents.append(generator.get_latent(exp_names[i],sparse_flags[i],trials,load_file))


	print(np.array(exp_latents).shape)

