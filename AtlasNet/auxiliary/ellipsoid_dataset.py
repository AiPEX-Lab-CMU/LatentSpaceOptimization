import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def gen_params(num_gens,cov=0.05):
	performances = [[5,4,3,4,5],[4,3,2,3,4],[3,2,1,2,3],[4,3,2,3,4],[5,4,3,4,5]]
	x_data = []
	y_data = []
	p_data = []
	np.random.seed(1)
	for i in range(5):
		for j in range(5):
			if (i != 0 and i !=4) or (j!=0 and j!=4): #don't generate the corner
				x = (2*i)+1
				y = (2*j)+1
				for k in range(num_gens):
					d = np.random.multivariate_normal(mean=[x,y],cov=cov*np.identity(2),size=1)[0].tolist()
					x_data.append(d[0])
					y_data.append(d[1])
					p_data.append(performances[i][j])
	np.random.seed()
	return x_data, y_data, p_data

def plot_data(x_data,y_data,c_data):
	cm = matplotlib.cm.get_cmap('viridis')
	fig = plt.figure()
	sc = plt.scatter(x_data,y_data,c=c_data,cmap=cm,vmin=0,vmax=5,s=5)
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.xlim(0,10)
	plt.ylim(0,10)
	plt.colorbar(sc)
	plt.show()

def get_ellipsoid_points(x_scale,y_scale,sqrt_num_points):
	x_scale = x_scale/10	#normalized for atlasnet
	y_scale = y_scale/10	#normalized for atlasnet

	u_space = np.linspace(0, 2 * np.pi, sqrt_num_points)
	v_space = np.linspace(0, np.pi, sqrt_num_points)
	u,v = np.meshgrid(u_space,v_space)

	x = x_scale * np.cos(u) * np.cos(v)
	y = y_scale * np.cos(u) * np.sin(v)
	z = np.sin(u)

	x = x.reshape(-1, 1).squeeze()
	y = y.reshape(-1, 1).squeeze()
	z = z.reshape(-1, 1).squeeze()
	return x,y,z

def plot_ellipsoid_points(xpts,ypts,zpts):
	#centers on [2.5,2.5] in the xy plane to be friendlier to matplotlib
	#x_plot = [x + 2.5 for x in xpts]
	#y_plot = [y + 2.5 for y in ypts]

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(xpts, ypts, zpts, c='b',s=1)
	ax.set_xlim(-10,10)
	ax.set_ylim(-10,10)
	ax.set_xlabel('X radius')
	ax.set_ylabel('Y radius')
	plt.show()

def gen_dataset(num_gens,sqrt_num_points): #this argument will be squared to get the total number of points in the ellipsoid
	x_data,y_data,perf_data = gen_params(num_gens)
	point_data = []
	for i in range(len(x_data)):
		x_scale = x_data[i]
		y_scale = y_data[i]
		x,y,z = get_ellipsoid_points(x_scale,y_scale,sqrt_num_points)
		points = np.column_stack((x,y,z))
		point_data.append(points)
	return point_data,perf_data

def points_to_file(points):
	save_dir = './AtlasNet/data/ellipsoid_points'
	import pickle
	for i,p in enumerate(points):
		ellipsoid = np.asarray(p)
		with open(save_dir + '/ellipsoid_' + str(i) + '.pkl','wb') as f:
			pickle.dump(ellipsoid,f)

if __name__ == '__main__':
	import os
	import sys
	sys.path.append('../../')
	import ellipsoid_eval as ee

	x_data,y_data,perform_data = gen_params(250)
	point_data = []
	for i in range(len(perform_data)):
		#print(i)
		x_scale = x_data[i]
	 	y_scale = y_data[i]
	 	x,y,z = get_ellipsoid_points(x_scale,y_scale,50)
	 	points = np.column_stack((x,y,z))
	 	point_data.append(points)
	points_to_file(point_data)
	
# 	x_data,y_data,perform_data = gen_params(250)
# 	x_radiis = []
# 	y_radiis = []
# 	performances = []
# 	for i in range(len(perform_data)):
# 		x_radii = x_data[i]
# 		y_radii = y_data[i]
# 		#x,y,z = get_ellipsoid_points(x_scale,y_scale,50)
# 		#points = 10*np.column_stack((x,y,z))
# 		#radii = ee.getMinVolEllipse(P=points)
# 		performance = ee.performance_from_radii(x_radii,y_radii)
# 		x_radiis.append(x_radii)
# 		y_radiis.append(y_radii)
# 		performances.append(performance)
# 	plot_data(x_radiis,y_radiis,performances)
