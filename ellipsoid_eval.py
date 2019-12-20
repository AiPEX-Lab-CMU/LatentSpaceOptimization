from __future__ import division
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys
import numpy as np
from numpy import linalg
from random import random

    
def getMinVolEllipse(P, tolerance=0.01):
    """ Find the minimum volume ellipsoid which holds all the points
    
    Based on work by Nima Moshtagh
    http://www.mathworks.com/matlabcentral/fileexchange/9542
    and also by looking at:
    http://cctbx.sourceforge.net/current/python/scitbx.math.minimum_covering_ellipsoid.html
    Which is based on the first reference anyway!
    
    Here, P is a numpy array of N dimensional points like this:
    P = [[x,y,z,...], <-- one point per line
         [x,y,z,...],
         [x,y,z,...]]
    
    Returns:
    (center, radii, rotation)
    
    """
    (N, d) = np.shape(P)
    d = float(d)

    # Q will be our working array
    Q = np.vstack([np.copy(P.T), np.ones(N)]) 
    QT = Q.T
    
    # initializations
    err = 1.0 + tolerance
    u = (1.0 / N) * np.ones(N)

    # Khachiyan Algorithm
    while err > tolerance:
        V = np.dot(Q, np.dot(np.diag(u), QT))
        M = np.diag(np.dot(QT , np.dot(linalg.inv(V), Q)))    # M the diagonal vector of an NxN matrix
        j = np.argmax(M)
        maximum = M[j]
        step_size = (maximum - d - 1.0) / ((d + 1.0) * (maximum - 1.0))
        new_u = (1.0 - step_size) * u
        new_u[j] += step_size
        err = np.linalg.norm(new_u - u)
        u = new_u

    # center of the ellipse 
    center = np.dot(P.T, u)

    # the A matrix for the ellipse
    A = linalg.inv(
                   np.dot(P.T, np.dot(np.diag(u), P)) - 
                   np.array([[a * b for b in center] for a in center])
                   ) / d
                   
    # Get the values we'd like to return
    U, s, rotation = linalg.svd(A)
    radii = 1.0/np.sqrt(s)
    radii *= 10
    return radii

def performance_from_radii(rx,ry):
    performances = [[5,4,3,4,5],[4,3,2,3,4],[3,2,1,2,3],[4,3,2,3,4],[5,4,3,4,5]]
    cluster_centers = [1,3,5,7,9]
    x_cluster = cluster_centers.index(min(cluster_centers, key=lambda x:abs(x-rx)))
    y_cluster = cluster_centers.index(min(cluster_centers, key=lambda y:abs(y-ry)))
    performance = performances[x_cluster][y_cluster]
    return performance

if __name__ == '__main__':
    load_file = './AtlasNet/data/ellipsoid_points/ellipsoid_2439.pkl'
    import pickle
    with open(load_file,'rb') as f:
        points = pickle.load(f)
    radii = getMinVolEllipse(P=points)
    performance = performance_from_radii(radii[0],radii[1])