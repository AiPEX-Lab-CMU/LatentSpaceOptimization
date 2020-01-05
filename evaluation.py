import multiprocessing as mp
import numpy as np
import sys, os
import ellipsoid_eval as ee
            
def ensure_bounds(vec, bounds):
    vec_new = []
    # cycle through each variable in vector
    for i in range(len(vec)):
        # variable exceedes the minimum boundary
        if vec[i] < bounds[0]:
            vec_new.append(bounds[0])
        # variable exceedes the maximum boundary
        if vec[i] > bounds[1]:
            vec_new.append(bounds[1])
        # the variable is fine
        if bounds[0] <= vec[i] <= bounds[1]:
            vec_new.append(vec[i])
    return vec_new

def _EllipsoidEvalFunc(name,points,input_vector,sparse=False,lam=None,norm=0):
    radii = ee.getMinVolEllipse(P=points)
    score = 1/ee.performance_from_radii(radii[0],radii[1])
    if sparse:
        adj_score = score
        score += lam*np.linalg.norm(input_vector,ord=norm)/len(input_vector)
    else:
        adj_score = None
    rval = {"input":input_vector, "output":score, "name":name,"l0":np.linalg.norm(input_vector,ord=0),"adj_score":adj_score}
    return rval
