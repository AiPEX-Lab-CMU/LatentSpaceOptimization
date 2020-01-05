from __future__ import division
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys
import numpy as np
from numpy import linalg
from random import random
from scipy.spatial.distance import directed_hausdorff

def get_diversity(pop):
    hsum = 0
    for i in range(len(pop)):
        for j in range(i,len(pop)):
            hsum += directed_hausdorff(pop[i],pop[j])
    return hsum

def H1_Testing(exp_name,load_dir):
    pass

def H2_Testing():
    pass

def H3_Testing():
    pass

def H4_Testing():

if __name__ == '__main__':
    