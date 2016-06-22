# -*- coding: utf-8 -*-
"""
Created on Mon May  9 12:11:21 2016

@author: Joseph Corbett-Davies
"""

from __future__ import division
import gmm
import matplotlib.pyplot as plt
import numpy as np
import hgmm
import gaussian_splitting
import dynamics

plt.close('all')

n = 3
mu = np.zeros((n,1), dtype=np.float64)
P = np.eye(n)*0.005
P[1,1] = 1e-3
P[2,2] = 1e-3

print np.linalg.eigvals(P)
R = gaussian_splitting.get_rot_mat_from_vects(
               np.array([[1,0,0]]).T, np.array([[1,1,0]]).T  )
P = np.dot(R, np.dot(P, R.T))
print np.linalg.eigvals(P)
cov_v = np.array([[1e-3]], dtype=np.float64) 
           
           
'''
# UNGM parameters
n = 1
mu = np.zeros((n,1), dtype=np.float64)
P = np.array([[1]])           
#'''
           
#mixands = [gmm.Mixand(1/3, mu, P),gmm.Mixand(1/3, mu+0.1, P),gmm.Mixand(1/3, mu-0.1, P)]
mixands = [gmm.Mixand(1, mu, P)]
           
mixture = gmm.GaussianMixture(mixands)
print mixture


plt.figure()
mixture.plot()
plt.xlim(-11,11)
plt.ylim(-21,1)
plt.pause(0.1)
#'''

gaussian_splitting.precompute(N=3, cov_reduction=0.3, n=n)

for k in range(100):
    print '\n timestep:', k
    print 'N =', mixture.N   
    
    hgmm.propagate(mixture, hgmm.f, k, cov_v, e_res_max=7e-10, max_mixands=10)
    #print mixture
    mixture.plot(nsteps=200)
    plt.pause(0.01)
    #'''