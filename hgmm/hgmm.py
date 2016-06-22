# -*- coding: utf-8 -*-
"""
Implementation of Frank Havlak's hybrid Gaussian mixture model, from his 2013 
journal paper: Discrete and Continuous, Probabilistic Anticipation
for Autonomous Robots in Urban Environments (https://arxiv.org/pdf/1309.0766.pdf)
 
Created on Sun Apr 24 07:27:13 2016

@author: Joseph Corbett-Davies
"""

from __future__ import division
import numpy as np
import scipy.linalg as la
import gaussian_splitting
from matplotlib import pyplot as plt
from myplot.covariance_ellipse import plot_cov_ellipse



def get_sigma_points(mu, cov):
    ''' Find sigma points for a gaussian with mean mu, covariance cov 
    and a parameter lambda
    
    Returns
    --------
        pts 
            shape (n,2n+1) np array of sigma points
        weights
            shape (2n+1,) array of point weights
            '''
     
    n = cov.shape[0]  
    mu = mu.flatten()
    
    kappa = 3 - n # n + kappa = 3 is a good heuristic for gaussian dist data
    
    S_squared = (n+kappa)*cov # matrix we want the square root of
    
    S = la.cholesky(S_squared)
    
    n_pts = 1 + 2*n
    pts = np.zeros((n, n_pts))    
    weights = np.zeros(n_pts)    
    
    for j in range(n_pts):
        if j == 0:
            pts[:,j] = mu
            weights[j] = kappa/(n+kappa)
        elif j > 0 and j <= n:
            pts[:,j] = mu + S[j-1,:]
            weights[j] = 0.5/(n+kappa)
            
            pts[:,j+n] = mu - S[j-1,:]
            weights[j+n] = 0.5/(n+kappa)
            
    
    assert np.allclose(weights.sum(), 1.0)
    
    assert compare_sigma_pts(mu, cov, pts, weights)
    return pts, weights
        
def get_sigma_points_mean_cov(pts, weights):
    '''Find mean and covariance of a set of weighted sigma points pts
    
    Returns
    -------
        mean, cov
            np arrays (n,1) and (n,n) respectively
    '''

    n = pts.shape[0]
    
    n_pts = pts.shape[1]
    
    mean = np.sum(pts*weights, axis=1)[:,np.newaxis]
    cov = np.dot(weights*(pts-mean), (pts-mean).T)
    
    try: # check positive semi-definiteness
        la.cholesky(cov)
    except la.LinAlgError:
        print 'Covariance matrix is not positive semi-definite, aproximating...'
        # take 'covariance' about propagated 0th sigma point instead of new mean
        X0 = pts[:,0,np.newaxis] # first sigma point
        cov = np.dot(weights*(pts-X0), (pts-X0).T) 
        la.cholesky(cov) # Check positive semi-definiteness again (should always be)
        
    return mean, cov
    

def compare_sigma_pts(mu, cov, pts, weights):
    ''' return true if the generated pts have a mean and cov that matches the input 
    to get_sigma_points'''
    mu = mu.flatten()
    mu_check, cov_check = get_sigma_points_mean_cov(pts, weights)
    try:        #assert np.allclose(mu, mu_check.flatten()), 'mu values are fucked'
        assert np.allclose(cov, cov_check), 'cov values are fucked'
    except:
        print 'mu check:', mu_check.flatten()
        print 'mu', mu
        print cov
        print cov_check
        assert False, 'covariance is bung'
    return True
    
    
def f(x, v=np.array([[0]]), k=0):
    ''' Nonlinear discrete-time dynamics function'''
    r = 10 - k/49
    vel = np.pi*r/200 + v*np.ones_like(x[0,:])
    dt = 1
    
    x = np.reshape(x, (3,-1))
    x_new = np.zeros_like(x)

    for j in range(x_new.shape[1]):
        x_new[0,j] = x[0,j] + r * (np.sin(x[2,j] + vel[0,j]*dt/r) - np.sin(x[2,j]))
        x_new[1,j] = x[1,j] + r * (np.cos(x[2,j] + vel[0,j]*dt/r) - np.cos(x[2,j]))
        x_new[2,j] = x[2,j] + vel[0,j]*dt/r

    return x_new



def propagate(mixture, func, k, cov_v, e_res_max=0.5, max_mixands=30):
    ''' Propagate mixands in a gaussian mixture, according to function func(x,v,k), 
    where x (state - nxm) and v (noise - pxm) are matrices and func returns an n x m.
    func is a discrete time dynamics function.
    Mixand will split if linearity residual criteria e_res_max is exceeded, using a
    precomputed split solution.
    Mixture will reduce if number of mixands exceeds max_mixands.'''
    for m in mixture.mixands:
        break # get a single mixand from the set of mixands
    n_x = m.mu.shape[0] # number of states
    n_v = cov_v.shape[0] # number of noise components
    
    for m in mixture.mixands.copy():
        # Create augmented mean vector and covariance matrix
        cov = la.block_diag(m.P, cov_v)
        
        assert m.mu.shape == (n_x,1), str(m.mu.shape)
        mu = np.concatenate((m.mu, np.zeros((n_v,1))))
        
        # Find sigma points
        pts, weights = get_sigma_points(mu, cov)
        X = pts[:n_x,:]
        Y = pts[n_x:,:]
        
        # Calcs to help find splitting axis
        Xbar, Xbar_weights = get_sigma_points(m.mu, m.P) # n_x by 2*n_x + 1
        Xbar_augmented = np.concatenate( ( Xbar, np.ones((1,2*n_x+1)) )  )
        # LQ factorisation
        Q, R = np.linalg.qr(Xbar_augmented.T, mode='complete') 
        Q = Q.T
        L = R.T

        # Propagate sigma points and get new mean, cov, and sigma points
        X_new = func(X, Y, k)
        mu_x_new, cov_x_new = get_sigma_points_mean_cov(X_new, weights)
        
        Xbar_new, Xbar_new_weights = get_sigma_points(mu_x_new, cov_x_new) # sigma points relating to state
        Xhat_all_new = np.dot(Xbar_new, Q.T)
        Xhat_res_new = Xhat_all_new[:,n_x+1:] # X_hat_all partition not explained by linear model
    
        # Assign stuff
        m.mu = mu_x_new
        m.P = cov_x_new            
            
        # Calculate linearisation error residual e_res (mixand splitting criteria)
        zeromat = np.zeros((n_x,n_x+1))
        e_res = np.linalg.norm( np.concatenate((zeromat, Xhat_res_new), axis=1) ) 
        
        print 'e_res =', e_res,
        if e_res > e_res_max:
            print ' splitting....'
            # Residual associated with each sigma point
            E_new = np.dot( np.concatenate( (zeromat, Xhat_res_new), axis=1 ),
                            Q )
            error_norms = np.linalg.norm(E_new, axis=0)
            # 2nd mom of weighted sigma pts before propagation
            second_moments = np.cov(Xbar, aweights=error_norms, bias=True) 
            # first eigvector (one with largest eigval) is the optimal split axis
            w, V = np.linalg.eig(second_moments) # unsorted eigvals and eigvects
            idx = w.argsort()[::-1] # sort by largest eigval
            w = w[idx]
            V = V[:,idx]
                        
            e_split = V[:,0] # split axis
            
            ### Now split gaussian
            new_mixands = gaussian_splitting.split_gaussian(m, e_split)
            
            # Remove current mixand and add split results
            mixture.mixands.remove(m)
            mixture.mixands.update(new_mixands)
        else:
            print ''
        
    ### Now reduce number of mixands
    mixture.reduce_N(max_mixands)
        
        
if __name__ == "__main__":
    

    
    
    mu = np.zeros(3)
    cov = np.array([[3e-3, 2e-3, 0],
                    [2e-3, 3e-3, 0],
                    [0,    0,    1e-5]])
    
    mu = np.zeros(2)
    cov = np.array([[3e-3, 2e-3],
                    [2e-3, 3e-3]])
    calc_result = get_sigma_points_mean_cov(*get_sigma_points(mu, cov))
    print 'mu:', mu
    print calc_result[0].flatten()
    print 'cov:', calc_result[1]
    assert np.allclose(mu, calc_result[0].flatten())
    assert np.allclose(cov, calc_result[1])
