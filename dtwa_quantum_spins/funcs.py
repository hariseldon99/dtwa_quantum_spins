#Some general functions
import random
import numpy as np
from scipy.signal import fftconvolve
from consts import *
from classes import *

def sample(param, sampling, seed):
  """
  Different phase space sampling schemes for the initial state,
  hardcoded as a fully polarized product state
  """
  random.seed(seed)
  np.random.seed(seed)
  N = param.latsize
  sx_init = np.ones(N)
  if sampling == "spr":
    #According to Schachenmayer, the wigner function of the quantum
    #state generates the below initial conditions classically
    sy_init = 2.0 * np.random.randint(0,2, size=N) - 1.0
    sz_init = 2.0 * np.random.randint(0,2, size=N) - 1.0
    #Set initial conditions for the dynamics locally to vector 
    #s_init and store it as [s^x,s^x,s^x, .... s^y,s^y,s^y ..., 
    #s^z,s^z,s^z, ...]
    s_init_spins = np.concatenate((sx_init, sy_init, sz_init))
  elif sampling == "1-0":
    spin_choices = np.array([(1, 1,0),(1, 0,1),(1, -1,0),(1, 0,-1)])
    spins = np.array([random.choice(spin_choices) for i in xrange(N)])
    s_init_spins = spins.T.flatten()
  elif sampling == "all":
    spin_choices_spr = np.array([(1, 1,1),(1, 1,-1),(1, -1,1),(1, -1,-1)])
    spin_choices_10 = np.array([(1, 1,0),(1, 0,1),(1, -1,0),(1, 0,-1)])
    spin_choices = np.concatenate((spin_choices_10, spin_choices_spr))
    spins = np.array([random.choice(spin_choices) for i in xrange(N)])
    s_init_spins = spins.T.flatten()
  else:
    pass
  # Set initial correlations to 0.
  s_init_corrs = np.zeros(9*N*N)
  return s_init_spins, s_init_corrs
      
def bbgky_observables(t_output, s, params):
  N = params.latsize
  """
  Compute expectations <sx> and \sum_{ij}<sx_i sx_j> -<sx>^2 with
  wigner func at t_output values LOCALLY for each initcond and
  return them as an 'OutData' object. This assumes bbgky routine.
  For dtwa only, the observables are coded inline
  """
  sx_expct = np.sum(s[:, 0:N], axis=1) 
  sy_expct = np.sum(s[:, N:2*N], axis=1) 
  sz_expct = np.sum(s[:, 2*N:3*N], axis=1) 
	  
  #svec  is the tensor s^l_\mu
  #G = s[3*N:].reshape(3,3,N,N) is the tensor g^{ab}_{\mu\nu}.
  sview = s.view()
  gt = sview[:, 3*N:].reshape(s.shape[0], 3, 3, N, N)
  gt[:,:,:,range(N),range(N)] = 0.0 #Set the diagonals of g_munu to 0
  #Quantum spin variance 
  sx_var = np.sum(gt[:,0,0,:,:], axis=(-1,-2))
  sx_var += (np.sum(s[:, 0:N], axis=1)**2 \
    - np.sum(s[:, 0:N]**2, axis=1))
	  
  sy_var = np.sum(gt[:,1,1,:,:], axis=(-1,-2))
  sy_var += (np.sum(s[:, N:2*N], axis=1)**2 \
  - np.sum(s[:, N:2*N]**2, axis=1))
  
  sz_var = np.sum(gt[:,2,2,:,:], axis=(-1,-2))
  sz_var += (np.sum(s[:, 2*N:3*N], axis=1)**2 \
  - np.sum(s[:, 2*N:3*N]**2, axis=1))
  
  sxy_var = np.sum(gt[:,0,1,:,:], axis=(-1,-2))
  sxy_var += np.sum([fftconvolve(s[m, 0:N], s[m, N:2*N]) \
  for m in xrange(t_output.size)], axis=1)
  #Remove the diagonal parts
  sxy_var -= np.sum(s[:, 0:N] *  s[:, N:2*N], axis=1) 

  sxz_var = np.sum(gt[:,0,2,:,:], axis=(-1,-2))
  sxz_var += np.sum([fftconvolve(s[m, 0:N], s[m, 2*N:3*N]) \
    for m in xrange(t_output.size)], axis=1)
  #Remove the diagonal parts
  sxz_var -= np.sum(s[:, 0:N] *  s[:, 2*N:3*N], axis=1)
  
  syz_var = np.sum(gt[:,1,2,:,:], axis=(-1,-2))
  syz_var += np.sum([fftconvolve(s[m, N:2*N], s[m, 2*N:3*N]) \
    for m in xrange(t_output.size)], axis=1)
  #Remove the diagonal parts
  syz_var -= np.sum(s[:, N:2*N] *  s[:, 2*N:3*N], axis=1)
	  
  localdata = OutData(t_output, sx_expct, sy_expct,\
    sz_expct, sx_var, sy_var, sz_var, sxy_var, sxz_var, \
      syz_var, params)
  
  return localdata

def t_deriv(quantities, times):
  """
  Computes the time derivative of quantities wrt times
  """
  dt = np.gradient(times)
  return np.gradient(quantities, dt)

def weyl_hamilt(s,times,param):
  """
  Evaluates the Weyl Symbols of the Hamiltonian, H_w
  Does this at all times
  If |s^a> = (s^a_0, s^a_1 ... s^a_N), and
  H_w = -(1/2) * \sum_{nm} J_{nm} (J_x s^n_x s^m_x + J_y s^n_y s^m_y
	  + J_z s^n_z s^m_z) - h(t) * \sum_n (h_x s^n_x +h_y s^n_y
	  + h_z s^n_z)
  """
  N = param.latsize
  #s[:, 0:N] = sx , s[:, N:2*N] = sy, s[:, 2*N:3*N] = sz
  hw = param.jx * np.dot(s[:,0*N:1*N],param.jmat.dot(s[:,0*N:1*N].T))
  hw += param.jy * np.dot(s[:,1*N:2*N],param.jmat.dot(s[:,1*N:2*N].T))
  hw += param.jz * np.dot(s[:,2*N:3*N],param.jmat.dot(s[:,2*N:3*N].T))
  hw = hw /(2.0 * param.norm)
  hw += (param.hx * np.sum(s[:, 0:N]) +\
    param.hy * np.sum(s[:, N:2*N]) + param.hz * np.sum(s[:, 2*N:3*N]))
  return -hw
