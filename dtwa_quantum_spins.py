#!/usr/bin/env python

"""
    Discrete Truncated Wigner Approximation (dTWA) for quantum spins
    and transverse fields with time-periodic drive

    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    * Copyright (c) 2015 Analabha Roy (daneel@utexas.edu)
    *
    *This is free software: you can redistribute it and/or modify it under
    *the terms of version 2 of the GNU Lesser General Public License
    *as published by the Free Software Foundation.
    *Notes:
    *1. The initial state is currently hard coded to be the classical ground
    *    state
    *2. Primary references are
    *   Anatoli: Ann. Phys 325 (2010) 1790-1852
    *   Mauritz: arXiv:1209.3697
    *   Schachenmayer: arXiv:1408.4441
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
"""

from __future__ import division, print_function
from mpi4py import MPI
from reductions import Intracomm
from redirect_stdout import stdout_redirected

import sys
import copy
import random
import numpy as np
from scipy.signal import fftconvolve
from scipy.sparse import *

from scipy.integrate import odeint

from pprint import pprint
from tabulate import tabulate
   
threshold = 1e-4
root = 0
#This is the kronecker delta symbol for vector indices
deltaij = np.eye(3)

#This is the Levi-Civita symbol for vector indices
eijk = np.zeros((3, 3, 3))
eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1

def t_deriv(quantities, times):
  """
  Computes the time derivative of quantities wrt times
  """
  dt = np.gradient(times)
  return np.gradient(quantities, dt)

def drive(t, params):
    """
    Returns the time-dependent drive h(t)
    """
    return params.h0 * np.cos(params.omega * t)

def initconds(sampling, size, seed=0, bbgky=True):
  """
  Initiates a single sampled trajectory based on sampling scheme and seed
  """
  random.seed(seed)
  if sampling == "spr": #According to Schachenmayer
    sy_init = np.array([2.0 * random.randint(0,2) - 1.0 \
      for i in xrange(size)])
    sz_init = np.array([2.0 * random.randint(0,2) - 1.0 \
      for i in xrange(size)])
    #Set initial conditions for the dynamics locally to vector 
    #s_init and store it as [s^x,s^x,s^x, .... s^y,s^y,s^y ..., 
    #s^z,s^z,s^z, ...]
    s_init_spins = np.concatenate((np.ones(size), sy_init, sz_init))
  elif sampling == "1-0": #According to PRM
    spin_choices = np.array([(1, 1,0),(1, 0,1),(1, -1,0),(1, 0,-1)])
    spins = np.array([random.choice(spin_choices) for i in xrange(size)])
    s_init_spins = spins.T.flatten()
  elif sampling == "all": #Logical union of the above two sampling schemes
    spin_choices_spr = np.array([(1, 1,1),(1, 1,-1),(1, -1,1),(1, -1,-1)])
    spin_choices_10 = np.array([(1, 1,0),(1, 0,1),(1, -1,0),(1, 0,-1)])
    spin_choices = np.concatenate((spin_choices_10, spin_choices_spr))
    spins = np.array([random.choice(spin_choices) for i in xrange(size)])
    s_init_spins = spins.T.flatten()
  else:
    pass
  # Set initial correlations to 0 if bbgky is wanted
  s_init_corrs = np.zeros(9*size*size) if bbgky else None
  return s_init_spins, s_init_corrs


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
  drvs = drive(times, param)
  hw = param.jx * np.dot(s[:,0*N:1*N],param.jmat.dot(s[:,0*N:1*N].T))
  hw += param.jy * np.dot(s[:,1*N:2*N],param.jmat.dot(s[:,1*N:2*N].T))
  hw += param.jz * np.dot(s[:,2*N:3*N],param.jmat.dot(s[:,2*N:3*N].T))
  hw = hw /(2.0 * param.norm)
  hw += drvs * (param.hx * np.sum(s[:, 0:N]) +\
    param.hy * np.sum(s[:, N:2*N]) + param.hz * np.sum(s[:, 2*N:3*N]))
  return -hw

def func_dtwa(s, t, param):
    """
    The RHS of general case, per Schachemmayer eq A2
    """
    N = param.latsize
    #s[0:N] = sx , s[N:2*N] = sy, s[2*N:3*N] = sz
    drv = drive(t, param)
    jsx = 2.0 * param.jx * param.jmat.dot(s[0:N])/param.norm
    jsx += 2.0 * drv * param.hx
    jsy = 2.0 * param.jy * param.jmat.dot(s[N:2*N])/param.norm
    jsy += 2.0 * drv * param.hy
    jsz = 2.0 * param.jz * param.jmat.dot(s[2*N:3*N])/param.norm
    jsz += 2.0 * drv * param.hz
    dsxdt = s[N:2*N] * jsz - s[2*N:3*N] * jsy
    dsydt = s[2*N:3*N] * jsx - s[0:N] * jsz
    dszdt = s[0:N] * jsy - s[N:2*N] * jsx
    return np.concatenate((dsxdt, dsydt, dszdt))

def jac_dtwa(s, t, param):
    """
    Jacobian of the general case. First order.
    This is given by 9 NXN submatrices:
    J00=J11=J22=0
    Although Jacobian is NOT antisymmetric in general! See below
    J01 = +J_z diag(J|s^x>) + h(t) h_z - J_y (J#|s^z>)
    J10 = -J_z diag(J|s^x>) - h(t) h_z + J_x (J#|s^z>)
    J02 = -J_y diag(J|s^y>) - h(t) h_y + J_z (J#|s^y>)
    J20 = +J_y diag(J|s^y>) + h(t) h_y - J_x (J#|s^y>)
    J12 = +J_x diag(J|s^x>) + h(t) h_x - J_z (J#|s^x>)
    J21 = -J_x diag(J|s^x>) - h(t) h_x + J_y (J#|s^x>)
    Here, '#' (hash operator) means multiply each row of a matrix by the
    corresponding vector element. This is implemented by numpy.multiply()
    """
    N = param.latsize
    #s[0:N] = sx , s[N:2*N] = sy, s[2*N:3*N] = sz
    full_jacobian = np.zeros(shape=(3*N, 3*N))
    drivemat = 2.0 * drive(t, param) * np.eye(N)
    diag_jsx = np.diagflat((param.jmat.dot(s[0:N])))/param.norm
    diag_jsy = np.diagflat((param.jmat.dot(s[N:2*N])))/param.norm
    #diag_jsz = np.diagflat((param.jmat.dot(s[2*N:3*N])))/param.norm
    hash_jsx = (np.multiply(param.jmat.T, s[0:N]).T)/param.norm
    hash_jsy = (np.multiply(param.jmat.T, s[N:2*N]).T)/param.norm
    hash_jsz = (np.multiply(param.jmat.T, s[2*N:3*N]).T)/param.norm
    full_jacobian[0:N, N:2*N] = param.jz * diag_jsx + drivemat * param.hz\
      -param.jy * hash_jsz
    full_jacobian[N:2*N, 0:N] = -param.jz * diag_jsx - \
      drivemat * param.hz + param.jx * hash_jsz
    full_jacobian[0:N, 2*N:3*N] = -param.jy * diag_jsy - drivemat * \
      param.hy + param.jz * hash_jsy
    full_jacobian[2*N:3*N, 0:N] = param.jy * diag_jsy + drivemat * \
      param.hy - param.jx * hash_jsy
    full_jacobian[N:2*N, 2*N:3*N] = param.jx * diag_jsx + drivemat * \
      param.hx - param.jz * hash_jsx
    full_jacobian[2*N:3*N, N:2*N] = -param.jx * diag_jsx - drivemat * \
      param.hx + param.jy * hash_jsx
    return full_jacobian

def func_dtwa_bbgky(s, t, param):
    """
    The RHS of general case, second order correction, per Lorenzo
    "J" is the J_{ij} hopping matrix
    -\partial_t |s^x> = -first order + 2 (J^y  Jg^{yz} - J^z  Jg^{zy})
								  /norm,
    -\partial_t |s^y> = -first order + 2 (-J^z  Jg^{zx} + J^x  Jg^{xz})
								  /norm,
    -\partial_t |s^z> = -first order + 2 (-J^x  Jg^{xy} + J^y  Jg^{yx})
								  /norm.
    """
    N = param.latsize
    #svec  is the tensor s^l_\mu
    #G = s[3*N:].reshape(3,3,N,N) is the tensor g^{ab}_{\mu\nu}.
    sview = s.view()
    stensor = sview[0:3*N].reshape(3, N)
    gtensor = sview[3*N:].reshape(3, 3, N, N)
    gtensor[:,:,range(N),range(N)] = 0.0 #Set the diagonals of g_munu to 0
      
    htensor = np.zeros_like(stensor)
    htensor[0].fill(param.hvec[0])
    htensor[1].fill(param.hvec[1])
    htensor[2].fill(param.hvec[2])
    Gtensor = np.einsum("mg,abgn->abmn", param.jmat, gtensor)/param.norm
    Mtensor = np.einsum("am,b,mn->abmn", stensor, param.jvec, \
      param.jmat)/param.norm
    hvec_dressed = htensor + np.einsum("llgm->lm", Mtensor)
    dtensor = gtensor + np.einsum("am,bn", stensor, stensor)
    dsdt_1 = func_dtwa(sview[0:3*N], t, param).reshape(3, N)
    dsdt = dsdt_1 - \
      2.0 * np.einsum("bcmm,b,abc->am", Gtensor, param.jvec, eijk)

    dgdt = -np.einsum("lbmn,abl->abmn", Mtensor, eijk) + \
      np.einsum("lanm,abl->abmn", Mtensor, eijk)
    
    dgdt -= np.einsum("lm,kbmn,lka->abmn", hvec_dressed, gtensor, eijk) -\
      np.einsum("llnm,kbmn,lka->abmn", Mtensor, gtensor, eijk) +\
	np.einsum("ln,akmn,lkb->abmn", hvec_dressed, gtensor, eijk) -\
	  np.einsum("llmn,akmn,lkb->abmn", Mtensor, gtensor, eijk)
      
    dgdt -= np.einsum("l,km,lbmn,lka->abmn", \
      param.jvec, stensor, Gtensor, eijk) + \
	np.einsum("l,kn,lanm,lkb->abmn", param.jvec, stensor, \
	  Gtensor, eijk)  
 
    dgdt += np.einsum("almn,lkmn,lkb->abmn", Mtensor, dtensor, eijk)\
      + np.einsum("blnm,lknm,lka->abmn", Mtensor, dtensor, eijk)
 
    #Flatten it before returning
    return np.concatenate((dsdt.flatten(), 2.0 * dgdt.flatten()))
 
def jac_dtwa_bbgky(s, t, param):
  """
  Jacobian of the general case. Second order.
  """
  N = param.latsize
  fullsize_2ndorder = 3 * N + 9 * N**2
  #svec  is the tensor s^l_\mu
  #G = s[3*N:].reshape(3,3,N,N) is the tensor g^{ab}_{\mu\nu}.
  sview = s.view()
  stensor = sview[0:3*N].reshape(3, N)
  gtensor = sview[3*N:].reshape(3, 3, N, N)
  htensor = np.zeros_like(stensor)
  htensor[0].fill(param.hvec[0])
  htensor[1].fill(param.hvec[1])
  htensor[2].fill(param.hvec[2])
  jjtensor = np.einsum("a,mn->amn", param.jvec, param.jmat)
  sstensor = np.einsum("km,ln->klmn",stensor,stensor)
  Mtensor = np.einsum("am,b,mn->abmn", stensor, param.jvec, \
	param.jmat)/param.norm
  hvec_dressed = htensor + np.einsum("llgm->lm", Mtensor)


  full_jacobian = np.zeros(shape=(fullsize_2ndorder, fullsize_2ndorder))
  
  #J00 subblock
  full_jacobian[0:3*N, 0:3*N] = jac_dtwa(s, t, param)
  
  #J01 subblock. Precalculated
  full_jacobian[0:3*N, 3*N:] = param.dsdotdg  
  
  #J10 subblock
  full_jacobian[3*N:, 0:3*N] =  -(np.einsum("pml,kbmn,pka->abpmnl", \
    jjtensor,gtensor, eijk) +  np.einsum("pnl,akmn,pkb->abpmnl", \
      jjtensor, gtensor, eijk)).reshape(9*N*N,3*N)
  full_jacobian[3*N:, 0:3*N] -= (np.einsum("qmg,ml,bqng,qpa->abpmnl",\
    jjtensor, param.deltamn,gtensor, eijk) + \
      np.einsum("qng,nl,aqmg,qpb->abpmnl",jjtensor, param.deltamn, \
	gtensor, eijk) ).reshape(9*N*N,3*N)
  full_jacobian[3*N:, 0:3*N] += (np.einsum("qmn,ml,bqnn,qpa->abpmnl",\
    jjtensor, param.deltamn,gtensor, eijk) + \
      np.einsum("qnm,nl,aqmm,qpb->abpmnl", jjtensor,param.deltamn, \
	gtensor, eijk)).reshape(9*N*N,3*N)
  full_jacobian[3*N:, 0:3*N] += (np.einsum("qmn,ml,pa,qkmn,qkb->abpmnl",\
    jjtensor,param.deltamn,deltaij,gtensor+sstensor,eijk) + \
      np.einsum("qmn,nl,pb,kqmn,qka->abpmnl", jjtensor,param.deltamn, \
	deltaij,gtensor+sstensor,eijk)).reshape(9*N*N,3*N)
  full_jacobian[3*N:, 0:3*N] += (np.einsum("pmn,ml,akmn,pkb->abpmnl",\
    jjtensor,param.deltamn, sstensor, eijk) + \
      np.einsum("pmn,nl,bknm,pka->abpmnl", jjtensor,param.deltamn, \
	sstensor, eijk) + np.einsum("kmn,nl,akmm,kpb->abpmnl",\
	  jjtensor,param.deltamn, sstensor, eijk) + \
	    np.einsum("kmn,ml,bknn,kpa->abpmnl", jjtensor,param.deltamn, \
	      sstensor, eijk)).reshape(9*N*N,3*N)
  full_jacobian[3*N:, 0:3*N] = 2.0 * \
    (full_jacobian[3*N:, 0:3*N]/param.norm)
  full_jacobian[3*N:, 0:3*N] += param.dsdotdg.T
  
  #J11 subblock: 
  full_jacobian[3*N:, 3*N:] = -(np.einsum("qm,mlnhbpqra->abrpmnlh",\
     hvec_dressed, param.delta_eps_tensor)).reshape(9*N*N,9*N*N)
  full_jacobian[3*N:, 3*N:] += (np.einsum("qqmn,mlnhbpqra->abrpmnlh", \
	Mtensor, param.delta_eps_tensor)).reshape(9*N*N,9*N*N)
  full_jacobian[3*N:, 3*N:] -= (np.einsum("qn,mlnharqpb->abrpmnlh",\
    hvec_dressed, param.delta_eps_tensor)).reshape(9*N*N,9*N*N)
  full_jacobian[3*N:, 3*N:] += (np.einsum("qqnm,mlnharqpb->abrpmnlh",\
	Mtensor, param.delta_eps_tensor)).reshape(9*N*N,9*N*N)
  
  excl_tensor  = -np.einsum("qmh,km,nl,br,pka->abrpmnlh",\
	jjtensor,stensor, param.deltamn, deltaij,eijk)
  excl_tensor += -np.einsum("qnh,kn,ml,ar,pkb->abrpmnlh",\
	jjtensor,stensor, param.deltamn, deltaij,eijk)
  excl_tensor += -np.einsum("qml,km,nh,bp,rka->abrpmnlh",\
	jjtensor,stensor, param.deltamn, deltaij,eijk)
  excl_tensor += -np.einsum("qnl,kn,mh,ap,rkb->abrpmnlh",\
	jjtensor,stensor, param.deltamn, deltaij,eijk)
  #Set the \eta=\mu,\nu components of excl_tensor to 0
  excl_tensor[:,:,:,:,range(N),:,:,range(N)] = 0.0
  excl_tensor[:,:,:,:,:,range(N),:,range(N)] = 0.0
  full_jacobian[3*N:, 3*N:] += excl_tensor.reshape(9*N*N,9*N*N)
  
  full_jacobian[3*N:, 3*N:] += (np.einsum("rmn,am,ml,nh,rpb->abrpmnlh",\
    jjtensor,stensor,param.deltamn,param.deltamn,eijk) + \
      np.einsum("rmn,bn,mh,nl,rpa->abrpmnlh",\
	jjtensor,stensor,param.deltamn,param.deltamn,\
	  eijk)).reshape(9*N*N,9*N*N)
  full_jacobian[3*N:, 3*N:] -= (np.einsum("pmn,am,mh,nl,prb->abrpmnlh",\
    jjtensor,stensor,param.deltamn,param.deltamn,eijk) + \
      np.einsum("pmn,bn,ml,nh,pra->abrpmnlh",\
	jjtensor,stensor,param.deltamn,param.deltamn,\
	  eijk)).reshape(9*N*N,9*N*N)    
  full_jacobian[3*N:, 3*N:] = 2.0 * (full_jacobian[3*N:, 3*N:]/param.norm)
  
  return full_jacobian

class ParamData:
    """Class that stores Hamiltonian and lattice parameters 
       to be used in each dTWA instance. This class has no 
       methods other than the constructor.
    """
    
    def __init__(self, hopmat = None, norm=1.0, latsize=11, \
			  h0=1.0, omega=0.0, hx=0.0, hy=0.0, hz=0.0,\
			    jx=0.0, jy=0.0, jz=1.0):
      
      """
       Usage:
       p = ParamData(hopmat = None, norm=1.0, latsize=100, h0=1.0,\ 
		      omega=0.0, hx=0.0, hy=0.0, hz=0.0,\ 
			jx=0.0, jy=0.0, jz=1.0)
       
       All parameters (arguments) are optional.
       
       Parameters:
       hopmat 	=  The hopping matrix J for the Ising part of the 
		   Hamiltonian, i.e. J_{ij} \sigma^{xyz}_i \sigma^{xyz}_j
		   Example: In the 1-dimensional ising model with nearest 
		   neighbour hopping (open boundary conditions), J can be 
		   obtained via numpy by:
		   
		     import numpy as np
		     J = np.diagflat(np.ones(10),k=1) +\ 
			    np.diagflat(np.ones(10),k=-1)
		   
		   The diagonal is expected to be 0.0. There are no 
		   internal checks for this. When set to 'None', the module
		   defaults to the hopmat of the 1D ising model with long 
		   range coulomb hopping and open boundary conditions.
		   
       norm 	=  This quantity sinply scales the hopmat in case you 
	           need it		   
       latsize  =  The size of your lattice as an integer. This can be in 
		   any dimensions
       h0	=  The drive amplitude. This is the amplitude of a periodic
		   cosine drive on the transverse field (if any). Defaults to 
		   unity
       omega	=  The frequency of the abovementioned drive. Defaults to 0.
       h(x,y,z) =  The values of the uniform transverse fields i.e, the terms
		   that scale \sigma^{xyz} respectively in the Hamiltonian 
		   Defaults to 0.
       j(x,y,z) =  The values of the bare hopping i.e, the terms that scale
		   each \sigma^{xyz}_i\sigma^{xyz}_j. 
		   Defaults to (0,0,1). Set all to unity if so desired.
      
       Additional parameters:
       output_mag(x,y,z)  = Output file(s) for the (x,y,z) magnetization. 
			   Internal defaults
       output_s(x,y,z)var = Output file(s) for the (x,y,z) fluctuations in
			   magnetization. Internal defaults
       output_s(a)var	  = Output file(s) for the (a) correlations summed 
			   over all sites. Internal defaults. here, a can 
			   be (xy,xz,yz)
			   
       Return value: 
       An object that stores all the parameters above. 
      """
      
      #Default Output file names. Each file dumps a different observable
      self.output_magx = "sx_outfile.txt"
      self.output_magy = "sy_outfile.txt"
      self.output_magz = "sz_outfile.txt"
    
      self.output_sxvar = "sxvar_outfile.txt"
      self.output_syvar = "syvar_outfile.txt"
      self.output_szvar = "szvar_outfile.txt"

      self.output_sxyvar = "sxyvar_outfile.txt"
      self.output_sxzvar = "sxzvar_outfile.txt"
      self.output_syzvar = "syzvar_outfile.txt"
    
      #Whether to normalize with Kac norm or not
      self.norm = norm
      
      self.latsize = latsize
      
      self.h0 = h0 # Drive amplitude
      self.omega = omega #Drive frequency
      self.hx = hx #x transverse field
      self.hy = hy #y transverse field
      self.hz = hz #z transverse field
      self.jx = jx #x hopping
      self.jy = jy #y hopping
      self.jz = jz #z hopping
    
      self.jvec = np.array([jx, jy, jz])
      self.hvec = np.array([hx, hy, hz])
      N = self.latsize
      self.fullsize_2ndorder = 3 * N + 9 * N**2
      self.deltamn = np.eye(N)
      # These are the lattice  sites for two point density matrix calc.
      self.tpnt_sites = (np.floor(N/2), np.floor(N/2)+2) 
      if(hopmat == None): #Use the default hopping matrix
	self.periodic_boundary_conditions = True
	self.open_boundary_conditions = False
	#This is the dense Jmn hopping matrix with inverse 
	#power law decay for periodic boundary conditions.
	J = dia_matrix((N, N))
	mid_diag = np.floor(N/2).astype(int)
	for i in xrange(1,mid_diag+1):
	  elem = pow(i, -1.0)
	  J.setdiag(elem, k=i)
	  J.setdiag(elem, k=-i)
	for i in xrange(mid_diag+1, N):
	  elem = pow(N-i, -1.0)
	  J.setdiag(elem, k=i)
	  J.setdiag(elem, k=-i)
	  self.jmat = J.toarray()
      else: #Take the provided hopping matrix
	  self.jmat = hopmat  

class OutData:
    """Class to store output data in a dictionary"""
    def __init__(self, t, sx, sy, sz, sxx, syy, szz, sxy, sxz, syz,\
      params):
        self.t_output = t
        self.sx, self.sy, self.sz = sx, sy, sz
        self.sxvar, self.syvar, self.szvar = sxx, syy, szz
        self.sxyvar, self.sxzvar, self.syzvar = sxy, sxz, syz
        self.__dict__.update(params.__dict__)

    def normalize_data(self, w_totals, lsize):
        self.sx = self.sx/(w_totals * lsize)
        self.sy = self.sy/(w_totals * lsize)
        self.sz = self.sz/(w_totals * lsize)
        self.sxvar = (1/lsize) + (self.sxvar/(w_totals * lsize * lsize))
        self.sxvar = self.sxvar - (self.sx)**2
        self.syvar = (1/lsize) + (self.syvar/(w_totals * lsize * lsize))
        self.syvar = self.syvar - (self.sy)**2
        self.szvar = (1/lsize) + (self.szvar/(w_totals * lsize * lsize))
        self.szvar = self.szvar - (self.sz)**2
        self.sxyvar = (self.sxyvar/(w_totals * lsize * lsize))
        self.sxyvar = self.sxyvar - (self.sx * self.sy)
        self.sxzvar = (self.sxzvar/(w_totals * lsize * lsize))
        self.sxzvar = self.sxzvar - (self.sx * self.sz)
        self.syzvar = (self.syzvar/(w_totals * lsize * lsize))
        self.syzvar = self.syzvar - (self.sy * self.sz)

    def dump_data(self):
        np.savetxt(self.output_magx, \
	  np.vstack((self.t_output, self.sx)).T, delimiter=' ')
        np.savetxt(self.output_magy, \
	  np.vstack((self.t_output, self.sy)).T, delimiter=' ')
        np.savetxt(self.output_magz, \
	  np.vstack((self.t_output, self.sz)).T, delimiter=' ')
        np.savetxt(self.output_sxvar, \
          np.vstack((self.t_output, self.sxvar)).T, delimiter=' ')
        np.savetxt(self.output_syvar, \
          np.vstack((self.t_output, self.syvar)).T, delimiter=' ')
        np.savetxt(self.output_szvar, \
          np.vstack((self.t_output, self.szvar)).T, delimiter=' ')
        np.savetxt(self.output_sxyvar, \
          np.vstack((self.t_output, self.sxyvar)).T, delimiter=' ')
        np.savetxt(self.output_sxzvar, \
          np.vstack((self.t_output, self.sxzvar)).T, delimiter=' ')
        np.savetxt(self.output_syzvar, \
          np.vstack((self.t_output, self.syzvar)).T, delimiter=' ')

class Dtwa_System:
  """
    Class that creates the dTWA system.
    
       Introduction:  
	This class instantiates an object encapsulating the dTWA problem.
	It has all MPI_Gather routines for aggregating observable data
	from the different random samples of initial conditions (which 
	are run in parallel), and has methods that sample the trajectories
	and execute the dTWA methods (1st order and 2nd order i.e. with 
	BBGKY). These methods call integrators from scipy and time-evolve 
	all the randomly sampled initial conditions.
  """

  def __init__(self, params, mpicomm, n_t=2000, file_output=True,\
			  seed_offset=0,  bbgky=False, jac=False,\
			    verbose=True):
    """
    Initiates an instance of the Dtwa_System class. Copies parameters
    over from an instance of ParamData and stores precalculated objects .
    
       Usage:
       d = Dtwa_System(Paramdata, MPI_COMMUNICATOR, n_t=2000,\ 
			file_output=True, bbgky=False, jac=False,\ 
					verbose=True)
       
       Parameters:
       Paramdata 	= An instance of the class "ParamData". 
			  See the relevant docs
       MPI_COMMUNICATOR = The MPI communicator that distributes the samples
			  to parallel processes. Set to MPI_COMM_SELF if 
			  running serially
       n_t		= Number of initial conditions to sample randomly 
			  from the discreet spin phase space. Defaults to
			  2000.
       file_output      = Boolean for file output. Set to False if you 
			  don't want data dumped to text files.			    
       seed_offset      = Offset in the seed. The initial conditions are 
			  sampled randomly by each processor using the 
			  random generator in python with unique seeds for
			  each processor. Each processor adds seeed_offset 
			  to its seed. This allows you to ensure that 
			  separate dTWA objects have uniquely random initial 
			  states by changing seed_offset.
			  Defaults to 0.
       bbgky          = Boolean for choosing the second order method,
			  i.e. with BBGKY corrections on regular dTWA.
			  Defaults to False, leading to the first order
			  i.e. regular dTWA.
       jac		= Boolean for choosing to evaluate the jacobian
			  during the integration of the sampled initial
			  conditions. If this is set to "True", then stiff
			  regions of the trajectories allow the integrator 
			  to compute the jacobian from the analytical 
			  formula. Defaults to False. 
			  WARNING: Keep this boolean 'False' if the 
			  lattice size is very large, as the jacobian size
			  scales as size^2 X size^2, and can cause buffer 
			  overflows.
       verbose		= Boolean for choosing verbose outputs. Setting 
			  to 'True' dumps verbose output to stdout, which
			  consists of full output from the integrator, as
			  well as the output of the time derivative
			  of the Weyl symbol of the Hamiltonian that you
			  have provided via the 'hopmat' and other input
			  in ParamData. Defaults to 'False'.			  
			  
      Return value: 
      An object that stores all the parameters above. If bbgky
      and 'jac' are set to 'True', then this object includes 
      precalculated data for those parts of the second order 
      jacobian that are time-independent. This is named 'dsdotdg'.
    """
    
    self.jac = jac
    self.__dict__.update(params.__dict__)
    self.n_t = n_t
    self.file_output = file_output
    self.comm=mpicomm
    self.seed_offset = seed_offset
    self.bbgky = bbgky
    #Booleans for verbosity and for calculating site data
    self.verbose = verbose
    N = params.latsize
    
    #Only computes these if you want 2nd order
    if self.bbgky and self.jac:
      #Below are the constant subblocks of the 2nd order Jacobian
      #The 00 subblock is the first order Jacobian in func below
      #The entire 01 subblock, fully time independent (ds_dot/dg):
      self.dsdotdg = -np.einsum("p,mh,ml,apr->arpmlh",\
	self.jvec, self.jmat, self.deltamn, eijk)
      self.dsdotdg += np.einsum("r,ml,mh,arp->arpmlh", \
	self.jvec,self.jmat, self.deltamn, eijk)
      self.dsdotdg = 2.0 * (self.dsdotdg/self.norm)
      self.dsdotdg = self.dsdotdg.reshape(3*N, 9*N**2)
      self.delta_eps_tensor = np.einsum("ml,nh,ar,qpb->mlnharqpb",\
	self.deltamn,self.deltamn,deltaij,eijk)
      self.delta_eps_tensor += np.einsum("mh,nl,ap,qrb->mhnlapqrb",\
	self.deltamn,self.deltamn,deltaij,eijk)
      #The time independent part of the 10 subblock (dg_dot/ds):
      #is the SAME as ds_dot/dg	    

  def sum_reduce_all_data(self, datalist_loc,t, mpcomm):
      """
      Does the parallel sum reduction of all data
      """
      #Do local sums
      sx_locsum = np.sum(data.sx for data in datalist_loc)
      sy_locsum = np.sum(data.sy for data in datalist_loc)
      sz_locsum = np.sum(data.sz for data in datalist_loc)
      sxvar_locsum = np.sum(data.sxvar for data in datalist_loc)
      syvar_locsum = np.sum(data.syvar for data in datalist_loc)
      szvar_locsum = np.sum(data.szvar for data in datalist_loc)
      sxyvar_locsum = np.sum(data.sxyvar for data in datalist_loc)
      sxzvar_locsum = np.sum(data.sxzvar for data in datalist_loc)
      syzvar_locsum = np.sum(data.syzvar for data in datalist_loc)
      
      #Only root processor will actually get the data
      sx_totals = np.zeros_like(sx_locsum) if mpcomm.rank == root\
	else None
      sy_totals = np.zeros_like(sy_locsum) if mpcomm.rank == root\
	else None
      sz_totals = np.zeros_like(sz_locsum) if mpcomm.rank == root\
	else None
      sxvar_totals = np.zeros_like(sxvar_locsum) if mpcomm.rank == root\
	else None
      syvar_totals = np.zeros_like(syvar_locsum) if mpcomm.rank == root\
	else None
      szvar_totals = np.zeros_like(szvar_locsum) if mpcomm.rank == root\
	else None
      sxyvar_totals = np.zeros_like(sxyvar_locsum) if mpcomm.rank == root\
	else None
      sxzvar_totals = np.zeros_like(sxzvar_locsum) if mpcomm.rank == root\
	else None
      syzvar_totals = np.zeros_like(syzvar_locsum) if mpcomm.rank == root\
	else None
      
      #To prevent conflicts with other comms
      duplicate_comm = Intracomm(mpcomm)
      sx_totals = duplicate_comm.reduce(sx_locsum, root=root)
      sy_totals = duplicate_comm.reduce(sy_locsum, root=root)
      sz_totals = duplicate_comm.reduce(sz_locsum, root=root)
      sxvar_totals = duplicate_comm.reduce(sxvar_locsum, root=root)
      syvar_totals = duplicate_comm.reduce(syvar_locsum, root=root)
      szvar_totals = duplicate_comm.reduce(szvar_locsum, root=root)
      sxyvar_totals = duplicate_comm.reduce(sxyvar_locsum, root=root)
      sxzvar_totals = duplicate_comm.reduce(sxzvar_locsum, root=root)
      syzvar_totals = duplicate_comm.reduce(syzvar_locsum, root=root)

      if mpcomm.rank == root:
	return OutData(t, sx_totals, sy_totals, sz_totals, sxvar_totals, \
	    syvar_totals, szvar_totals, sxyvar_totals, sxzvar_totals,\
	      syzvar_totals, self)
      else:
	return None
      
  def dtwa_only(self, time_info, sampling):
      comm = self.comm
      N = self.latsize
      rank = comm.rank
      
      if rank == root and self.verbose:
	  pprint("# Run parameters:")
	  pprint(vars(self), depth=2)
      if rank == root and not self.verbose:
	  pprint("# Starting run ...")
      if type(time_info) is tuple:
	(t_init, n_cycles, n_steps) = time_info
	if self.omega == 0:
	    t_final = t_init + n_cycles
	else:
	    t_final = t_init + (n_cycles * (2.0* np.pi/self.omega))
	dt = (t_final-t_init)/(n_steps-1.0)
	t_output = np.arange(t_init, t_final, dt)
      elif type(time_info) is list or np.ndarray:
	t_output = time_info
      else:
	print("Please enter either a tuple or a list for the time interval") 
	exit(0)
      #Let each process get its chunk of n_t by round robin
      nt_loc = 0
      iterator = rank
      while iterator < self.n_t:
	  nt_loc += 1
	  iterator += comm.size
      #Scatter unique seeds for generating unique random number arrays :
      #each processor gets its own nt_loc seeds, and allocates nt_loc 
      #initial conditions. Each i.c. is a 2N sized array
      #now, each process sends its value of nt_loc to root
      all_ntlocs = comm.gather(nt_loc, root=root)
      #Let the root process initialize nt unique integers for random seeds
      if rank == root:
	  all_seeds = np.arange(self.n_t, dtype=np.int64)+1
	  all_ntlocs = np.array(all_ntlocs)
	  all_displacements = np.roll(np.cumsum(all_ntlocs), root+1)
	  all_displacements[root] = 0 # First displacement
      else:
	  all_seeds = None
	  all_displacements = None
      local_seeds = np.zeros(nt_loc, dtype=np.int64)
      #Root scatters nt_loc sized seed data to that particular process
      comm.Scatterv([all_seeds, all_ntlocs, all_displacements,\
	MPI.DOUBLE], local_seeds)

      list_of_local_data = []
      
      for runcount in xrange(0, nt_loc, 1):
	  s_init = initconds(sampling, N, \
	    local_seeds[runcount] + self.seed_offset, self.bbgky)
	  if self.verbose:
	    if self.jac:
	      s, info = odeint(func_dtwa, s_init, t_output,\
		args=(self,), Dfun=jac_dtwa, full_output=True)
	    else:
	      s, info = odeint(func_dtwa, s_init, t_output,\
		args=(self,), Dfun=None, full_output=True)
	  else:
	    if self.jac:
	      s = odeint(func_dtwa, s_init, t_output, args=(self,),\
		Dfun=jac_dtwa)
	    else:
	      s = odeint(func_dtwa, s_init, t_output, args=(self,),\
		Dfun=None)
	  #Compute expectations <sx> and \sum_{ij}<sx_i sx_j> -<sx>^2 with
	  #wigner func at t_output values LOCALLY for each initcond and
	  #store them
	  sx_expectations = np.sum(s[:, 0:N], axis=1) 
	  sy_expectations = np.sum(s[:, N:2*N], axis=1) 
	  sz_expectations = np.sum(s[:, 2*N:3*N], axis=1) 
	  
	  #Quantum spin variance maps to the classical expression
	  # (1/N) + (1/N^2)\sum_{i\neq j} S^x_i S^x_j - <S^x>^2 and
	  # (1/N) + (1/N^2)\sum_{i\neq j} S^y_i S^z_j
	  # since the i=j terms quantum average to unity
	  sx_var =   (np.sum(s[:, 0:N], axis=1)**2 \
	    - np.sum(s[:, 0:N]**2, axis=1))
	  sy_var =   (np.sum(s[:, N:2*N], axis=1)**2 \
	  - np.sum(s[:, N:2*N]**2, axis=1))
	  sz_var =   (np.sum(s[:, 2*N:3*N], axis=1)**2 \
	  - np.sum(s[:, 2*N:3*N]**2, axis=1))

	  sxy_var =   np.sum([fftconvolve(s[m, 0:N], \
	    s[m, N:2*N]) for m in xrange(t_output.size)], axis=1)
	  sxz_var =   np.sum([fftconvolve(s[m, 0:N], \
	    s[m, 2*N:3*N]) for m in xrange(t_output.size)], axis=1)
	  syz_var =   np.sum([fftconvolve(s[m, N:2*N], \
	    s[m, 2*N:3*N]) for m in xrange(t_output.size)], axis=1)

	  localdata = OutData(t_output, sx_expectations, sy_expectations,\
	    sz_expectations, sx_var, sy_var, sz_var, sxy_var, sxz_var, \
	      syz_var, self)
	  list_of_local_data.append(localdata)
      #After loop above  sum reduce (don't forget to average) all locally
      #calculated expectations at each time to root
      outdat = \
	self.sum_reduce_all_data(list_of_local_data, t_output, comm)    
	  
      if rank == root:
	  #Dump to file
	  outdat.normalize_data(self.n_t, N)
	  if self.file_output:
	    outdat.dump_data()
	  if self.verbose:
	      print("  ")
	      print("Integration output info:")
	      pprint(info)
	      print("""# Cumulative number of Jacobian evaluations
			  by root:""", \
		np.sum(info['nje']))
	  print('# Done!')
	  return outdat
      else:
	return None

  def dtwa_bbgky(self, time_info, sampling):
      comm = self.comm
      old_settings = np.seterr(all='ignore') #Prevent overflow warnings
      N = self.latsize
      rank = comm.rank
      if rank == root and self.verbose:
	  pprint("# Run parameters:")
          #Copy params to another object, then delete
          #the output that you don't want printed
	  out = copy.copy(self)
	  out.dsdotdg = 0.0
	  out.delta_eps_tensor = 0.0
	  out.jmat = 0.0
          out.deltamn = 0.0	
	  pprint(vars(out), depth=2)
      if rank == root and not self.verbose:
	  pprint("# Starting run ...")
      if type(time_info) is tuple:
	(t_init, n_cycles, n_steps) = time_info
	if self.omega == 0:
	    t_final = t_init + n_cycles
	else:
	    t_final = t_init + (n_cycles * (2.0* np.pi/self.omega))
	dt = (t_final-t_init)/(n_steps-1.0)
	t_output = np.arange(t_init, t_final, dt)
      elif type(time_info) is list  or np.ndarray:
	t_output = time_info
      else:
	print("Please enter either a tuple or a list for the time interval") 
	exit(0)
      #Let each process get its chunk of n_t by round robin
      nt_loc = 0
      iterator = rank
      while iterator < self.n_t:
	  nt_loc += 1
	  iterator += comm.size
      #Scatter unique seeds for generating unique random number arrays :
      #each processor gets its own nt_loc seeds, and allocates nt_loc 
      #initial conditions. Each i.c. is a 2N sized array
      #now, each process sends its value of nt_loc to root
      all_ntlocs = comm.gather(nt_loc, root=root)
      #Let the root process initialize nt unique integers for random seeds
      if rank == root:
	  all_seeds = np.arange(self.n_t, dtype=np.int64)+1
	  all_ntlocs = np.array(all_ntlocs)
	  all_displacements = np.roll(np.cumsum(all_ntlocs), root+1)
	  all_displacements[root] = 0 # First displacement
      else:
	  all_seeds = None
	  all_displacements = None
      local_seeds = np.zeros(nt_loc, dtype=np.int64)
      #Root scatters nt_loc sized seed data to that particular process
      comm.Scatterv([all_seeds, all_ntlocs, all_displacements,\
	MPI.DOUBLE],local_seeds)

      list_of_local_data = []
      
      if self.verbose:
	list_of_dhwdt_abs2 = []

      for runcount in xrange(0, nt_loc, 1):
	  s_init_spins, s_init_corrs = initconds(sampling, N, \
	    local_seeds[runcount] + self.seed_offset, self.bbgky)
	  
	  #Redirect unwanted stdout warning messages to /dev/null
	  with stdout_redirected():
	    if self.verbose:
	      if self.jac:
		s, info = odeint(func_dtwa_bbgky, \
		  np.concatenate((s_init_spins, s_init_corrs)), t_output, \
		    args=(self,), Dfun=jac_dtwa_bbgky, full_output=True)
	      else:
		s, info = odeint(func_dtwa_bbgky, \
		  np.concatenate((s_init_spins, s_init_corrs)),t_output, \
		    args=(self,), Dfun=None, full_output=True)
	    else:
	      if self.jac:
		s = odeint(func_dtwa_bbgky, \
		  np.concatenate((s_init_spins, s_init_corrs)), \
		    t_output, args=(self,), Dfun=jac_dtwa_bbgky)
	      else:
		s = odeint(func_dtwa_bbgky, \
		  np.concatenate((s_init_spins, s_init_corrs)), t_output, \
		    args=(self,), Dfun=None)
	  
	  #Computes |dH/dt|^2 for a particular alphavec & weighes it 
	  #If the rms over alphavec of these are 0, then each H is const
	  if self.verbose:
	    hws = weyl_hamilt(s,t_output, self)
	    dhwdt = np.array([t_deriv(hw, t_output) for hw in hws])
	    dhwdt_abs2 = np.square(dhwdt) 
	    list_of_dhwdt_abs2.extend(dhwdt_abs2)
	  
	  s = np.array(s, dtype="float128")#Widen memory to reduce overflows
	  #Compute expectations <sx> and \sum_{ij}<sx_i sx_j> -<sx>^2 with
	  #wigner func at t_output values LOCALLY for each initcond and
	  #store them
	  sx_expectations = np.sum(s[:, 0:N], axis=1) 
	  sy_expectations = np.sum(s[:, N:2*N], axis=1) 
	  sz_expectations = np.sum(s[:, 2*N:3*N], axis=1) 
		  
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
	  	  
	  localdata = OutData(t_output, sx_expectations, sy_expectations,\
	    sz_expectations, sx_var, sy_var, sz_var, sxy_var, sxz_var, \
	      syz_var, self)
	  list_of_local_data.append(localdata)
      #After loop above  sum reduce (don't forget to average) all locally
      #calculated expectations at each time to root
      outdat = \
	self.sum_reduce_all_data(list_of_local_data, t_output, comm) 
      if self.verbose:
	dhwdt_abs2_locsum = np.sum(list_of_dhwdt_abs2, axis=0)
	dhwdt_abs2_totals = np.zeros_like(dhwdt_abs2_locsum)\
	  if rank == root else None
	temp_comm = Intracomm(comm)
	dhwdt_abs2_totals = temp_comm.reduce(dhwdt_abs2_locsum, root=root)
	if rank == root:
	  dhwdt_abs2_totals = dhwdt_abs2_totals/(self.n_t * N * N)
	  dhwdt_abs_totals = np.sqrt(dhwdt_abs2_totals)
	
      #Dump to file
      if rank == root:
	  outdat.normalize_data(self.n_t, N)
	  if self.file_output:
	    outdat.dump_data()
	  if self.verbose:
	    print("t-deriv of Hamilt (abs square) with wigner avg: ")
	    print("  ")
	    print(tabulate({"time": t_output, \
	      "dhwdt_abs": dhwdt_abs_totals}, \
	      headers="keys", floatfmt=".6f"))
	  if self.jac and self.verbose:
	    print('# Cumulative number of Jacobian evaluations by root:', \
	      np.sum(info['nje']))
	  print('# Done!')
	  np.seterr(**old_settings)  # reset to default
	  return outdat
      else:
	np.seterr(**old_settings)  # reset to default
	return None
  
  def evolve(self, time_info, sampling="spr"):
    """
    This function calls the lsode 'odeint' integrator from scipy package
    to evolve all the randomly sampled initial conditions in time. 
    Depending on how the Dtwa_System class is instantiated, the function
    chooses either the first order (i.e. purely classical dynamics)
    or second order (i.e. classical + correlations via BBGKY corrections)
    dTWA method(s). The lsode integrator controls integrator method and 
    actual time steps adaptively. Verbosiy levels are decided during the
    instantiation of this class. After the integration is complete, each 
    process computes site observables for each trajectory, and used
    MPI_Reduce to aggregate the sum to root. The root then returns the 
    data as a dictionary, and, optionally (decided during instantiation),
    dumps it all to files. An optional argument is the sampling scheme.
    
    
       Usage:
       data = d.evolve(times, sampling="spr")
       
       Required parameters:
       times 		= Time information. There are 2 options: 
			  1. A 3-tuple (t0, t1, steps), where t0(1) is the 
			      initial (final) time, and steps are the number
			      of time steps that are in the output. 
			  2. A list or numpy array with the times entered
			      manually.
			      
			      Note that the integrator method and the actual step sizes
			      are controlled internally by the integrator. 
			      See the relevant docs for scipy.integrate.odeint.
			  
       Optional parameters:
       sampling		= The sampling scheme used. The choices are 
			  1. "spr" : The prescription obtained from the
				     phase point operators used by
				     Schachenmayer et. al. in their paper.
				     This is the default.
			  2."1-0"  : <DESCRIBE>
			  3."all"  : The prescription obtained from the 
				     logical union of both the phase point
				     operators above.

      Return value: 
      A dictionary that contains the times, and all observables.
      This contains: 
	1. The times, 
	2. The single site observables (x,y and z), and 
	3. All correlation sums (xx, yy, zz, xy, xz and yz).
    """
    
    if self.bbgky:
      return self.dtwa_bbgky(time_info, sampling)
    else:
      return self.dtwa_only(time_info, sampling)
      
if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    #Initiate the parameters in object
    p = ParamData(latsize=101, beta=1.0)
    #Initiate the DTWA system with the parameters and niter
    d = Dtwa_System(p, comm, n_t=2000)
    data = d.evolve((0.0, 1.0, 1000))