#!/usr/bin/env python
import numpy as np
import sys
from mpi4py import MPI
sys.path.append("/home/daneel/dtwa_quantum_spins/")
import dtwa_quantum_spins as dtwa
  
def run_dtwa():
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()

  #Parameters
  lattice_shapes = [(5,4)]
  alpha = 3.0
  jx, jy, jz = -0.5, -0.5, 0.0
  hx, hy, hz = 0.0, 0.0, 0.0
  niter = 20000
 
  for l in lattice_shapes:
    
    #Build the hopping matrix
    r, c = l
    latsize = r * c
    jmat = np.zeros((latsize, latsize))
    for mu in xrange(latsize):
      for nu in xrange(mu, latsize):
        if mu != nu:
	  mux, nux = np.floor(nu/c), np.floor(mu/c)
	  muy, nuy = nu%r, mu%r
	  dmn = np.sqrt((mux-nux)**2+(muy-nuy)**2)
	  jmat[mu,nu] = 1.0/pow(dmn,alpha)
    
    #Initiate the parameters in object
    p = dtwa.ParamData(hopmat=(jmat+jmat.T)/2.0,norm=1.0, latshape=l,\
			      jx=jx, jy=jy, jz=jz, hx=hx, hy=hy, hz=hz)

    p.output_magx = "sx_time_beta_"+str(alpha)+"_N_"+str(l)+"_2ndorder.txt"
    p.output_magy = "sy_time_beta_"+str(alpha)+"_N_"+str(l)+"_2ndorder.txt"
    p.output_magz = "sz_time_beta_"+str(alpha)+"_N_"+str(l)+"_2ndorder.txt"
  
    p.output_sxvar = "sxvar_time_beta_"+str(alpha)+"_N_"+str(l)+"_2ndorder.txt"
    p.output_syvar = "syvar_time_beta_"+str(alpha)+"_N_"+str(l)+"_2ndorder.txt"
    p.output_szvar = "szvar_time_beta_"+str(alpha)+"_N_"+str(l)+"_2ndorder.txt"

    p.output_sxyvar = "sxyvar_time_beta_"+str(alpha)+"_N_"+str(l)+"_2ndorder.txt"
    p.output_sxzvar = "sxzvar_time_beta_"+str(alpha)+"_N_"+str(l)+"_2ndorder.txt"
    p.output_syzvar = "syzvar_time_beta_"+str(alpha)+"_N_"+str(l)+"_2ndorder.txt"
    
    #Initiate the DTWA system with the parameters and niter
    d = dtwa.Dtwa_System(p, comm, n_t=niter, file_output=True, \
		      s_order=False, verbose=True, sitedata=False)

    #Prepare the times
    t0 = 0.0
    ncyc = 1.0
    nsteps = 200

    data = d.evolve((t0, ncyc, nsteps))

if __name__ == '__main__':
  run_dtwa()
