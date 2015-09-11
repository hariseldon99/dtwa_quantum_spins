#!/usr/bin/env python
import numpy as np
import sys
from mpi4py import MPI
sys.path.append("/home/daneel/gitrepos/dtwa_quantum_spins/")
import dtwa_quantum_spins as dtwa
  
def run_dtwa():
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()

  #Parameters
  lattice_shapes = [(4,2)]
  jx, jy, jz = 0.0, 0.0, -1.0
  hx, hy, hz = -0.2, 0.0, 0.0
  niter = 2000
 
  for l in lattice_shapes:
    #Initiate the parameters in object
    p = dtwa.ParamData(hopmat=jmat,norm=1.0, latshape=lattice_shape,\
			      jx=jx, jy=jy, jz=jz, hx=hx, hy=hy, hz=hz)

    p.output_magx = "sx_time_beta_"+str(b)+"_N_"+str(l)+"_2ndorder.txt"
    p.output_magy = "sy_time_beta_"+str(b)+"_N_"+str(l)+"_2ndorder.txt"
    p.output_magz = "sz_time_beta_"+str(b)+"_N_"+str(l)+"_2ndorder.txt"
  
    p.output_sxvar = "sxvar_time_beta_"+str(b)+"_N_"+str(l)+"_2ndorder.txt"
    p.output_syvar = "syvar_time_beta_"+str(b)+"_N_"+str(l)+"_2ndorder.txt"
    p.output_szvar = "szvar_time_beta_"+str(b)+"_N_"+str(l)+"_2ndorder.txt"

    p.output_sxyvar = "sxyvar_time_beta_"+str(b)+"_N_"+str(l)+"_2ndorder.txt"
    p.output_sxzvar = "sxzvar_time_beta_"+str(b)+"_N_"+str(l)+"_2ndorder.txt"
    p.output_syzvar = "syzvar_time_beta_"+str(b)+"_N_"+str(l)+"_2ndorder.txt"
    
    #Initiate the DTWA system with the parameters and niter
    d = dtwa.Dtwa_System(p, comm, n_t=niter, file_output=True, \
		      s_order=True, verbose=True, sitedata=False)

    #Prepare the times
    t0 = 0.0
    ncyc = 1.0
    nsteps = 100

    data = d.evolve((t0, ncyc, nsteps))

if __name__ == '__main__':
  run_dtwa()