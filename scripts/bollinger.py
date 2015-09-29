#!/usr/bin/env python
import numpy as np
import sys
import os
from mpi4py import MPI
sys.path.append("/home/daneel/dtwa_quantum_spins/")
import dtwa_quantum_spins as dtwa
  
def run_dtwa():
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()

  #Parameters
  jx, jy, jz = 0.0, 0.0, 1.0
  hx, hy, hz = 1.0, 0.0, 0.0
  niter = 2000
  
  #Import the hopping matrix
  root = 0
  filename = "mode_code_1443141813_Jij.csv"
  if rank == root:
    jmat = np.array(np.loadtxt(filename, delimiter=','))  
  else:
    jmat = None
  jmat = comm.bcast(jmat, root=root)
  
  (row, col) = jmat.shape  

  offset = os.environ['PBS_ARRAY_INDEX']
  #Initiate the parameters in object
  p = dtwa.ParamData(hopmat=(jmat+jmat.T),norm=1.0, latsize=row,\
		      jx=jx, jy=jy, jz=jz, hx=hx, hy=hy, hz=hz)

  p.output_magx = str(offset) + "sx_time_beta_" + filename + ".txt"
  p.output_magy = str(offset) + "sy_time_beta_" + filename + ".txt"
  p.output_magz = str(offset) + "sz_time_beta_" + filename + ".txt"
  
  p.output_sxvar = str(offset) + "sxvar_time_beta_"+ filename  + ".txt"
  p.output_syvar = str(offset) + "syvar_time_beta_"+ filename  + ".txt"
  p.output_szvar = str(offset) + "szvar_time_beta_"+ filename  + ".txt"

  p.output_sxyvar = str(offset) + "sxyvar_time_beta_"+ filename + ".txt"
  p.output_sxzvar = str(offset) + "sxzvar_time_beta_"+ filename + ".txt"
  p.output_syzvar = str(offset) + "syzvar_time_beta_"+ filename + ".txt"
  
  
  #Initiate the DTWA system with the parameters and niter
  d = dtwa.Dtwa_System(p, comm, n_t=niter, file_output=True, seed_offset=offset,\
      s_order=True, verbose=False, sitedata=False)

  #Prepare the times
  t0 = 0.0
  ncyc = 1.0
  nsteps = 100

  data = d.evolve((t0, ncyc, nsteps), sampling="all")

if __name__ == '__main__':
  run_dtwa()
