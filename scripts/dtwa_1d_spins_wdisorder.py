#!/usr/bin/env python
"""
This example scripts computes the hopping matrix of a 2D lattice with
power law (alpha) decay in the hopping amplitude
"""
import numpy as np
import csv
from mpi4py import MPI
import sys
import dtwa_quantum_spins as dtwa
from random import seed, choice

def run_dtwa():
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()

  #Parameters
  lattice_size = 11
  alphas = [0.7, 2.0]
  jx, jy, jz = 0.0, 0.0, -1.0
  hx, hy, hz = 1.0, 0.0, 0.0
  niter = 5000
  choices = [-1,1]
  s = 9 #Seed
 
  for alpha in alphas:
    seed(s)
    #Build the hopping matrix
    size = lattice_size
    jmat = np.zeros((size, size))
    for mu in xrange(size):
      for nu in xrange(mu, size):
        if mu != nu:
            dmn = np.abs(mu-nu)
            jmat[mu,nu] = choice(choices)/pow(dmn,alpha)
    
    #Initiate the parameters in object
    p = dtwa.ParamData(hopmat=(jmat+jmat.T),norm=1.0, latsize=size,\
			      jx=jx, jy=jy, jz=jz, hx=hx, hy=hy, hz=hz)

    #Initiate the DTWA system with the parameters and niter
    d = dtwa.Dtwa_BBGKY_System(p, comm, n_t=niter, seed_offset = 0, verbose=True)

    #Prepare the times
    t0 = 0.0
    ncyc = 1.3
    nsteps = 200

    data = d.evolve((t0, ncyc, nsteps), sampling="spr")
    
    if rank == 0:
      #Prepare the output files. One for each observable
      append_all = "_time_alpha_" + str(alpha) + "_N_"+str(l)+"_2ndorder.txt"
      
      outfile_magx = "sx" + append_all
      outfile_magy = "sy" + append_all
      outfile_magz = "sz" + append_all
    
      outfile_sxvar = "sxvar" + append_all
      outfile_syvar = "syvar" + append_all
      outfile_szvar = "szvar" + append_all

      outfile_sxyvar = "sxyvar" + append_all
      outfile_sxzvar = "sxzvar" + append_all
      outfile_syzvar = "syzvar" + append_all
      
      #Dump each observable to a separate file
      np.savetxt(outfile_magx, \
	np.vstack((data.t_output, data.sx)).T, delimiter=' ')
      np.savetxt(outfile_magy, \
	np.vstack((data.t_output, data.sy)).T, delimiter=' ')
      np.savetxt(outfile_magz, \
	np.vstack((data.t_output, data.sz)).T, delimiter=' ')
      np.savetxt(outfile_sxvar, \
	np.vstack((data.t_output, data.sxvar)).T, delimiter=' ')
      np.savetxt(outfile_syvar, \
	np.vstack((data.t_output, data.syvar)).T, delimiter=' ')
      np.savetxt(outfile_szvar, \
	np.vstack((data.t_output, data.szvar)).T, delimiter=' ')
      np.savetxt(outfile_sxyvar, \
	np.vstack((data.t_output, data.sxyvar)).T, delimiter=' ')
      np.savetxt(outfile_sxzvar, \
	np.vstack((data.t_output, data.sxzvar)).T, delimiter=' ')
      np.savetxt(outfile_syzvar, \
	np.vstack((data.t_output, data.syzvar)).T, delimiter=' ')
      
      #Alternatively, convert output to discionary and dump to single file
      ##Either as a csv file
      w = csv.writer(open("output.csv", "w"))
      for key, val in vars(data).items():
	w.writerow([key, val])
    ##Or any other way you want :)

if __name__ == '__main__':
  run_dtwa()
