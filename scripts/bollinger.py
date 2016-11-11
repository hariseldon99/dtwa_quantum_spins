#!/usr/bin/env python
"""
This example scripts reads the hopping matrix from an input file
"""

import numpy as np
import os
import csv
from mpi4py import MPI
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

    #Initiate the DTWA system with the parameters and niter
    d = dtwa.Dtwa_BBGKY_System(p, comm, n_t=niter, seed_offset=offset, \
              verbose=False)

    #Prepare the times
    t0 = 0.0
    ncyc = 1.0
    nsteps = 100

    data = d.evolve((t0, ncyc, nsteps), sampling="all")
    #Prepare the output files. One for each observable
    append_all = "_time_2ndorder.txt"

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
