#!/usr/bin/env python
"""
This example scripts computes the hopping matrix of a 2D triangular lattice
with power law (alpha) decay in the hopping amplitude
"""
import numpy as np
from mpi4py import MPI
import dtwa_quantum_spins as dtwa
from numpy.linalg import norm
#import matplotlib.pyplot as plt
#from pprint import pprint

def run_dtwa():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nrows = 3
    alphas = [0.7, 2]
    jx, jy, jz = 0.0, 0.0, -1.0
    hx, hy, hz = 1.0, 0.0, 0.0
    niter = 20000

    #Generate the lattice
    x = 0.5
    y = np.sqrt(3.)/2.
    edgex = x * np.arange(nrows+1)
    edgey = -y * np.arange(nrows+1)
    edge = np.asarray(zip(edgex,edgey))
    points = edge
    for i in xrange(nrows):
        newedge  = np.delete(np.array([point - [x,y] for point in edge]),-1,0)
        points = np.vstack((points,newedge))
        edge = newedge
    #plt.plot(points[:,0], points[:,1], 'ro')
    #plt.show()
    #lattice size
    N = points.shape[0]
    for alpha in alphas:
        #Generate the hopping matrix
        J = np.zeros((N,N))
        for i in xrange(N):
            for j in xrange(N):
                pi = points[i]
                pj = points[j]
                d = norm(pi-pj)
                if i!=j:
                    J[i,j] = 1./pow(d,alpha)
        #pprint(J,depth=2)

        #Initiate the parameters in object
        p = dtwa.ParamData(hopmat=J,norm=1.0, latsize=N,\
                              jx=jx, jy=jy, jz=jz, hx=hx, hy=hy, hz=hz)
        #Initiate the DTWA system with the parameters and niter
        d = dtwa.Dtwa_BBGKY_Lindblad_System(p, comm, n_t=niter, \
            seed_offset = 0, decoherence=(0.5, 0.5, 0.5), verbose=True)
        #Prepare the times
        t0 = 0.0
        ncyc = 6.0
        nsteps = 10000

        data = d.evolve((t0, ncyc, nsteps), sampling="spr")
        if rank == 0:
            #Prepare the output files. One for each observable
            append_all = "_time_alpha_" + str(alpha) + "_N_"+str(N)+"_2ndorder.txt"

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

if __name__ == '__main__':
    run_dtwa()
