#Class Library
from mpi4py import MPI
from reductions import Intracomm
import numpy as np
from itertools import starmap
import operator as op
from consts import *
from scipy.sparse import dia_matrix
class ParamData:
    """Class that stores Hamiltonian and lattice parameters
       to be used in each dTWA instance. This class has no
       methods other than the constructor.
    """

    def __init__(self, hopmat = None, norm=1.0, latsize=11, \
                          hx=0.0, hy=0.0, hz=0.0,\
                            jx=0.0, jy=0.0, jz=1.0):

        """
         Usage:
         p = ParamData(hopmat = None, norm=1.0, latsize=100, \
                        hx=0.0, hy=0.0, hz=0.0,\
                          jx=0.0, jy=0.0, jz=1.0)

         All parameters (arguments) are optional.

         Parameters:
         hopmat   =  The hopping matrix J for the Ising part of the
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

         norm     =  This quantity sinply scales the hopmat in case you
                     need it
         latsize  =  The size of your lattice as an integer. This can be in
                     any dimensions
         h(x,y,z) =  The values of the uniform transverse fields i.e, the terms
                     that scale \sigma^{xyz} respectively in the Hamiltonian
                     Defaults to 0.
         j(x,y,z) =  The values of the bare hopping i.e, the terms that scale
                     each \sigma^{xyz}_i\sigma^{xyz}_j.
                     Defaults to (0,0,1). Set all to unity if so desired.

         Return value:
         An object that stores all the parameters above.
        """

        self.norm = norm
        self.latsize = latsize
        self.hx, self.hy, self.hz  = hx , hy, hz#x transverse field
        self.jx, self.jy, self.jz = jx, jy, jz #hopping
        self.jvec, self.hvec = np.array([jx, jy, jz]), np.array([hx, hy, hz])
        N = self.latsize
        self.fullsize_2ndorder = 3 * N + 9 * N**2
        self.deltamn = np.eye(N)
        if(hopmat == None): #Use the default hopping matrix
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
    """Class to store output data in this object"""
    def __init__(self, t, sx, sy, sz, sxx, syy, szz, sxy, sxz, syz,\
      params):
        self.t_output = t
        self.sx, self.sy, self.sz = sx, sy, sz
        self.sxvar, self.syvar, self.szvar = sxx, syy, szz
        self.sxyvar, self.sxzvar, self.syzvar = sxy, sxz, syz
        self.__dict__.update(params.__dict__)

    def normalize_data(self, w_totals, lsize):
        n, m, t = w_totals * lsize, w_totals * lsize * lsize, (1/lsize)
        (self.sx, self.sy, self.sz, self.sxvar, self.syvar, self.szvar, \
          self.sxyvar, self.sxzvar, self.syzvar) = \
            starmap(op.itruediv, zip((self.sx, self.sy, self.sz,\
              self.sxvar, self.syvar, self.szvar,\
                self.sxyvar, self.sxzvar, self.syzvar), \
                  (n, n, n, m, m, m, m, m, m)))
        (self.sxvar, self.syvar, self.szvar) = starmap(op.iadd,\
          zip((self.sxvar, self.syvar, self.szvar),(t,t,t)))
        (self.sxvar, self.syvar, self.szvar, \
          self.sxyvar, self.sxzvar, self.syzvar) = starmap(op.isub,\
          zip((self.sxvar, self.syvar, self.szvar, \
            self.sxyvar, self.sxzvar, self.syzvar),\
            ((self.sx)**2,(self.sy)**2,(self.sz)**2,\
              (self.sx * self.sy),(self.sx * self.sz),(self.sy * self.sz))))


def sum_reduce_all_data(param, datalist_loc,t, mpcomm):
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
              syzvar_totals, param)
    else:
        return None
