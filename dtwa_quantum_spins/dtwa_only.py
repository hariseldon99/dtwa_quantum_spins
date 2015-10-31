#!/usr/bin/env python
from __future__ import division, print_function


from mpi4py import MPI
from reductions import Intracomm
from redirect_stdout import stdout_redirected
import random
import numpy as np
from scipy.integrate import odeint
from pprint import pprint
from tabulate import tabulate

from consts import *
from funcs import *
from classes import *

#Try to import mkl if it is available
try:
  import mkl
  mkl_avail = True
except ImportError:
  mkl_avail = False

def func_dtwa(s, t, param):
    """
    The RHS of general case, per Schachemmayer eq A2
    """
    N = param.latsize
    #s[0:N] = sx , s[N:2*N] = sy, s[2*N:3*N] = sz
    jsx = 2.0 * param.jx * param.jmat.dot(s[0:N])/param.norm
    jsx += 2.0 * param.hx
    jsy = 2.0 * param.jy * param.jmat.dot(s[N:2*N])/param.norm
    jsy += 2.0 * param.hy
    jsz = 2.0 * param.jz * param.jmat.dot(s[2*N:3*N])/param.norm
    jsz += 2.0 * param.hz
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
    diag_jsx = np.diagflat((param.jmat.dot(s[0:N])))/param.norm
    diag_jsy = np.diagflat((param.jmat.dot(s[N:2*N])))/param.norm
    #diag_jsz = np.diagflat((param.jmat.dot(s[2*N:3*N])))/param.norm
    hash_jsx = (np.multiply(param.jmat.T, s[0:N]).T)/param.norm
    hash_jsy = (np.multiply(param.jmat.T, s[N:2*N]).T)/param.norm
    hash_jsz = (np.multiply(param.jmat.T, s[2*N:3*N]).T)/param.norm
    full_jacobian[0:N, N:2*N] = param.jz * diag_jsx + drivemat * param.hz\
      -param.jy * hash_jsz
    full_jacobian[N:2*N, 0:N] = -param.jz * diag_jsx - \
      param.hz + param.jx * hash_jsz
    full_jacobian[0:N, 2*N:3*N] = -param.jy * diag_jsy - \
      param.hy + param.jz * hash_jsy
    full_jacobian[2*N:3*N, 0:N] = param.jy * diag_jsy + \
      param.hy - param.jx * hash_jsy
    full_jacobian[N:2*N, 2*N:3*N] = param.jx * diag_jsx + \
      param.hx - param.jz * hash_jsx
    full_jacobian[2*N:3*N, N:2*N] = -param.jx * diag_jsx - \
      param.hx + param.jy * hash_jsx
    return full_jacobian

class Dtwa_System:
  """
    Class that creates the dTWA system.
    
       Introduction:  
	This class instantiates an object encapsulating the dTWA problem.
	It has all MPI_Gather routines for aggregating observable data
	from the different random samples of initial conditions (which 
	are run in parallel), and has methods that sample the trajectories
	and execute the dTWA methods ( without BBGKY). 
	These methods call integrators from scipy and time-evolve 
	all the randomly sampled initial conditions.
  """

  def __init__(self, params, mpicomm, n_t=2000,\
			  seed_offset=0,   jac=False,\
			    verbose=True):
    """
    Initiates an instance of the Dtwa_System class. Copies parameters
    over from an instance of ParamData and stores precalculated objects .
    
       Usage:
       d = Dtwa_System(Paramdata, MPI_COMMUNICATOR, n_t=2000,\ 
			 bbgky=False, jac=False,\ 
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
       seed_offset      = Offset in the seed. The initial conditions are 
			  sampled randomly by each processor using the 
			  random generator in python with unique seeds for
			  each processor. Each processor adds seeed_offset 
			  to its seed. This allows you to ensure that 
			  separate dTWA objects have uniquely random initial 
			  states by changing seed_offset.
			  Defaults to 0.
       bbgky          =   Boolean for choosing the BBGKY evolution,
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

    self.__dict__.update(params.__dict__)
    self.jac = jac
    self.n_t = n_t
    self.comm = mpicomm
    self.seed_offset = seed_offset
    #Booleans for verbosity and for calculating site data
    self.verbose = verbose
    N = params.latsize
 
  def dtwa_only(self, time_info):
      comm = self.comm
      N = self.latsize
      rank = comm.rank
      
      if rank == root and self.verbose:
	  pprint("# Run parameters:")
	  pprint(vars(self), depth=2)
      if rank == root and not self.verbose:
	  pprint("# Starting run ...")
      if type(time_info) is tuple:
	(t_init, t_final, n_steps) = time_info
	dt = (t_final-t_init)/(n_steps-1.0)
	t_output = np.arange(t_init, t_final, dt)
      elif type(time_info) is list or np.ndarray:
	t_output = time_info
      elif rank == root:
	print("Please enter either a tuple or a list for the time interval") 
	exit(0)
      else:
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
	  np.random.seed(local_seeds[runcount] + self.seed_offset)
	  #According to Schachenmayer, the wigner function of the quantum
	  #state generates the below initial conditions classically
	  sx_init = np.ones(N)
	  sy_init = 2.0 * np.random.randint(0,2, size=N) - 1.0
	  sz_init = 2.0 * np.random.randint(0,2, size=N) - 1.0
	  #Set initial conditions for the dynamics locally to vector 
	  #s_init and store it as [s^x,s^x,s^x, .... s^y,s^y,s^y ..., 
	  #s^z,s^z,s^z, ...]
	  s_init = np.concatenate((sx_init, sy_init, sz_init))
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
	  sx_expct = np.sum(s[:, 0:N], axis=1) 
	  sy_expct = np.sum(s[:, N:2*N], axis=1) 
	  sz_expct = np.sum(s[:, 2*N:3*N], axis=1) 
	  
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

	  localdata = OutData(t_output, sx_expct, sy_expct,\
	    sz_expct, sx_var, sy_var, sz_var, sxy_var, sxz_var, \
	      syz_var, self)
	  list_of_local_data.append(localdata)
	  
      #After loop above  sum reduce (don't forget to average) all locally
      #calculated expectations at each time to root
      outdat = \
	sum_reduce_all_data(self, list_of_local_data, t_output, comm)    
	  
      if rank == root:
	  outdat.normalize_data(self.n_t, N)
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
    data as an object. 
    
    
       Usage:
       data = d.evolve(times)
       
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
      Return value: 
      An OutData object that contains:
	1. The times, bound to the method t_output
	2. The single site observables (x,y and z), 
	   bound to the methods 'sx,sy,sz' respectively and 
	3. All correlation sums (xx, yy, zz, xy, xz and yz), 
	   bound to the methods 
	   'sxvar, syvar, szvar, sxyvar, sxzvar, syzvar'
	   respectively
    """
    
    return self.dtwa_only(time_info)

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    #Initiate the parameters in object
    p = ParamData(latsize=101, beta=1.0)
    #Initiate the DTWA system with the parameters and niter
    d = Dtwa_System(p, comm, n_t=20)
    data = d.evolve((0.0, 1.0, 1000))
