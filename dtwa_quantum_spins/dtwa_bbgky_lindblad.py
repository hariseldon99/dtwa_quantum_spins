#!/usr/bin/env python
from __future__ import division, print_function

from mpi4py import MPI
from reductions import Intracomm
from redirect_stdout import stdout_redirected
import copy
import numpy as np
#from scipy.integrate import odeint
from odeintw import odeintw
from pprint import pprint
from tabulate import tabulate

from consts import *
from funcs import *
from classes import *
from dtwa_only import func_dtwa
from dtwa_bbgky import func_dtwa_bbgky

#Try to import mkl if it is available
try:
  import mkl
  mkl_avail = True
except ImportError:
  mkl_avail = False

#Try to import progressbars if available
try:
    import progressbar
    pbar_avail = True
except ImportError:
    pbar_avail = False
 
def func_dtwa_bbgky_lindblad(s, t, param):
    """
    Lindblad Decoherence terms added to func_dtwa_bbgky
    """
    N = param.latsize
    dsgdt_closed = func_dtwa_bbgky(s, t, param)
    #svec  is the tensor s^l_\mu
    #G = s[3*N:].reshape(3,3,N,N) is the tensor g^{ab}_{\mu\nu}.
    sview = s.view()
    dsview_c = dsgdt_closed.view()
    
    stensor = sview[0:3*N].reshape(3, N)
    dstensor_c = dsview_c[0:3*N].reshape(3, N)
    
    gtensor = sview[3*N:].reshape(3, 3, N, N)    
    dgtensor_c = dsview_c[3*N:].reshape(3, 3, N, N)
    gtensor[:,:,range(N),range(N)] = 0.0 #Set the diagonals of g_munu to 0
    dgtensor_c[:,:,range(N),range(N)] = 0.0 #Set the diagonals of g_munu to 0
    dsxdt = dstensor_c[0,:] - 0.5 * (param.gamma_r + param.gamma_el) * stensor[0,:]
    dsydt = dstensor_c[1,:] - 0.5 * (param.gamma_r + param.gamma_el) * stensor[1,:]
    dszdt = dstensor_c[2,:] - param.gamma_r * stensor[2,:]
    dgdt = dgtensor_c - (param.gamma_r + 0.5 * param.gamma_el) * gtensor
    
    ampl = 0.5 * (param.gamma_r - param.gamma_el)
    dgdt[0,0,:,:] += ampl * gtensor[0,0,:,:]
    dgdt[1,1,:,:] += ampl * gtensor[1,1,:,:]
    dgdt[2,2,:,:] -= ampl * gtensor[1,1,:,:]
    
    dgdt[0,1,:,:] += ampl * gtensor[0,1,:,:]
    dgdt[1,0,:,:] += ampl * gtensor[1,0,:,:]

    dgdt[:,:,range(N),range(N)] = 0.0 #Set the diagonals of g_munu to 0
    #Flatten it before returning
    return np.concatenate((dsxdt.flatten(),dsydt.flatten(),dszdt.flatten(), dgdt.flatten()))

class Dtwa_BBGKY_Lindblad_System:
  """
    Class that creates the dTWA system.
    
       Introduction:  
	This class instantiates an object encapsulating the dTWA problem.
	It has all MPI_Gather routines for aggregating observable data
	from the different random samples of initial conditions (which 
	are run in parallel), and has methods that sample the trajectories
	and execute the dTWA methods (2nd order i.e. with 
	BBGKY). These methods call integrators from scipy and time-evolve 
	all the randomly sampled initial conditions.
  """

  def __init__(self, params, mpicomm, n_t=2000,\
			  seed_offset=0, decoherence=None ,verbose=True):
    """
    Initiates an instance of the Dtwa_System class. Copies parameters
    over from an instance of ParamData and stores precalculated objects .
    
       Usage:
       d = Dtwa_System(Paramdata, MPI_COMMUNICATOR, n_t=2000,\ 
                                                   bbgky=False, verbose=True)
       
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
       decoherence      = Decoherence amplitudes as defined in Foss-Feig.
                      Enter as tuple (Gamma_ud, Gamma_du, Gamma_el)
                      Defaults to None
       verbose		= Boolean for choosing verbose outputs. Setting 
			  to 'True' dumps verbose output to stdout, which
			  consists of full output from the integrator, as
			  well as the output of the time derivative
			  of the Weyl symbol of the Hamiltonian that you
			  have provided via the 'hopmat' and other input
			  in ParamData. Defaults to 'False'.			  
			  
      Return value: 
      An object that stores all the parameters above.
    """

    self.__dict__.update(params.__dict__)
    self.n_t = n_t
    self.comm = mpicomm
    self.seed_offset = seed_offset
    #Booleans for verbosity and for calculating site data
    self.verbose = verbose
    if decoherence is not None:
        self.gamma_ud, self.gamma_du, self.gamma_el = decoherence
    else:
        self.gamma_ud, self.gamma_du, self.gamma_el = 0.0, 0.0, 0.0
    self.gamma_r = self.gamma_ud + self.gamma_du
    
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
	(t_init, t_final, n_steps) = time_info
	dt = (t_final-t_init)/(n_steps-1.0)
	t_output = np.arange(t_init, t_final, dt)
      elif type(time_info) is list  or np.ndarray:
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
      if pbar_avail:
	if self.comm.rank == root and self.verbose: 
	  pbar_max = nt_loc-1
	  bar = progressbar.ProgressBar(widgets=widgets_bbgky,\
	    max_value=pbar_max, redirect_stdout=False)	  
	  
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
	  s_init_spins, s_init_corrs = sample(self, sampling, \
	    local_seeds[runcount] + self.seed_offset, complexify=True)
	  #Redirect unwanted stdout warning messages to /dev/null
	  with stdout_redirected():
	    if self.verbose:
		s, info = odeintw(func_dtwa_bbgky_lindblad, \
		  np.concatenate((s_init_spins, s_init_corrs)),t_output, \
		    args=(self,), Dfun=None, full_output=True)
	    else:
		s = odeintw(func_dtwa_bbgky_lindblad, \
		  np.concatenate((s_init_spins, s_init_corrs)), t_output, \
		    args=(self,), Dfun=None)
	  
	  #Computes |dH/dt|^2 for a particular alphavec & weighes it 
	  #If the rms over alphavec of these are 0, then each H is const
	  if self.verbose:
	    hws = weyl_hamilt(s,t_output, self)
	    dhwdt = np.array([t_deriv(hw, t_output) for hw in hws])
	    dhwdt_abs2 = np.square(dhwdt) 
	    list_of_dhwdt_abs2.extend(dhwdt_abs2)
	  
	  s = np.array(s, dtype=np.complex128)#Widen memory to reduce overflows
	  localdata = bbgky_observables(t_output, s, self)
	  list_of_local_data.append(localdata)
	  if self.verbose and pbar_avail and self.comm.rank == root:
	    bar.update(runcount)
	  
      #After loop above  sum reduce (don't forget to average) all locally
      #calculated expectations at each time to root
      outdat = \
	sum_reduce_all_data(self, list_of_local_data, t_output, comm) 
      if self.verbose:
	dhwdt_abs2_locsum = np.sum(list_of_dhwdt_abs2, axis=0)
	dhwdt_abs2_totals = np.zeros_like(dhwdt_abs2_locsum)\
	  if rank == root else None
	temp_comm = Intracomm(comm)
	dhwdt_abs2_totals = temp_comm.reduce(dhwdt_abs2_locsum, root=root)
	if rank == root:
	  dhwdt_abs2_totals = dhwdt_abs2_totals/(self.n_t * N * N)
	  dhwdt_abs_totals = np.sqrt(dhwdt_abs2_totals)
	
      if rank == root:
	  outdat.normalize_data(self.n_t, N)
	  if self.verbose:
	    print("\nt-deriv of Hamilt (abs square) with wigner avg: ")
	    print("  ")
	    print(tabulate({"time": t_output, \
	      "dhwdt_abs": dhwdt_abs_totals}, \
	      headers="keys", floatfmt=".6f"))
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
    data as an object. An optional argument is the sampling scheme.
    
    
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
			  Note that this is implemented only if the 'bbgky' 
			  option in the 'Dtwa_System' object is set to True.
			  If not (ie if you're running pure dtwa), then only
			  "spr" sampling is implemented no matter what this
			  option is set to.

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
    
    return self.dtwa_bbgky(time_info, sampling)

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    #Initiate the parameters in object
    p = ParamData(latsize=101, beta=1.0)
    #Initiate the DTWA system with the parameters and niter
    d = Dtwa_System(p, comm, n_t=20)
    data = d.evolve((0.0, 1.0, 1000))
