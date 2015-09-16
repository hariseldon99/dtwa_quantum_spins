dtwa_ising_longrange
=============

Discrete Truncated Wigner Approximation for quantum spins and transverse fields with time-periodic drive.
There are three (3) sampling schemes used, depending on the choice of phase point operators


Usage
-----
The code is a python module. An instance is shown below.

```python
import numpy as np
import sys
from mpi4py import MPI
sys.path.append("/path/to/dtwa_quantum_spins/")
import dtwa_quantum_spins as dtwa

#Your codes
#...
#...

#Initiate the parameters for dTWA
p = dtwa.ParamData(hopmat=hopping_matrix,norm=scales_hopmat, latsize=lattice_size,\
	h0=periodic_drive_amplitude, omega=periodic_drive_frequency, \
	  hx=x_transverse_field, hy=y_transverse_field, hz=z_transverse_field\
	    jx=x_hopping, jy=x_hopping, jz=x_hopping)
#Additional parameters like filenames etc. can be set directly
p.output_magx = "sx_time.txt"
p.output_magy = "sy_time.txt"
p.output_magz = "sz_time.txt"

p.output_sxvar = "sxvar_time.txt"
p.output_syvar = "syvar_time.txt"
p.output_szvar = "szvar_time.txt"

p.output_sxyvar = "sxyvar_time.txt"
p.output_sxzvar = "sxzvar_time.txt"
p.output_syzvar = "syzvar_time.txt"

#Initiate the DTWA system with the parameters:
#Set file_output to True if you want the data in the output dictionary to be dumped to 
#the files named in 'p' above.
#Set 's_order' to True if you want second order dTWA.
#Set 'jac' to True if you want the integrator to use the jacobian of the BBGKY dynamics. Use
#sparingly, since the size of the jacobian grows as lattice_size**2.
#The 'sitedata' boolean dumps the single site and correlations of 
#the middle site and its neighbor. Set to False if not needed.

d = dtwa.Dtwa_System(p, MPI_COMMUNICATOR, n_t=number_of_sampled_trajectories, \
	      file_output=True, s_order=False, jac=False, verbose=True, sitedata=False)

#Prepare the times
t0 = 0.0
t1 = 1.0
nsteps = 200

#Get the output dictionary
#This is basically the times and the single site observables (x,y and z) and 
#all correlation sums (xx, yy, zz, xy, xz and yz)

data = d.evolve((t0, t1, nsteps))

```



###Relevant docs for the bundled version of mpi4py reduce:
* [GitHub](https://github.com/mpi4py/mpi4py/blob/master/demo/reductions/reductions.py)
* [readthedocs.org](https://mpi4py.readthedocs.org/en/latest/overview.html#collective-communications)
* [Google Groups](https://groups.google.com/forum/#!msg/mpi4py/t8HZoYg8Ldc/-erl6BMKpLAJ)

###Relevant Literature:
* [Wooters: Annals of Physics 176, 1–21 (1987)](http://dx.doi.org/10.1016/0003-4916(87)90176-X)
* [Anatoli : Ann. Phys 325 (2010) 1790-1852](http://arxiv.org/abs/0905.3384)
* [Mauritz: New J. Phys. 15, 083007 (2013)](http://arxiv.org/abs/1209.3697)
* [Schachenmayer: Phys. Rev. X 5 011022 (2015)](http://arxiv.org/abs/1408.4441)

###External dependencies:
1. mpi4py - MPI for Python

    _\_-MPI (Parallelizes the different samplings of the dtwa)

2. numpy - Numerical Python (Various uses)

3. scipy  - Scientific Python

    _\_-integrate 

    _| \_-odeint (Integrates the BBGKY dynamics of the sampled state)

    _| \_-signal 
    
    _| \_-fftconvolve (Used for calculating spin correlations)

4. tabulate - Tabulate module 
    
    _\_-tabulate (Used for dumping tabular data)
