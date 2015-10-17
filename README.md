dtwa_quantum_spins
=============

Discrete Truncated Wigner Approximation (dTWA) for quantum spins and transverse fields with time-periodic drive.
There are three (3) sampling schemes used, depending on the choice of phase point operators. 

Introduction
-----
The code can be used in a single processor environment, or a multiprocessor grid using [mpi4py](http://mpi4py.scipy.org/),  the Python bindings of the MPI standard.

Installation
-----
Installation involves three steps. Install git, clone this code repository, install python and all dependencies and manually add import the module in a python session by adding the path to sys.path.

1. Installing git: If git is not already installed in your system, [follow the instructions here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git). 

2. Cloning this repository: If you are on any unix-like shell environment, whether an actual unix shell (like [bash](https://www.gnu.org/software/bash/) ), a graphical terminal emulator (like [xterm](http://invisible-island.net/xterm/xterm.html), [gnome-terminal](https://help.gnome.org/users/gnome-terminal/stable/), [yakuake](https://yakuake.kde.org/) etc.) on Linux with X-Windows ([Ubuntu](http://www.ubuntu.com/), [Debian](https://www.debian.org/), [OpenSuse](https://www.opensuse.org/en/) etc.) or an environment like [Cygwin](https://www.cygwin.com/) or [MinGW](http://mingw.org/) on Microsoft Windows, just install git if necessary and run
     ```
     $ git clone https://github.com/hariseldon99/dtwa-quantum_spins
     $ echo $PWD
     ```
This installs git to the path $PWD/dtwa_quantum_systems.     
In other cases, refer to [git setup guide](https://git-scm.com/book/en/v2/Getting-Started-First-Time-Git-Setup) and 
[git basics](https://git-scm.com/book/en/v2/Git-Basics-Getting-a-Git-Repository).

3. Install python and dependencies: If python is not already installed in your system, then refer to [these general instructions](https://wiki.python.org/moin/BeginnersGuide/Download) to download and install python and the dependencies given in the 'External dependencies' section below. Alternatively, install a python distribution like [anaconda](https://store.continuum.io/cshop/anaconda/) and use it's internal package management to install the required dependencies.

4. Import the module in python: Start a python shell (or write a python script) and run

  ```python
  >>> import sys
  >>> sys.path.append("/path/to/dtwa_quantum_spins/")
  >>> import dtwa_quantum_spins as dtwa
  ```
Here, /path/to/dtwa_quantum_spins/ is the output of ```echo $PWD``` in item 2 above.

Usage
-----
Usage examples are shown below.

Example 1: Obtaining Documentation
```python
import sys
sys.path.append("/path/to/dtwa_quantum_spins/")
import dtwa_quantum_spins as dtwa
help(dtwa)
```

Example 2:
```python
import numpy as np
import sys
from mpi4py import MPI
sys.path.append("/path/to/dtwa_quantum_spins/")
import dtwa_quantum_spins as dtwa

#Your codes
#...
#You'll need to create the hopping matrix.
#If you don't, then the default is is the dense Jmn hopping matrix with inverse 
#power law decay for periodic boundary conditions.
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
#1. Set file_output to True if you want the data in the output dictionary to be dumped to 
#   the files named in 'p' above.
#2. Set 's_order' to True if you want second order dTWA.
#3. Set 'jac' to True if you want the integrator to use the jacobian of the BBGKY dynamics. Use
#   sparingly, since the size of the jacobian grows as lattice_size**2.
#4. The 'sitedata' boolean dumps the single site and correlations of 
#   the middle site and its neighbor. Set to False if not needed.

d = dtwa.Dtwa_System(p, MPI_COMMUNICATOR, n_t=number_of_sampled_trajectories, \
	      file_output=True, s_order=False, jac=False, verbose=True, sitedata=False)

#Prepare the times
t0 = 0.0
t1 = 1.0
nsteps = 200

#Get the output dictionary
#This contains: 
#1. The times,
#2. The single site observables (x,y and z), and 
#3. All correlation sums (xx, yy, zz, xy, xz and yz).

#You can choose different sampling schemes
data = d.evolve((t0, t1, nsteps), sampling="spr")

```



Relevant Literature:
-----

###Relevant papers:
* [Wooters: Annals of Physics 176, 1–21 (1987)](http://dx.doi.org/10.1016/0003-4916(87)90176-X)
* [Anatoli : Ann. Phys 325 (2010) 1790-1852](http://arxiv.org/abs/0905.3384)
* [Mauritz: New J. Phys. 15, 083007 (2013)](http://arxiv.org/abs/1209.3697)
* [Schachenmayer: Phys. Rev. X 5 011022 (2015)](http://arxiv.org/abs/1408.4441)

###Relevant docs for the bundled version of mpi4py reduce:
* [GitHub](https://github.com/mpi4py/mpi4py/blob/master/demo/reductions/reductions.py)
* [readthedocs.org](https://mpi4py.readthedocs.org/en/latest/overview.html#collective-communications)
* [Google Groups](https://groups.google.com/forum/#!msg/mpi4py/t8HZoYg8Ldc/-erl6BMKpLAJ)


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

5. Lorenzo's bbgky module

    _\_-gcc/automake (compilers for the C module)
    
    _\_-cblas - Any BLAS library written in C

###TODO:
1. Let the final output be a matrix of spins at end time AND observables. Let the user calculate what he wants and distribute the memory via multiple MPI communicators.

2. Incorporate Lorenzo's code for the BBGKY function. His code is faster and better optimized. This can be done by either:
   
   a. Creating an actual extension module in C. Probably overkill, and I'd also like to avoid the overhead of learning 
      extension writing.
   
   b. Do the whole thing in Python, using ctypes to communicate with the external library. Apparently this might be too slow      for repeated calls.

3. Add the dynamics of open quantum systems (Lindblad) as a separate class similar to 'Dtwa_System', and add CPython 
   module of the same to lorenzo_bbgky

4. Lots of work on the docs
