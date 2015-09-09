dtwa_ising_longrange
=============

Discrete Truncated Wigner Approximation for quantum spins and transverse fields with time-periodic drive


Usage
-----


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
