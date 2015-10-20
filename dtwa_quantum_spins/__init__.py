# Author:  Analabha roy
# Contact: daneel@utexas.edu
from __future__ import division, print_function

"""
    Discrete Truncated Wigner Approximation (dTWA) for quantum spins
    and transverse fields with time-periodic drive

    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    * Copyright (c) 2015 Analabha Roy (daneel@utexas.edu)
    *
    *This is free software: you can redistribute it and/or modify it under
    *the terms of version 2 of the GNU Lesser General Public License
    *as published by the Free Software Foundation.
    *Notes:
    *1. The initial state is currently hard coded to be the classical ground
    *    state
    *2. Primary references are
    *   PRM:  arXiv:1510.03768
    *   Anatoli: Ann. Phys 325 (2010) 1790-1852
    *   Mauritz: arXiv:1209.3697
    *   Schachenmayer: arXiv:1408.4441
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
"""
__version__   = '0.1'
__author__    = 'Analabha Roy'
__credits__   = 'Lorenzo Pucci, NiTheP Stellenbosch'

__all__ = ["dtwa_quantum_spins", "reductions","redirect_stdout"]
from dtwa_quantum_spins import *
