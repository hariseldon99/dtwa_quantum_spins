from distutils.core import setup, Extension
 
module1 = Extension('lorenzo_bbgky', sources = ['testmodule.c', 'libtest.c'])
 
setup (name = 'dtwa_quantum_spins',
        version = '1.0',
        description = 'Discrete Truncated Wigner Approximation (dTWA) for quantum spins',
        long_description=\
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
         """,
        url='https://github.com/hariseldon99/dtwa_quantum_spins',
         # Author details
	author='Analabha Roy',
	author_email='daneel@utexas.edu',

	# Choose your license
	license='GPL',

	# See https://pypi.python.org/pypi?%3Aaction=list_classifiers
	classifiers=[
	    # How mature is this project? Common values are
	    #   3 - Alpha
	    #   4 - Beta
	    #   5 - Production/Stable
	    'Development Status :: 3 - Alpha',

	    # Indicate who your project is intended for
	    'Intended Audience :: Physicists',
	    'Topic :: Numerical Quantum Simulations :: Dynamics',

	    # Pick your license as you wish (should match "license" above)
	    'License :: GPL License',

	    # Specify the Python versions you support here. In particular, ensure
	    # that you indicate whether you support Python 2, Python 3 or both.
	    'Programming Language :: Python :: 2',
	    'Programming Language :: Python :: 2.6',
	    'Programming Language :: Python :: 2.7',
	],
        ext_modules = [module1])
