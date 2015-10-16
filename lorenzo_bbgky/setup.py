from distutils.core import setup, Extension
 
module1 = Extension('bbgkymodule', sources = ['bbgkymodule.c', 'lorenzo_bbgky.c'])
 
setup (name = 'bbgkymodule',
        version = '1.0',
        description = 'Lorenzos BBGKY subroutine',
        ext_modules = [module1])
