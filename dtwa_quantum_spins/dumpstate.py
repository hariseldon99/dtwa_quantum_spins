from consts import *
import numpy as np
from mpi4py import MPI
import h5py

def open_statefile(params, filename=fullstate_fname,mode='w', driver='mpio'):
    f = h5py.File(filename, mode, driver=driver, comm=params.comm)
    return f

def build_statefile(hdf5_fp, all_ntlocs, t_output, params):
    """
    Create the datasets. All procs must do this
    The datasets are arranged in groups of times, and each time
    has one dataset per sample, labelled by (MPI rank)_iter where iter
    is an iterable in [0, nt_loc)
    """
    N = params.latsize
    for t in params.fullstate_times:
        tm = t_output[(np.abs(t_output-t)).argmin()] #find nearest matching time
        time_group = hdf5_fp.create_group('time_' + str(tm).decode("utf-8"))
        time_group.attrs['time'] = t
        for (rank, ntloc) in enumerate(all_ntlocs):
            for loc_it in xrange(ntloc):
                samplename = "sample_" + str(rank).decode("utf-8") +\
                                            "_" + str(loc_it).decode("utf-8")
                sample_group = time_group.create_group(samplename)
                sample_group.create_dataset("s",(3,N),dtype='f')
                sample_group.create_dataset("g",(3,3,N,N),dtype='f')
                sample_group.attrs['phasepoint'] = np.array([rank, loc_it])

def dump_states(hdf5_fp, sdata, sample_iter, t_output, params):
    """
    A particular MPI process with a particular rank dumps it's state data at a 
    particular iteration "iter" of nt_loc to the corresponding "rank_iter" named 
    dataset.
    """
    rank = params.comm.rank  
    N = params.latsize    
    #Have a particular process write data to its sample points
    #all procs must do this
    for t in params.fullstate_times:
        tidx = (np.abs(t_output-t)).argmin()
        samplename = 'sample_' + str(rank).decode("utf-8") + "_" + \
                                               str(sample_iter).decode("utf-8")
        s = hdf5_fp['time_' + str(t).decode("utf-8")+'/'+ samplename +'/'+ 's'] 
        s[:] = sdata[tidx, 0:3*N].reshape(3,N)
        g = hdf5_fp['time_' + str(t).decode("utf-8")+'/'+ samplename +'/'+ 'g'] 
        g[:] = sdata[tidx, 3*N:].reshape(3,3,N,N)
   

def close_statefile(hdf5_fp):
    hdf5_fp.close()