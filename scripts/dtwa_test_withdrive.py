import numpy as np
from mpi4py import MPI
import dtwa_quantum_spins as dtwa
import matplotlib.pyplot as plt
#Parameters
niter = 5000
lattice_size = 9
alpha = 0.0
jx, jy, jz = -1.0, 0.0, 0.0
hx, hy, hz = 0.0, 0.0, -1.0
amp = 25.0
f = 25.0
hdc = 0.1

#Prepare the times
t0 = 0.0
ncyc = 20.0
nsteps = 1000

def run_dtwa(timetuple, amp, f, hdc, comm):
    
    rank = comm.Get_rank()
    size = comm.Get_size()

    #seed(s)
    #Build the hopping matrix
    size = lattice_size
    jmat = np.zeros((size, size))
    for mu in xrange(size):
        for nu in xrange(mu, size):
            if mu != nu:
                dmn = np.abs(mu-nu)
                jmat[mu,nu] = 1.0/pow(dmn,alpha)

    #Initiate the parameters in object
    mid = np.floor(size/2).astype(int)
    kacnorm =2.0 * np.sum(1/(pow(np.arange(1, mid+1), alpha).astype(float)))
    #kacnorm = 1.0
    p = dtwa.ParamData(hopmat=(jmat+jmat.T),norm = kacnorm, latsize=size,\
                              jx=jx, jy=jy, jz=jz, hx=hx, hy=hy, hz=hz, omega=f, hdc=hdc, amp=amp)
  
    d = dtwa.Dtwa_System(p, comm, n_t=niter, verbose=False)
    

    data = d.evolve(timetuple, sampling="spr")

    if rank == 0:
        return data.t_output, data.sz.real
    else:
        return None

if __name__ == '__main__':
    
    comm = MPI.COMM_WORLD
    ts = (t0, ncyc, nsteps)
    time, sz_dtwa = run_dtwa(ts, amp, f, hdc, comm)
    
    if comm.Get_rank() == 0:
        plt.title("l8_hz-1_hdc_0p1_amp_25_dtwa")
        plt.plot(time, sz_dtwa, label = "sz dtwa")
        plt.xlabel("time")
        plt.ylim(-1,1)
        plt.legend()
        plt.show()