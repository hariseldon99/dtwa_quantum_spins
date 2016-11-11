# Some constant objects
from numpy import eye, zeros

threshold = 1e-4
root = 0
#This is the kronecker delta symbol for vector indices
deltaij = eye(3)
#This is the Levi-Civita symbol for vector indices
eijk = zeros((3, 3, 3))
eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1

#Progressbar widgets
try:
    from progressbar import Bar, Counter, ETA, Percentage
    pbar_avail = True
    widgets_bbgky = ['BBGKY Dynamics (MPI root): ', Percentage(), ' ', Bar(), ' ', ETA()]
except ImportError:
    pbar_avail = False
    widgets = None
