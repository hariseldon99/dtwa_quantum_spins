#!/bin/bash
#########################################################################
## Name of my job
#PBS -N dtwa_test

## Name of the job queue
#PBS -q S30

## Walltime
#PBS -l walltime=00:02:00

##Number of nodes and procs per node.
#PBS -l nodes=1:ppn=14

##Send me email when my job aborts, begins, or ends
#PBS -m ea
#PBS -M daneel@utexas.edu

## Name of python script to be executed
SCRIPT="./dtwa_test.py"
#########################################################################

##Export all PBS environment variables
#PBS -V
##Output file. Combine stdout and stderr into one
#PBS -j oe
cd $PBS_O_WORKDIR
## Number of OpenMP threads to be used by the blas library. Keep this small
export OMP_NUM_THREADS=2
##Load these modules before running
module load openblas openmpi anaconda
BEGINTIME=$(date +"%s")
mpirun python -W ignore $SCRIPT 
ENDTIME=$(date +"%s")
ELAPSED_TIME=$(($ENDTIME-$BEGINTIME))
echo "#Runtime: $(($ELAPSED_TIME / 60)) minutes and $(($ELAPSED_TIME % 60)) seconds."
