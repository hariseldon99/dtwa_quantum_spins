#!/bin/bash
#########################################################################
## Name of my job
#PBS -N dtwa
#PBS -l walltime=1:00:00
#########################################################################
##Export all PBS environment variables
#PBS -V
#########################################################################
##Output file. Combine stdout and stderr into one
#PBS -j oe 
#########################################################################
##Number of nodes and procs per node.
##The ib at the end means infiniband. Use that or else MPI gets confused 
##with ethernet
#PBS -l nodes=10:ppn=8
#########################################################################
##Send me email when my job aborts, begins, or ends
#PBS -m ea
#PBS -M daneel@sun.ac.za
#########################################################################
SCRIPT="./dtwa_2d_spins.py"

cd $PBS_O_WORKDIR


#########################################################################
##Make a list of allocated nodes(cores)
##Note that if multiple jobs run in same directory, use different names
##for example, add on jobid nmber.
#########################################################################
# How many cores total do we have?
NO_OF_CORES=$(cat $PBS_NODEFILE | wc -l)
#########################################################################

#########################################################################
##Now run my prog
module load dot
BEGINTIME=$(date +"%s")
mpirun -np $NO_OF_CORES python -W ignore $SCRIPT 
ENDTIME=$(date +"%s")
ELAPSED_TIME=$(($ENDTIME-$BEGINTIME))

echo "#Runtime: $(($ELAPSED_TIME / 60)) minutes and $(($ELAPSED_TIME % 60)) seconds."
