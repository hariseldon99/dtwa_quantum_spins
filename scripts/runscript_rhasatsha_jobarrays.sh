#!/bin/bash
#########################################################################
## Name of my job
#PBS -N bolliger
#PBS -l walltime=7:00:00:00
#########################################################################
##Export all PBS environment variables
#PBS -V
#########################################################################
##Output file. Combine stdout and stderr into one
#PBS -j oe ${PBS_JOBNAME}.o${PBS_JOBID}
#########################################################################
##Number of nodes and procs per node.
##The ib at the end means infiniband. Use that or else MPI gets confused 
##with ethernet
#PBS -l nodes=20:ppn=8:scratch
#########################################################################
##Send me email when my job aborts, begins, or ends
#PBS -m ea
#PBS -M daneel@sun.ac.za
#########################################################################
#Job array 
#PBS -t 1-50

SCRIPT="./run_dtwa.py"

# make sure I'm the only one that can read my output
umask 0077

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
mpirun -np $NO_OF_CORES python $SCRIPT 
ENDTIME=$(date +"%s")
ELAPSED_TIME=$(($ENDTIME-$BEGINTIME))

echo "#Runtime: $(($ELAPSED_TIME / 60)) minutes and $(($ELAPSED_TIME % 60)) seconds."
