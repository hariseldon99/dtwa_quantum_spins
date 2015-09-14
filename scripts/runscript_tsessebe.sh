#!/bin/bash
#########################################################################
## Name of my job
#PBS -N test
#PBS -l walltime=1:00:00
#########################################################################
##Export all PBS environment variables
#PBS -V
#########################################################################
##Output file. Combine stdout and stderr into one
#PBS -o stdout.dat
#PBS -e stderr.dat
#PBS -j oe 
#########################################################################
##Number of nodes and procs per node.
##See docs at http://wiki.chpc.ac.za/howto:pbs-pro_job_submission_examples 
#PBS -l select=5:ncpus=8:mpiprocs=8
#########################################################################
##Send me email when my job aborts, begins, or ends
#PBS -m ea
#PBS -M daneel@sun.ac.za
#########################################################################

SCRIPT = "./dtwa_2d_spins.py"
# Make sure I'm the only one that can read my output
umask 0077
# Load the module system
source /etc/profile.d/modules.sh
#Load relevant modules. Load them with THESE TWO LINES, NOT FROM ONE LINE
module load dot intel
module load gcc/4.9.1 Anaconda/2.1.0

cd $PBS_O_WORKDIR

#########################################################################
# How many cores total do we have?
NO_OF_CORES=$(cat $PBS_NODEFILE | wc -l)
#########################################################################

#########################################################################
##Now, run the code
BEGINTIME=$(date +"%s")
mpirun -np $NO_OF_CORES -machinefile $PBS_NODEFILE  python -W ignore $SCRIPT 
ENDTIME=$(date +"%s")
ELAPSED_TIME=$(($ENDTIME-$BEGINTIME))

echo "#Runtime: $(($ELAPSED_TIME / 60)) minutes and $(($ELAPSED_TIME % 60)) seconds."
