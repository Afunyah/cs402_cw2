#!/bin/bash

#SBATCH --job-name=*_*_*_*_*_*_*
#SBATCH --account=cs402users
#SBATCH --partition=cs402-viper
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=00:10:00


module purge
module load cs402-mpi

# export OMP_NUM_THREADS=4

# mpirun ./build/bin/karman -i initial.bin
# mpirun -n 2 valgrind --track-origins=yes ./build/bin/karman -i initial.bin
mpirun -n 1 ./karman -i --imax=1320 --jmax=240 --infile='doesnotexist.bin'
# gprof ./karman
# srun -n 4 ./build/bin/karman --imax=1320 --jmax=240 --infile='doesnotexist.bin'
./save2ppm karman_$SLURM_JOB_ID