#!/bin/bash

#SBATCH --job-name=*_*_*_*_*_*_*
#SBATCH --account=cs402users
#SBATCH --partition=cs402-viper
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=00:03:00
#SBATCH -o ./slurms_out/slurm-%j.out


module purge
module load cs402-mpi

# mpirun ./build/bin/karman -i initial.bin
# mpirun -n 2 valgrind --track-origins=yes ./build/bin/karman -i initial.bin
# mpirun -n 2 valgrind ./build/bin/karman -i initial.bin
mpirun -n 1 ./build/bin/karman -i initial.bin

./save2ppm karman_$SLURM_JOB_ID