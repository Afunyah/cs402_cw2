#!/bin/bash

#SBATCH --job-name=*_*_*_*_*_*_*
#SBATCH --account=cs402users
#SBATCH --partition=cs402-viper
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=00:01:00
#SBATCH -o ./slurms_out/vanilla/slurm-%j.out


module purge
module load cs402-mpi

mpirun ./vanilla/karman_vanilla -i initial.bin -o vanilla/karman_vanilla.bin
./save2ppm vanilla/karman_vanilla_$SLURM_JOB_ID