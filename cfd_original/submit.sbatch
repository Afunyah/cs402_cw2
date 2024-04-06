#!/bin/bash

#SBATCH --job-name=*_*_*_*_*_*_*
#SBATCH --account=cs402users
#SBATCH --partition=cs402-viper
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=00:01:00
#SBATCH -o ./slurms_out/slurm-%j.out


module purge
module load cs402-mpi

srun ./build/bin/karman -i initial.bin
./save2ppm karman_$SLURM_JOB_ID