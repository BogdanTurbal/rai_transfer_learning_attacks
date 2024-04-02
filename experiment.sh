#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH -t 10:00:00
#SBATCH --cpus-per-task=18
#SBATCH --mem=80G
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=bogdan.turbal.y@gmail.com
#SBATCH -o /gpfs/work4/0/tese0660/projects/transf_learn_attack/gin_imp_%j.out

module purge
module load 2023

cd /gpfs/work4/0/tese0660/projects/transf_learn_attack

activate att

CONT_PATH = /gpfs/work4/0/tese0660/projects/transf_learn_attack/

mkdir /gpfs/work4/0/tese0660/projects/transf_learn_attack/0_id/
mkdir /gpfs/work4/0/tese0660/projects/transf_learn_attack/1_id/
mkdir /gpfs/work4/0/tese0660/projects/transf_learn_attack/2_id/
mkdir /gpfs/work4/0/tese0660/projects/transf_learn_attack/3_id/
mkdir /gpfs/work4/0/tese0660/projects/transf_learn_attack/4_id/
mkdir /gpfs/work4/0/tese0660/projects/transf_learn_attack/5_id/
mkdir /gpfs/work4/0/tese0660/projects/transf_learn_attack/6_id/
mkdir /gpfs/work4/0/tese0660/projects/transf_learn_attack/7_id/
mkdir /gpfs/work4/0/tese0660/projects/transf_learn_attack/8_id/

python /gpfs/work4/0/tese0660/projects/transf_learn_attack/attack_trans.py /gpfs/work4/0/tese0660/projects/transf_learn_attack/0_id/ --msl 2 --ne 2 --mae 1024 --mel 0 --seed 1 --run 0 --mod_id 0&
python /gpfs/work4/0/tese0660/projects/transf_learn_attack/attack_trans.py /gpfs/work4/0/tese0660/projects/transf_learn_attack/1_id/ --msl 2 --ne 2 --mae 1024 --mel 0 --seed 42 --run 1 --mod_id 0&
python /gpfs/work4/0/tese0660/projects/transf_learn_attack/attack_trans.py /gpfs/work4/0/tese0660/projects/transf_learn_attack/2_id/ --msl 2 --ne 2 --mae 1024 --mel 0 --seed 1234 --run 2 --mod_id 0&
python /gpfs/work4/0/tese0660/projects/transf_learn_attack/attack_trans.py /gpfs/work4/0/tese0660/projects/transf_learn_attack/3_id/ --msl 2 --ne 2 --mae 1024 --mel 0 --seed 1 --run 0 --mod_id 1&
python /gpfs/work4/0/tese0660/projects/transf_learn_attack/attack_trans.py /gpfs/work4/0/tese0660/projects/transf_learn_attack/4_id/ --msl 2 --ne 2 --mae 1024 --mel 0 --seed 42 --run 1 --mod_id 1&
python /gpfs/work4/0/tese0660/projects/transf_learn_attack/attack_trans.py /gpfs/work4/0/tese0660/projects/transf_learn_attack/5_id/ --msl 2 --ne 2 --mae 1024 --mel 0 --seed 1234 --run 2 --mod_id 1&
python /gpfs/work4/0/tese0660/projects/transf_learn_attack/attack_trans.py /gpfs/work4/0/tese0660/projects/transf_learn_attack/6_id/ --msl 2 --ne 2 --mae 1024 --mel 0 --seed 1 --run 0 --mod_id 2&
python /gpfs/work4/0/tese0660/projects/transf_learn_attack/attack_trans.py /gpfs/work4/0/tese0660/projects/transf_learn_attack/7_id/ --msl 2 --ne 2 --mae 1024 --mel 0 --seed 42 --run 1 --mod_id 2&
python /gpfs/work4/0/tese0660/projects/transf_learn_attack/attack_trans.py /gpfs/work4/0/tese0660/projects/transf_learn_attack/8_id/ --msl 2 --ne 2 --mae 1024 --mel 0 --seed 1234 --run 2 --mod_id 2&

wait

