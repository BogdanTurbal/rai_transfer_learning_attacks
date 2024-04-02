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
module load CUDA/11.3.1

cd /gpfs/work4/0/tese0660/projects/transf_learn_attack

conda activate att

CONT_PATH = /gpfs/work4/0/tese0660/projects/transf_learn_attack/
python /gpfs/work4/0/tese0660/projects/transf_learn_attack/attack_trans.py CONT_PATH --msl 2 --ne 2 --mae 1024 --mel 0 --seed 1 --run 0 --mod_id 0&
python /gpfs/work4/0/tese0660/projects/transf_learn_attack/attack_trans.py CONT_PATH --msl 2 --ne 2 --mae 1024 --mel 0 --seed 42 --run 1 --mod_id 0&
python /gpfs/work4/0/tese0660/projects/transf_learn_attack/attack_trans.py CONT_PATH --msl 2 --ne 2 --mae 1024 --mel 0 --seed 1234 --run 2 --mod_id 0&

python /gpfs/work4/0/tese0660/projects/transf_learn_attack/attack_trans.py CONT_PATH --msl 2 --ne 2 --mae 1024 --mel 0 --seed 1 --run 0 --mod_id 1&
python /gpfs/work4/0/tese0660/projects/transf_learn_attack/attack_trans.py CONT_PATH --msl 2 --ne 2 --mae 1024 --mel 0 --seed 42 --run 1 --mod_id 1&
python /gpfs/work4/0/tese0660/projects/transf_learn_attack/attack_trans.py CONT_PATH --msl 2 --ne 2 --mae 1024 --mel 0 --seed 1234 --run 2 --mod_id 1&

python /gpfs/work4/0/tese0660/projects/transf_learn_attack/attack_trans.py CONT_PATH --msl 2 --ne 2 --mae 1024 --mel 0 --seed 1 --run 0 --mod_id 2&
python /gpfs/work4/0/tese0660/projects/transf_learn_attack/attack_trans.py CONT_PATH --msl 2 --ne 2 --mae 1024 --mel 0 --seed 42 --run 1 --mod_id 2&
python /gpfs/work4/0/tese0660/projects/transf_learn_attack/attack_trans.py CONT_PATH --msl 2 --ne 2 --mae 1024 --mel 0 --seed 1234 --run 2 --mod_id 2&

wait

