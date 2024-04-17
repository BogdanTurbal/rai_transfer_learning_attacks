#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH -t 14:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=bogdan.turbal.y@gmail.com
#SBATCH -o /gpfs/work4/0/tese0660/projects/transf_learn_attack/v8/gin_imp_%j.out

# Load modules
module purge
module load 2023

# Activate conda environment
source ~/.bashrc
conda activate att

# Base directory
BASE_DIR="/gpfs/work4/0/tese0660/projects/transf_learn_attack/v8"
CODE_DIR="/gpfs/work4/0/tese0660/projects/transf_learn_attack/rai_transfer_learning_attacks"
#SAVE_DIR="/gpfs/work4/0/tese0660/projects/transf_learn_attack"
#cd $BASE_DIR

# Define a list of specific seeds
declare -a seeds=(1 42 1234 1 42 1234 1 42 1234)

# Loop through the list of seeds
for i in {6..7}; do
    # Directory creation
    mkdir -p $BASE_DIR/${i}_id
    
    # Use the seed from the seeds array
    seed=${seeds[$i]}
    # Determine mod_id based on the iteration
    let "mod_id = $i / 3"
    let "run = $i % 3"

    OUT_FILE=$BASE_DIR/gin_imp_${i}_%j.out
    
    # Launch the Python script with specified parameters
    python $CODE_DIR/attack_trans.py $BASE_DIR/${i}_id/ $BASE_DIR/ --msl 2 --ne 3 --mae 2000 --mel 0 --seed $seed --run $run --mod_id $mod_id --load_best 1 > $OUT_FILE 2>&1 &
done

# Wait for all background jobs to finish
wait
