#!/bin/bash
#SBATCH --job-name=hybrid_cv_mouse_L
#SBATCH --array=0-127
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=Amirali136344@gmail.com

# 1. Read from your NEW text file name
CONFIG_LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" /nfs/nhome/live/apourdehghan/hybrid_mice_config_list.txt)
SUBJECT=$(echo $CONFIG_LINE | cut -d',' -f1)
SEED=$(echo $CONFIG_LINE | cut -d',' -f2)
SAMPLER=$(echo $CONFIG_LINE | cut -d',' -f3 | tr -d '\r')

# 2. Match the NEW folder structure from your Python script
LOG_DIR="/nfs/nhome/live/apourdehghan/Hybrid_CV/Mouse/Asym_left/results/${SAMPLER}/${SUBJECT}/logs"
mkdir -p "$LOG_DIR"

# 3. Define log filenames
OUT_LOG="${LOG_DIR}/output_Hybrid_${SAMPLER}_seed${SEED}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
ERR_LOG="${LOG_DIR}/error_Hybrid_${SAMPLER}_seed${SEED}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"

# Print for general SLURM output
echo "Starting Job: Subject=$SUBJECT | Seed=$SEED | Sampler=$SAMPLER"

# 4. Run the Python script
python3 Hybrid_run_cv.py \
    --p_id=$SUBJECT \
    --seed=$SEED \
    --sampler=$SAMPLER > "$OUT_LOG" 2> "$ERR_LOG"
