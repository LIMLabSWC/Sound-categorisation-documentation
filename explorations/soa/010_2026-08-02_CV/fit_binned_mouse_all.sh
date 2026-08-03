#!/bin/bash
#SBATCH --job-name=mouse_binned_cv
#SBATCH --array=0-33
#SBATCH --time=22:00:00
#SBATCH --cpus-per-task=15
#SBATCH --mem=240G
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=Amirali136344@gmail.com
#SBATCH --output=/ceph/akrami/Amir/slurm_logs/slurm_%A_%a.out
#SBATCH --error=/ceph/akrami/Amir/slurm_logs/slurm_%A_%a.err

USERNAME="pentousil"
HOME_ROOT="/nfs/nhome/live/${USERNAME}"

SPECIES="mouse"   # change to: human, rat, or mouse
MODEL_MODE="all"
SAMPLER="TPE"

CONFIG_FILE="/ceph/akrami/Amir/hybrid_fit_${SPECIES}_config_list.txt"

SUBJECT=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CONFIG_FILE" | tr -d '\r')

SPECIES_CAP="$(tr '[:lower:]' '[:upper:]' <<< ${SPECIES:0:1})${SPECIES:1}"

RESULT_DIR="/ceph/akrami/Amir/Hybrid_repeated_cv_binned/${SPECIES_CAP}/results/${SAMPLER}/${MODEL_MODE}/${SUBJECT}"
LOG_DIR="${RESULT_DIR}/logs"
mkdir -p "$LOG_DIR"
mkdir -p "/ceph/akrami/Amir/slurm_logs"

OUT_LOG="${LOG_DIR}/output_${SUBJECT}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
ERR_LOG="${LOG_DIR}/error_${SUBJECT}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"


python3 binned_Hybrid_run_cv.py \
    --p_id="$SUBJECT" \
    --species="$SPECIES" \
    --model_mode="all" \
    --sampler="$SAMPLER" \
    --trials=200 \
    --folds=2 \
    --repetitions=5 \
    --model_seeds=5 \
    --parallel_jobs=15 > "$OUT_LOG" 2> "$ERR_LOG"
