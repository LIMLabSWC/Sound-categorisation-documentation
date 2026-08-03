#!/bin/bash
#SBATCH --job-name=human_full_alpha_scan
#SBATCH --array=0
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=9
#SBATCH --mem=128G
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=Amirali136344@gmail.com

USERNAME="apourdehghan"
HOME_ROOT="/nfs/nhome/live/${USERNAME}"

FIT_MODE="full_alpha_scan"
MODEL_MODE="fixed_alpha"
SAMPLER="TPE"

TRIALS=200
MODEL_SEEDS=5
PARALLEL_JOBS=9

mkdir -p "/ceph/akrami/Amir/slurm_logs"

# Two example subjects: one human and one mouse.
# Change these to the exact subjects you want.
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    SPECIES="human"
    SUBJECT="6266630.0"
else
    SPECIES="mouse"
    SUBJECT="QP0100"
fi

SPECIES_CAP="$(tr '[:lower:]' '[:upper:]' <<< ${SPECIES:0:1})${SPECIES:1}"

RESULT_DIR="/ceph/akrami/Amir/Hybrid_repeated_cv_binned/${SPECIES_CAP}/results/${SAMPLER}/${FIT_MODE}/${SUBJECT}"
LOG_DIR="${RESULT_DIR}/logs"
mkdir -p "$LOG_DIR"

OUT_LOG="${LOG_DIR}/output_${SUBJECT}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
ERR_LOG="${LOG_DIR}/error_${SUBJECT}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"

echo "Starting full alpha scan"
echo "Subject=$SUBJECT"
echo "Species=$SPECIES"
echo "Fit mode=$FIT_MODE"
echo "Model mode=$MODEL_MODE"
echo "Sampler=$SAMPLER"
echo "Trials=$TRIALS"
echo "Model seeds=$MODEL_SEEDS"
echo "Parallel jobs=$PARALLEL_JOBS"
echo "Result dir=$RESULT_DIR"

python3 binned_Hybrid_run_cv.py \
    --p_id="$SUBJECT" \
    --species="$SPECIES" \
    --fit_mode="$FIT_MODE" \
    --model_mode="$MODEL_MODE" \
    --sampler="$SAMPLER" \
    --trials="$TRIALS" \
    --model_seeds="$MODEL_SEEDS" \
    --parallel_jobs="$PARALLEL_JOBS" \
    --alpha_values="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9" > "$OUT_LOG" 2> "$ERR_LOG"
