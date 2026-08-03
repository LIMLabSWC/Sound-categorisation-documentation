#!/bin/bash
#SBATCH --job-name=total_human_binned_cv
#SBATCH --array=0-28
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=15
#SBATCH --mem=128G
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=Amirali136344@gmail.com


USERNAME="apourdehghan"
HOME_ROOT="/nfs/nhome/live/${USERNAME}"

SPECIES="human"
FIT_MODE="cv"
MODEL_MODE="all"
SAMPLER="TPE"

TRIALS=200
FOLDS=2
REPETITIONS=5
MODEL_SEEDS=5
PARALLEL_JOBS=15

CONFIG_FILE="/ceph/akrami/Amir/hybrid_fit_${SPECIES}_config_list.txt"

SUBJECT=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CONFIG_FILE" | tr -d '\r')

SPECIES_CAP="$(tr '[:lower:]' '[:upper:]' <<< ${SPECIES:0:1})${SPECIES:1}"

RESULT_DIR="/ceph/akrami/Amir/Hybrid_repeated_cv_binned/${SPECIES_CAP}/results/${SAMPLER}/${FIT_MODE}/${MODEL_MODE}/${SUBJECT}"
LOG_DIR="${RESULT_DIR}/logs"

mkdir -p "$LOG_DIR"
mkdir -p "/ceph/akrami/Amir/slurm_logs"

OUT_LOG="${LOG_DIR}/output_${SUBJECT}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
ERR_LOG="${LOG_DIR}/error_${SUBJECT}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"

EXTRA_ARGS=""

if [ "$FIT_MODE" = "full_alpha_scan" ]; then
    EXTRA_ARGS="--alpha_values=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"
fi

python3 "${HOME_ROOT}/binned_Hybrid_run_cv.py" \
    --p_id="$SUBJECT" \
    --species="$SPECIES" \
    --fit_mode="$FIT_MODE" \
    --model_mode="$MODEL_MODE" \
    --sampler="$SAMPLER" \
    --trials="$TRIALS" \
    --folds="$FOLDS" \
    --repetitions="$REPETITIONS" \
    --model_seeds="$MODEL_SEEDS" \
    --parallel_jobs="$PARALLEL_JOBS" \
    $EXTRA_ARGS > "$OUT_LOG" 2> "$ERR_LOG"
