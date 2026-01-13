#!/bin/bash

# run_all_experiments.sh
# Runs fuzz.py for multiple seeds & batch sizes, then runs analysis/main.py for each run.
# Usage: ./run_all_experiments.sh

PY=python            # or `python3` if required
OUTDIR="results"     # directory to store any outputs (optional)
LOGDIR="logs"        # directory for logs

# fuzz parameters (change if you want different defaults)
MODEL="resnet50"
ATTACKED_MODEL="mobilevit"
DATASET="ImageNet"
SPLIT="val"
SEED_COUNT=3923
TIME_BUDGET=300
NUM_CLASSES=1000
CLEAN_SEED_COUNT=3121
COVERAGE_METRIC="NLC"

# experiment grid
SEEDS=(0)
BATCHES=(1 24)

mkdir -p "$OUTDIR"
mkdir -p "$LOGDIR"

for seed in "${SEEDS[@]}"; do
  for batch in "${BATCHES[@]}"; do
    echo "======================================================"
    echo "RUN: seed=${seed}  batch=${batch}"
    echo "------------------------------------------------------"

    FUZZ_LOG="${LOGDIR}/fuzz_seed${seed}_batch${batch}.log"
    ANALYSIS_LOG="${LOGDIR}/analysis_seed${seed}_batch${batch}.log"

    # Run fuzzing
    echo "[$(date +'%F %T')] Starting fuzz.py (seed=${seed}, batch=${batch})" | tee -a "$FUZZ_LOG"
    $PY fuzz.py \
      --model "$MODEL" \
      --dataset "$DATASET" \
      --split "$SPLIT" \
      --time-budget "$TIME_BUDGET" \
      --seed "$seed" \
      --batch-size "$batch" \
      --seed-count "$SEED_COUNT" \
      --coverage-metric "$COVERAGE_METRIC" \
      # --random-mutation \
      2>&1 | tee -a "$FUZZ_LOG"
    FUZZ_EXIT=$?
    echo "[$(date +'%F %T')] Finished fuzz.py (exit=${FUZZ_EXIT})" | tee -a "$FUZZ_LOG"

    # Optionally: check fuzz exit and decide whether to continue to analysis
    if [ $FUZZ_EXIT -ne 0 ]; then
      echo "WARNING: fuzz.py returned non-zero exit (${FUZZ_EXIT}) for seed=${seed},batch=${batch}. Continuing to analysis step." | tee -a "$FUZZ_LOG"
    fi

    # Run analysis (use same parameters so analysis can map to fuzz run)
    echo "[$(date +'%F %T')] Starting analysis/main.py (seed=${seed}, batch=${batch})" | tee -a "$ANALYSIS_LOG"
    $PY analysis/main.py \
      --seed-count "$SEED_COUNT" \
      --clean-seed-count "$CLEAN_SEED_COUNT" \
      --batch-size "$batch" \
      --model-name "$MODEL" \
      --attacked-model-name "$ATTACKED_MODEL" \
      --dataset-name "$DATASET" \
      --time-budget "$TIME_BUDGET" \
      --num-classes "$NUM_CLASSES" \
      --seed "$seed" \
      # --random-mutation \
      2>&1 | tee -a "$ANALYSIS_LOG"
    ANALYSIS_EXIT=$?
    echo "[$(date +'%F %T')] Finished analysis/main.py (exit=${ANALYSIS_EXIT})" | tee -a "$ANALYSIS_LOG"

    # Record summary line for quick scan
    echo "$(date +'%F %T') SUMMARY seed=${seed} batch=${batch} fuzz_exit=${FUZZ_EXIT} analysis_exit=${ANALYSIS_EXIT}" >> "${LOGDIR}/run_summary.log"

    echo "------------------------------------------------------"
    # small sleep to give system a short breather (optional)
    sleep 1
  done
done

echo "ALL DONE. Summary in ${LOGDIR}/run_summary.log"


# analysis
#   --attacked-model-path "$ATTACKED_MODEL_PATH" \
#       --model-path "$MODEL_PATH" \
