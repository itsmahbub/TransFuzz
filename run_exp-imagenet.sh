#!/bin/bash

PY=python           
LOGDIR="logs"        # directory for logs

MODEL="resnet50" # mobilevit
ATTACKED_MODEL="mobilevit" # resnet50
DATASET="ImageNet"
SPLIT="val"
SEED_COUNT=3923
TIME_BUDGET=300
NUM_CLASSES=1000
CLEAN_SEED_COUNT=3121
COVERAGE_METRIC="NLC"

# experiment grid
SEEDS=(0 1 2)
BATCHES=(1 24) # (N values)
NO_GRAD=" --random-mutation"

mkdir -p "$LOGDIR"

for seed in "${SEEDS[@]}"; do
  for batch in "${BATCHES[@]}"; do
    echo "======================================================"
    echo "RUN: seed=${seed}  batch=${batch}"
    echo "------------------------------------------------------"

    FUZZ_LOG="${LOGDIR}/fuzz_seed${seed}_batch${batch}.log"
    ANALYSIS_LOG="${LOGDIR}/analysis_seed${seed}_batch${batch}.log"

    # Run fuzzing
    echo "[$(date +'%F %T')] Starting transfuzz.py (seed=${seed}, batch=${batch})" | tee -a "$FUZZ_LOG"
    $PY transfuzz.py \
      --model "$MODEL" \
      --seed-dataset "$DATASET" \
      --split "$SPLIT" \
      --time-budget "$TIME_BUDGET" \
      --seed "$seed" \
      --N "$batch" \
      --seed-count "$SEED_COUNT" \
      --coverage-metric "$COVERAGE_METRIC" "$NO_GRAD" \
      2>&1 | tee -a "$FUZZ_LOG"
    FUZZ_EXIT=$?
    echo "[$(date +'%F %T')] Finished transfuzz.py (exit=${FUZZ_EXIT})" | tee -a "$FUZZ_LOG"

    if [ $FUZZ_EXIT -ne 0 ]; then
      echo "WARNING: transfuzz.py returned non-zero exit (${FUZZ_EXIT}) for seed=${seed},batch=${batch}. Continuing to analysis step." | tee -a "$FUZZ_LOG"
    fi

    # Run analysis 
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
      --seed "$seed" "$NO_GRAD" \
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
