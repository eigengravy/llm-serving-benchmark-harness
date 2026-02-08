#!/bin/bash
# launch_vllm_sweep.sh
# Unified launcher for vLLM experiments with full configuration via arguments

set -euo pipefail

# -------------------------
# Default values
# -------------------------
NUM_GPUS=1
ASYNC_SCHEDULING="true"
LOG_ROOT_SUFFIX="default"
TIME_LIMIT="01:00:00"
QUEUE="regular"

# Config grid defaults
REQUEST_RATES="10 20 40 80"
BATCHED_TOKENS="16384 32768 65536"
NUM_SEQS="2048 4096"
TRIALS=5

# -------------------------
# Usage
# -------------------------
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    -g, --gpus NUM              Number of GPUs (default: 1)
    -a, --async BOOL            Use async-scheduling: true/false (default: true)
    -l, --log-suffix SUFFIX     Log root suffix (default: default)
    -t, --time LIMIT            Time limit HH:MM:SS (default: 01:00:00)
    -q, --queue QUEUE           SLURM queue (default: regular)
    
    Config grid options:
    --request-rates "R1 R2 ..." Request rates (default: "10 20 40 80")
    --batched-tokens "B1 B2..." Batched token limits (default: "16384 32768 65536")
    --num-seqs "N1 N2 ..."      Max num seqs (default: "2048 4096")
    --trials NUM                Number of trials per config (default: 5)
    
    -h, --help                  Show this help

Examples:
    # Single GPU with async
    $0 -g 1 -a true -l single_gpu_async
    
    # 2 GPU tensor parallel without async
    $0 -g 2 -a false -l dual_gpu_sync --batched-tokens "65536 131072"
    
    # Quick test with fewer configs
    $0 -g 1 --request-rates "10 20" --trials 2 -l test
EOF
    exit 1
}

# -------------------------
# Parse arguments
# -------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        -g|--gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        -a|--async)
            ASYNC_SCHEDULING="$2"
            shift 2
            ;;
        -l|--log-suffix)
            LOG_ROOT_SUFFIX="$2"
            shift 2
            ;;
        -t|--time)
            TIME_LIMIT="$2"
            shift 2
            ;;
        -q|--queue)
            QUEUE="$2"
            shift 2
            ;;
        --request-rates)
            REQUEST_RATES="$2"
            shift 2
            ;;
        --batched-tokens)
            BATCHED_TOKENS="$2"
            shift 2
            ;;
        --num-seqs)
            NUM_SEQS="$2"
            shift 2
            ;;
        --trials)
            TRIALS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# -------------------------
# Derive config
# -------------------------
TENSOR_PARALLEL=$NUM_GPUS
LOG_ROOT="\$SCRATCH/vllm_logs_${LOG_ROOT_SUFFIX}"

if [[ "$ASYNC_SCHEDULING" == "true" ]]; then
    ASYNC_FLAG="--async-scheduling"
else
    ASYNC_FLAG=""
fi

# -------------------------
# Generate SBATCH script
# -------------------------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SBATCH_SCRIPT="sbatch_${LOG_ROOT_SUFFIX}_${TIMESTAMP}.sh"

cat > "$SBATCH_SCRIPT" << 'SBATCH_EOF'
#!/bin/bash
#SBATCH -A m5083_g
#SBATCH -C "gpu&hbm80g"
#SBATCH -q __QUEUE__
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=__NUM_GPUS__
#SBATCH --cpus-per-task=16
#SBATCH -t __TIME_LIMIT__
#SBATCH -J vllm
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err

set -euo pipefail

# -------------------------
# Environment
# -------------------------
source ~/env.sh
source $SCRATCH/venvs/vllm-env/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_DEVICE_ORDER=PCI_BUS_ID

mkdir -p logs

echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

# -------------------------
# Config grid (from launcher)
# -------------------------
REQUEST_RATES=(__REQUEST_RATES__)
BATCHED_TOKENS=(__BATCHED_TOKENS__)
NUM_SEQS=(__NUM_SEQS__)
TRIALS=__TRIALS__

CONFIGS=()
for r in "${REQUEST_RATES[@]}"; do
  for b in "${BATCHED_TOKENS[@]}"; do
    for n in "${NUM_SEQS[@]}"; do
      for t in $(seq 1 $TRIALS); do
        CONFIGS+=("$r $b $n $t")
      done
    done
  done
done

CFG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
read REQUEST_RATE MAX_BATCHED_TOKENS MAX_NUM_SEQS TRIAL <<< "$CFG"

export REQUEST_RATE
export MAX_BATCHED_TOKENS
export MAX_NUM_SEQS
export TRIAL
export TENSOR_PARALLEL=__TENSOR_PARALLEL__
export ASYNC_FLAG="__ASYNC_FLAG__"
export VLLM_PORT=$((8000 + SLURM_ARRAY_TASK_ID % 1000))
export LOG_ROOT="__LOG_ROOT__"
export SBATCH_SCRIPT_NAME="__SBATCH_SCRIPT_NAME__"

echo "Running config: $CFG"
echo "Tensor Parallel: $TENSOR_PARALLEL"
echo "Async: __ASYNC_SCHEDULING__"
echo "Port: $VLLM_PORT"

# -------------------------
# Run
# -------------------------
python run_vllm_experiment.py
SBATCH_EOF

# Replace placeholders
sed -i "s|__QUEUE__|$QUEUE|g" "$SBATCH_SCRIPT"
sed -i "s|__NUM_GPUS__|$NUM_GPUS|g" "$SBATCH_SCRIPT"
sed -i "s|__TIME_LIMIT__|$TIME_LIMIT|g" "$SBATCH_SCRIPT"
sed -i "s|__REQUEST_RATES__|$REQUEST_RATES|g" "$SBATCH_SCRIPT"
sed -i "s|__BATCHED_TOKENS__|$BATCHED_TOKENS|g" "$SBATCH_SCRIPT"
sed -i "s|__NUM_SEQS__|$NUM_SEQS|g" "$SBATCH_SCRIPT"
sed -i "s|__TRIALS__|$TRIALS|g" "$SBATCH_SCRIPT"
sed -i "s|__TENSOR_PARALLEL__|$TENSOR_PARALLEL|g" "$SBATCH_SCRIPT"
sed -i "s|__ASYNC_FLAG__|$ASYNC_FLAG|g" "$SBATCH_SCRIPT"
sed -i "s|__LOG_ROOT__|$LOG_ROOT|g" "$SBATCH_SCRIPT"
sed -i "s|__ASYNC_SCHEDULING__|$ASYNC_SCHEDULING|g" "$SBATCH_SCRIPT"
sed -i "s|__SBATCH_SCRIPT_NAME__|$SBATCH_SCRIPT|g" "$SBATCH_SCRIPT"

# -------------------------
# Calculate array size
# -------------------------
IFS=' ' read -ra RR_ARR <<< "$REQUEST_RATES"
IFS=' ' read -ra BT_ARR <<< "$BATCHED_TOKENS"
IFS=' ' read -ra NS_ARR <<< "$NUM_SEQS"

TOTAL_CONFIGS=$((${#RR_ARR[@]} * ${#BT_ARR[@]} * ${#NS_ARR[@]} * TRIALS))
ARRAY_MAX=$((TOTAL_CONFIGS - 1))

# -------------------------
# Display and submit
# -------------------------
echo "======================================"
echo "vLLM Experiment Launcher"
echo "======================================"
echo "GPUs: $NUM_GPUS (tensor-parallel-size=$TENSOR_PARALLEL)"
echo "Async scheduling: $ASYNC_SCHEDULING"
echo "Log root: $LOG_ROOT"
echo "Time limit: $TIME_LIMIT"
echo "Queue: $QUEUE"
echo ""
echo "Config grid:"
echo "  Request rates: $REQUEST_RATES"
echo "  Batched tokens: $BATCHED_TOKENS"
echo "  Num seqs: $NUM_SEQS"
echo "  Trials: $TRIALS"
echo ""
echo "Total configs: $TOTAL_CONFIGS"
echo "Array range: 0-$ARRAY_MAX"
echo "======================================"
echo ""
echo "Generated SBATCH script: $SBATCH_SCRIPT"
echo ""
read -p "Submit job? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Expand LOG_ROOT for local use
    LOG_ROOT_EXPANDED="${LOG_ROOT//\$SCRATCH/$SCRATCH}"
    
    # Create results directory and save SBATCH script
    mkdir -p "$LOG_ROOT_EXPANDED"
    cp "$SBATCH_SCRIPT" "$LOG_ROOT_EXPANDED/"
    
    # Submit job
    sbatch --array=0-$ARRAY_MAX "$SBATCH_SCRIPT"
    
    echo ""
    echo "Job submitted!"
    echo "SBATCH script: $SBATCH_SCRIPT"
    echo "Saved to: $LOG_ROOT_EXPANDED/$SBATCH_SCRIPT"
    echo ""
    echo "Monitor with: squeue -u \$USER"
    echo "View logs: ls -lh logs/"
    echo "Results: ls -lh $LOG_ROOT_EXPANDED/"
else
    echo "Submission cancelled."
    echo "You can manually submit with: sbatch --array=0-$ARRAY_MAX $SBATCH_SCRIPT"
    echo "SBATCH script saved: $SBATCH_SCRIPT"
fi
