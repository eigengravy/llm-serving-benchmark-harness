```
Usage: ./launch_vllm_sweep.sh [OPTIONS]

GPU Configuration:
    -g, --gpus NUM              Number of GPUs (default: 1)
                                Also sets tensor-parallel-size automatically
    
Scheduling:
    -a, --async BOOL            Use async-scheduling: true/false (default: true)
    
Output:
    -l, --log-suffix SUFFIX     Log directory suffix (default: default)
                                Creates: $SCRATCH/vllm_logs_<suffix>
    
SLURM:
    -t, --time LIMIT            Time limit HH:MM:SS (default: 01:00:00)
    -q, --queue QUEUE           SLURM queue (default: regular)
    
Config Grid:
    --request-rates "R1 R2..."  Space-separated request rates 
                                (default: "10 20 40 80")
    --batched-tokens "B1 B2..." Space-separated batch token limits
                                (default: "16384 32768 65536")
    --num-seqs "N1 N2..."       Space-separated max num seqs
                                (default: "2048 4096")
    --trials NUM                Trials per config (default: 5)
```

## Common Use Cases

### Single GPU Experiments

```bash
# Quick test
./launch_vllm_sweep.sh \
    -g 1 -a true -l test \
    --request-rates "10 20" \
    --batched-tokens "16384" \
    --trials 2

# Full sweep
./launch_vllm_sweep.sh \
    -g 1 -a true -l single_gpu_async \
    --request-rates "10 20 40 80 120 160 200" \
    --batched-tokens "16384 32768 65536" \
    --num-seqs "2048 4096" \
    --trials 5
```

### Multi-GPU Tensor Parallel

```bash
# 2 GPU with larger configs
./launch_vllm_sweep.sh \
    -g 2 -a true -l dual_gpu_async \
    --batched-tokens "65536 131072" \
    --num-seqs "4096 8192"

# 4 GPU with very large configs
./launch_vllm_sweep.sh \
    -g 4 -a true -l quad_gpu_large \
    --batched-tokens "131072 262144" \
    --num-seqs "8192 16384"
```

## Output Structure

```
$SCRATCH/vllm_logs_<suffix>/
├── sbatch_<suffix>_<timestamp>.sh   # Generated SBATCH script for reproducibility
└── <rate>-<tokens>-<seqs>-<trial>[-tp<N>]/
    ├── sbatch_script.sh              # Copy of SBATCH script
    ├── <exp>_bench.json              # Benchmark results
    ├── <exp>_gpu.csv                 # GPU utilization over time
    └── <exp>_metrics.txt             # Prometheus metrics
```

Example:
```
$SCRATCH/vllm_logs_dual_gpu_async/
├── sbatch_dual_gpu_async_20260208_143022.sh
└── 40-65536-4096-1-tp2/
    ├── sbatch_script.sh
    ├── 40-65536-4096-1-tp2_bench.json
    ├── 40-65536-4096-1-tp2_gpu.csv
    └── 40-65536-4096-1-tp2_metrics.txt
```
