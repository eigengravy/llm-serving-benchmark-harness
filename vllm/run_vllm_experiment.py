#!/usr/bin/env python3
"""
run_vllm_experiment.py
Unified vLLM experiment runner - fully configurable via environment variables.
Works for single-GPU or multi-GPU (tensor parallel) configurations.
"""

import os
import sys
import time
import json
import signal
import subprocess
import urllib.request


# -------------------------
# Utilities
# -------------------------
def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def get_env_int(name, default):
    v = os.getenv(name)
    return int(v) if v is not None else default


def get_env_bool(name, default):
    v = os.getenv(name, "").lower()
    if v in ("true", "1", "yes"):
        return True
    elif v in ("false", "0", "no"):
        return False
    return default


# -------------------------
# Config from env (SBATCH)
# -------------------------
MODEL = os.getenv("MODEL", "openai/gpt-oss-20b")

REQUEST_RATE = get_env_int("REQUEST_RATE", 10)
MAX_BATCHED_TOKENS = get_env_int("MAX_BATCHED_TOKENS", 16384)
MAX_NUM_SEQS = get_env_int("MAX_NUM_SEQS", 2048)
TRIAL = get_env_int("TRIAL", 1)
TENSOR_PARALLEL = get_env_int("TENSOR_PARALLEL", 1)

# Async flag comes as a string like "--async-scheduling" or empty
ASYNC_FLAG = os.getenv("ASYNC_FLAG", "").strip()

PORT = get_env_int("VLLM_PORT", 8000)

SERVER_HOST = "0.0.0.0"  # bind
CLIENT_HOST = "127.0.0.1"  # connect

LOG_ROOT = os.getenv("LOG_ROOT", "./logs")

# Experiment naming includes tensor parallel info
if TENSOR_PARALLEL > 1:
    EXP_NAME = f"{REQUEST_RATE}-{MAX_BATCHED_TOKENS}-{MAX_NUM_SEQS}-{TRIAL}-tp{TENSOR_PARALLEL}"
else:
    EXP_NAME = f"{REQUEST_RATE}-{MAX_BATCHED_TOKENS}-{MAX_NUM_SEQS}-{TRIAL}"

EXP_DIR = os.path.join(LOG_ROOT, EXP_NAME)
os.makedirs(EXP_DIR, exist_ok=True)

GPU_CSV = os.path.join(EXP_DIR, f"{EXP_NAME}_gpu.csv")
METRICS_FILE = os.path.join(EXP_DIR, f"{EXP_NAME}_metrics.txt")
BENCH_FILE = os.path.join(EXP_DIR, f"{EXP_NAME}_bench.json")
SBATCH_COPY = os.path.join(EXP_DIR, "sbatch_script.sh")

# Copy SBATCH script to experiment directory for reproducibility
SBATCH_SCRIPT_NAME = os.getenv("SBATCH_SCRIPT_NAME", "")
if SBATCH_SCRIPT_NAME and os.path.exists(SBATCH_SCRIPT_NAME):
    import shutil
    shutil.copy2(SBATCH_SCRIPT_NAME, SBATCH_COPY)

# Dataset path
DATASET_PATH = os.getenv("DATASET_PATH", "./ShareGPT_V3_unfiltered_cleaned_split.json")
NUM_PROMPTS = get_env_int("NUM_PROMPTS", 25000)
BURSTINESS = os.getenv("BURSTINESS", "0.01")

# -------------------------
# Commands
# -------------------------
VLLM_SERVE_CMD = [
    "vllm",
    "serve",
    MODEL,
    "--host",
    SERVER_HOST,
    "--port",
    str(PORT),
    "--max-num-batched-tokens",
    str(MAX_BATCHED_TOKENS),
    "--max-num-seqs",
    str(MAX_NUM_SEQS),
]

# Add async scheduling if flag is set
if ASYNC_FLAG:
    VLLM_SERVE_CMD.append(ASYNC_FLAG)

# Add tensor parallel if > 1
if TENSOR_PARALLEL > 1:
    VLLM_SERVE_CMD.extend(["--tensor-parallel-size", str(TENSOR_PARALLEL)])

VLLM_BENCH_CMD = [
    "vllm",
    "bench",
    "serve",
    "--backend",
    "vllm",
    "--base-url",
    f"http://{CLIENT_HOST}:{PORT}",
    "--model",
    MODEL,
    "--dataset-name",
    "sharegpt",
    "--dataset-path",
    DATASET_PATH,
    "--num-prompts",
    str(NUM_PROMPTS),
    "--request-rate",
    str(REQUEST_RATE),
    "--burstiness",
    BURSTINESS,
    "--save-result",
    "--result-dir",
    EXP_DIR,
    "--result-filename",
    BENCH_FILE,
]

# -------------------------
# Process handling
# -------------------------
procs = []


def cleanup():
    log("Cleaning up processes")
    for p in procs:
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()


def sig_handler(sig, frame):
    cleanup()
    sys.exit(1)


signal.signal(signal.SIGTERM, sig_handler)
signal.signal(signal.SIGINT, sig_handler)


# -------------------------
# Readiness check (CRITICAL)
# -------------------------
def wait_for_vllm_ready(port, timeout=600):
    url = f"http://{CLIENT_HOST}:{port}/v1/models"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                json.loads(r.read())
            return True
        except Exception:
            time.sleep(2)
    return False


# -------------------------
# Run
# -------------------------
log(f"Starting experiment {EXP_NAME}")
log(f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')}")
log(f"Tensor Parallel Size: {TENSOR_PARALLEL}")
log(f"Async Scheduling: {'Enabled' if ASYNC_FLAG else 'Disabled'}")
log(f"Using port {PORT}")
log(f"vLLM serve command: {' '.join(VLLM_SERVE_CMD)}")

# start vLLM server
log("Starting vLLM server")
serve_proc = subprocess.Popen(
    VLLM_SERVE_CMD,
    stdout=sys.stdout,
    stderr=sys.stderr,
)
procs.append(serve_proc)

log("Waiting for vLLM to become ready...")
if not wait_for_vllm_ready(PORT):
    log("vLLM did not become ready")
    cleanup()
    sys.exit(1)

log("vLLM server is ready!")

# start GPU logger
if subprocess.run(["which", "nvidia-smi"], capture_output=True).returncode == 0:
    log("Starting GPU logger")
    gpu_f = open(GPU_CSV, "w")
    nvsmi_proc = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.free",
            "--format=csv,noheader,nounits",
            "-l",
            "1",
        ],
        stdout=gpu_f,
    )
    procs.append(nvsmi_proc)

# run benchmark
log("Running vLLM bench")
bench_proc = subprocess.Popen(
    VLLM_BENCH_CMD,
    stdout=sys.stdout,
    stderr=sys.stderr,
)
rc = bench_proc.wait()

if rc != 0:
    log(f"vLLM bench failed with exit code {rc}")
    cleanup()
    sys.exit(1)

# fetch metrics
log("Fetching Prometheus metrics")
try:
    with urllib.request.urlopen(f"http://{CLIENT_HOST}:{PORT}/metrics", timeout=5) as r:
        with open(METRICS_FILE, "wb") as f:
            f.write(r.read())
    log(f"Metrics saved to {METRICS_FILE}")
except Exception as e:
    log(f"Metrics fetch failed: {e}")

cleanup()
time.sleep(2)
log("Experiment completed successfully")
