#!/usr/bin/env python3
"""
==========================================================================
 SentinelEdge Test Suite - Full Dataset Evaluation (CPU / Laptop)
 
 Runs the fine-tuned GGUF model against every entry (or a stratified
 sample) of the SentinelEdge dataset and produces a comprehensive
 evaluation report with:
 
   - Per-label accuracy (SAFE / SUSPICIOUS / THREAT)
   - Full confusion matrix with recall/precision/F1
   - Confidence calibration (reliability diagram buckets)
   - Latency distribution (p50/p95/p99)
   - Source-stratified accuracy (physics vs robust)
   - Register-level accuracy (which command types fail most)
   - Per-failure mode analysis with raw outputs
   - JSON export of all predictions for further analysis
   - CSV export of failures only
 
 Designed for LAPTOP CPU execution:
   - Resumes automatically on crash/interrupt (checkpoint file)
   - Parallel workers via multiprocessing (configurable)
   - Stratified sampling mode for quick runs (--sample N)
   - Progress bar with ETA
   - Graceful Ctrl+C handling (saves partial results)
 
 Prerequisites:
   pip install llama-cpp-python tqdm
 
 Usage:
   # Full 1830-entry evaluation (Q4_K_M on RPi5 ~4 hours, on laptop ~2 hours)
   python 06_test_suite.py \\
     --model sentineledge-gemma2-2b-q5_k_m.gguf \\
     --dataset sentineledge_dataset.json
   
   # Quick stratified sample (200 entries, ~15 min on laptop)
   python 06_test_suite.py \\
     --model sentineledge-gemma2-2b-q5_k_m.gguf \\
     --dataset sentineledge_dataset.json \\
     --sample 200
   
   # Multi-worker for speed (use physical core count)
   python 06_test_suite.py \\
     --model sentineledge-gemma2-2b-q5_k_m.gguf \\
     --dataset sentineledge_dataset.json \\
     --workers 4
   
   # Resume from a previous crash
   python 06_test_suite.py \\
     --model sentineledge-gemma2-2b-q5_k_m.gguf \\
     --dataset sentineledge_dataset.json \\
     --resume
==========================================================================
"""

import argparse
import json
import os
import re
import sys
import time
import signal
import platform
import multiprocessing as mp
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


# ============================================================
# SECTION 1: DATA STRUCTURES
# ============================================================

@dataclass
class PredictionRecord:
    """Everything we record about a single test case."""
    entry_id: str
    expected_label: str
    predicted_label: str
    expected_confidence: float
    predicted_confidence: float
    expected_reasoning: str
    predicted_reasoning: str
    is_correct: bool
    source: str
    register: str
    function_code: str
    source_ip: str
    latency_ms: float
    prompt_tokens: int
    generated_tokens: int
    raw_output: str
    parse_method: str


@dataclass
class EvalReport:
    """Aggregate metrics across all predictions."""
    model_name: str
    dataset_name: str
    total_entries: int
    total_evaluated: int
    elapsed_seconds: float
    
    # Accuracy
    overall_accuracy: float = 0.0
    per_label_accuracy: Dict[str, float] = field(default_factory=dict)
    per_source_accuracy: Dict[str, float] = field(default_factory=dict)
    
    # Confusion matrix (rows = expected, cols = predicted)
    confusion_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    # Per-class metrics
    per_label_precision: Dict[str, float] = field(default_factory=dict)
    per_label_recall: Dict[str, float] = field(default_factory=dict)
    per_label_f1: Dict[str, float] = field(default_factory=dict)
    macro_f1: float = 0.0
    
    # Confidence calibration (10 buckets)
    calibration_buckets: List[Dict[str, Any]] = field(default_factory=list)
    expected_calibration_error: float = 0.0
    
    # Latency
    latency_mean_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    tokens_per_sec_mean: float = 0.0
    
    # Failures
    failure_count: int = 0
    parse_error_count: int = 0
    per_register_accuracy: Dict[str, Dict[str, Any]] = field(default_factory=dict)


# ============================================================
# SECTION 2: GGUF WORKER (runs in subprocess)
# ============================================================

# Global holder inside a worker process - llama-cpp-python can't be pickled
_WORKER_LLM = None
_WORKER_CONFIG = None


def worker_init(model_path: str, n_ctx: int, n_threads: int,
                use_mmap: bool, use_mlock: bool):
    """Initialize llama.cpp model once per worker process."""
    global _WORKER_LLM, _WORKER_CONFIG
    from llama_cpp import Llama
    _WORKER_LLM = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=0,
        use_mmap=use_mmap,
        use_mlock=use_mlock,
        verbose=False,
    )
    _WORKER_CONFIG = {
        "model_path": model_path,
        "n_ctx": n_ctx,
        "n_threads": n_threads,
    }


def preflight_context_check(sample_user_content: str, n_ctx: int,
                            max_new_tokens: int = 200) -> Tuple[bool, str]:
    """
    Tokenize a real SentinelEdge prompt and verify it fits in n_ctx with
    headroom for generation. Returns (is_ok, message).
    
    Must be called AFTER worker_init() because it uses _WORKER_LLM.
    """
    global _WORKER_LLM
    if _WORKER_LLM is None:
        return False, "Worker not initialized"
    
    full_prompt = GEMMA2_PROMPT.format(user=sample_user_content)
    
    # llama.cpp tokenizer is exposed via tokenize() taking bytes
    try:
        tokens = _WORKER_LLM.tokenize(full_prompt.encode("utf-8"))
    except Exception as e:
        return False, f"Tokenization failed: {e}"
    
    prompt_tokens = len(tokens)
    required = prompt_tokens + max_new_tokens
    
    if required > n_ctx:
        needed_ctx = 2 ** (required - 1).bit_length()  # next power of 2
        needed_ctx = max(needed_ctx, 2048)
        msg = (
            f"Context window too small!\n"
            f"   Prompt tokens:      {prompt_tokens}\n"
            f"   Max new tokens:     {max_new_tokens}\n"
            f"   Required (total):   {required}\n"
            f"   Current --ctx-size: {n_ctx}\n"
            f"   FIX: re-run with --ctx-size {needed_ctx}"
        )
        return False, msg
    
    return True, (
        f"Context check passed: {prompt_tokens} prompt + {max_new_tokens} new "
        f"= {required} tokens fit in {n_ctx} context"
    )


GEMMA2_PROMPT = "<bos><start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n"


def parse_response(raw: str) -> Tuple[str, float, str, str]:
    """
    Parse a plain-text SentinelEdge response.
    Returns (category, confidence, reasoning, parse_method).
    """
    category = "UNKNOWN"
    confidence = 0.0
    reasoning = ""
    
    raw = raw.strip()
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    
    # Method 1: Line-based
    for line in lines:
        up = line.upper()
        if up.startswith("CATEGORY:"):
            cat = line.split(":", 1)[1].strip().upper()
            cat = "".join(c for c in cat if c.isalpha())
            if cat in ("SAFE", "SUSPICIOUS", "THREAT"):
                category = cat
        elif up.startswith("CONFIDENCE:"):
            m = re.search(r'([\d.]+)', line.split(":", 1)[1])
            if m:
                try:
                    confidence = round(max(0.0, min(1.0, float(m.group(1)))), 2)
                except ValueError:
                    pass
        elif up.startswith("REASONING:"):
            reasoning = line.split(":", 1)[1].strip()
    
    if category != "UNKNOWN":
        return category, confidence, reasoning, "line_based"
    
    # Method 2: regex fallback
    cat_m = re.search(r'CATEGORY[:\s]+([A-Z]+)', raw, re.IGNORECASE)
    if cat_m:
        cat = cat_m.group(1).upper()
        if cat in ("SAFE", "SUSPICIOUS", "THREAT"):
            category = cat
    conf_m = re.search(r'CONFIDENCE[:\s]+([\d.]+)', raw, re.IGNORECASE)
    if conf_m:
        try:
            confidence = round(max(0.0, min(1.0, float(conf_m.group(1)))), 2)
        except ValueError:
            pass
    reason_m = re.search(r'REASONING[:\s]+(.+?)(?:\n\n|\Z)', raw,
                         re.IGNORECASE | re.DOTALL)
    if reason_m:
        reasoning = reason_m.group(1).strip()
    
    return category, confidence, reasoning, ("regex" if category != "UNKNOWN" else "failed")


def worker_predict(task: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single entry through the model. Returns a dict (pickle-safe)."""
    global _WORKER_LLM
    
    user_content = task["user_content"]
    prompt = GEMMA2_PROMPT.format(user=user_content)
    
    start = time.time()
    output = _WORKER_LLM(
        prompt,
        max_tokens=200,
        temperature=0.15,
        top_p=0.90,
        top_k=40,
        repeat_penalty=1.18,
        stop=["<end_of_turn>", "<eos>"],
    )
    latency_ms = (time.time() - start) * 1000
    
    raw = output["choices"][0]["text"].strip()
    category, confidence, reasoning, method = parse_response(raw)
    
    return {
        "entry_id": task["entry_id"],
        "expected_label": task["expected_label"],
        "expected_confidence": task["expected_confidence"],
        "expected_reasoning": task["expected_reasoning"],
        "source": task["source"],
        "register": task["register"],
        "function_code": task["function_code"],
        "source_ip": task["source_ip"],
        "predicted_label": category,
        "predicted_confidence": confidence,
        "predicted_reasoning": reasoning,
        "is_correct": category == task["expected_label"],
        "latency_ms": round(latency_ms, 1),
        "prompt_tokens": output["usage"]["prompt_tokens"],
        "generated_tokens": output["usage"]["completion_tokens"],
        "raw_output": raw[:500],
        "parse_method": method,
    }


# ============================================================
# SECTION 3: PLATFORM DETECTION (for worker defaults)
# ============================================================

def detect_platform_defaults() -> Dict[str, Any]:
    """Figure out sensible defaults for this CPU."""
    arch = platform.machine().lower()
    system = platform.system()
    
    physical_cores = os.cpu_count() or 4
    ram_gb = 8.0
    cpu_model = "Unknown"
    is_rpi = False
    
    try:
        if system == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        cpu_model = line.split(":")[1].strip()
                        break
            with open("/proc/meminfo") as f:
                for line in f:
                    if "MemTotal" in line:
                        ram_gb = int(line.split()[1]) / 1024 / 1024
                        break
            # Physical cores
            import subprocess
            r = subprocess.run(["lscpu"], capture_output=True, text=True)
            cps = sockets = 1
            for line in r.stdout.split("\n"):
                if "Core(s) per socket" in line:
                    cps = int(line.split(":")[1].strip())
                if "Socket(s)" in line:
                    sockets = int(line.split(":")[1].strip())
            physical_cores = cps * sockets
            # RPi detection
            if os.path.exists("/proc/device-tree/model"):
                with open("/proc/device-tree/model") as f:
                    if "Raspberry Pi" in f.read():
                        is_rpi = True
        elif system == "Darwin":
            import subprocess
            r = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                             capture_output=True, text=True)
            cpu_model = r.stdout.strip()
            r = subprocess.run(["sysctl", "-n", "hw.memsize"],
                             capture_output=True, text=True)
            ram_gb = int(r.stdout.strip()) / 1024**3
    except Exception:
        pass
    
    # Worker strategy:
    #  - 1 worker using all physical cores: fastest single-stream
    #  - 2 workers each using half cores: better throughput for batches
    # For test suites, we prefer 1 worker + all threads by default.
    # Memory budget: each worker loads ~2 GB of model, so on 8 GB RPi only 1 worker.
    #
    # ctx_size MUST be >= 2048: SentinelEdge prompts are ~800-850 tokens
    # and generation adds up to 200 more, so we need ~1050 for a single turn
    # plus KV cache headroom. 1024 is NOT enough. Gemma 2 2B with GQA-4 at
    # 2048 ctx only uses ~218 MB for KV cache, fits even on RPi5.
    if is_rpi or ram_gb < 10:
        default_workers = 1
        default_threads = min(physical_cores, 4)
        default_ctx = 2048
    else:
        default_workers = 1  # single-stream still best for most laptop sizes
        default_threads = physical_cores
        default_ctx = 2048
    
    return {
        "arch": arch,
        "cpu_model": cpu_model,
        "ram_gb": ram_gb,
        "physical_cores": physical_cores,
        "is_rpi": is_rpi,
        "default_workers": default_workers,
        "default_threads": default_threads,
        "default_ctx": default_ctx,
    }


# ============================================================
# SECTION 4: DATASET LOADER
# ============================================================

def load_dataset(path: str) -> Dict[str, Any]:
    """Load and validate the SentinelEdge dataset JSON."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    
    size = os.path.getsize(path)
    if size == 0:
        raise RuntimeError(f"Dataset file is empty: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if "entries" not in data:
        raise RuntimeError(f"Dataset missing 'entries' key: {path}")
    
    return data


def stratified_sample(entries: List[Dict], n: int,
                      seed: int = 42) -> List[Dict]:
    """Sample n entries preserving label and source distribution."""
    import random
    rng = random.Random(seed)
    
    # Bucket by (label, source)
    buckets = defaultdict(list)
    for e in entries:
        key = (e["label"], e["source"])
        buckets[key].append(e)
    
    total = len(entries)
    sampled = []
    for key, bucket in buckets.items():
        share = len(bucket) / total
        take = max(1, round(n * share))
        take = min(take, len(bucket))
        rng.shuffle(bucket)
        sampled.extend(bucket[:take])
    
    # Trim or pad if off-by-a-few due to rounding
    rng.shuffle(sampled)
    return sampled[:n] if len(sampled) > n else sampled


def build_tasks(entries: List[Dict]) -> List[Dict[str, Any]]:
    """Convert dataset entries into pickle-safe worker tasks."""
    tasks = []
    for e in entries:
        cmd = e["metadata"]["modbus_command"]
        tasks.append({
            "entry_id": e["id"],
            "user_content": e["messages"][0]["content"],
            "expected_label": e["label"],
            "expected_confidence": e["metadata"]["confidence"],
            "expected_reasoning": e["metadata"]["reasoning"],
            "source": e["source"],
            "register": cmd.get("reg", "unknown"),
            "function_code": cmd.get("fc", "unknown"),
            "source_ip": cmd.get("ip", "unknown"),
        })
    return tasks


# ============================================================
# SECTION 5: CHECKPOINT / RESUME
# ============================================================

def checkpoint_path(model_name: str, output_dir: str) -> str:
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', model_name)
    return os.path.join(output_dir, f".checkpoint_{safe_name}.jsonl")


def load_checkpoint(path: str) -> List[Dict]:
    """Resume from a checkpoint file (JSONL, one prediction per line)."""
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def append_checkpoint(path: str, record: Dict):
    """Append a single prediction to the checkpoint file (atomic per-line)."""
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


# ============================================================
# SECTION 6: METRICS COMPUTATION
# ============================================================

LABELS = ["SAFE", "SUSPICIOUS", "THREAT"]


def compute_report(records: List[Dict], model_name: str,
                   dataset_name: str, total_entries: int,
                   elapsed: float) -> EvalReport:
    """Compute all evaluation metrics from a list of prediction records."""
    report = EvalReport(
        model_name=model_name,
        dataset_name=dataset_name,
        total_entries=total_entries,
        total_evaluated=len(records),
        elapsed_seconds=round(elapsed, 1),
    )
    
    if not records:
        return report
    
    # Overall accuracy
    correct = sum(1 for r in records if r["is_correct"])
    report.overall_accuracy = round(correct / len(records), 4)
    
    # Per-label accuracy
    by_label = defaultdict(list)
    for r in records:
        by_label[r["expected_label"]].append(r)
    for label in LABELS:
        bucket = by_label.get(label, [])
        if bucket:
            c = sum(1 for r in bucket if r["is_correct"])
            report.per_label_accuracy[label] = round(c / len(bucket), 4)
        else:
            report.per_label_accuracy[label] = 0.0
    
    # Per-source accuracy
    by_source = defaultdict(list)
    for r in records:
        by_source[r["source"]].append(r)
    for src, bucket in by_source.items():
        if bucket:
            c = sum(1 for r in bucket if r["is_correct"])
            report.per_source_accuracy[src] = round(c / len(bucket), 4)
    
    # Confusion matrix
    matrix = {exp: {pred: 0 for pred in LABELS + ["UNKNOWN"]} for exp in LABELS}
    for r in records:
        exp = r["expected_label"]
        pred = r["predicted_label"] if r["predicted_label"] in LABELS else "UNKNOWN"
        if exp in matrix:
            matrix[exp][pred] += 1
    report.confusion_matrix = matrix
    
    # Precision / recall / F1 per label
    for label in LABELS:
        tp = matrix[label][label]
        fn = sum(matrix[label][p] for p in matrix[label] if p != label)
        fp = sum(matrix[exp][label] for exp in LABELS if exp != label)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        report.per_label_precision[label] = round(precision, 4)
        report.per_label_recall[label] = round(recall, 4)
        report.per_label_f1[label] = round(f1, 4)
    
    report.macro_f1 = round(
        sum(report.per_label_f1.values()) / len(LABELS), 4
    )
    
    # Confidence calibration (10 buckets of 0.1 width)
    buckets = [{"low": i/10, "high": (i+1)/10, "count": 0, "correct": 0,
                "avg_confidence": 0.0, "accuracy": 0.0, "gap": 0.0}
               for i in range(10)]
    for r in records:
        conf = r["predicted_confidence"]
        idx = min(int(conf * 10), 9)
        buckets[idx]["count"] += 1
        buckets[idx]["correct"] += int(r["is_correct"])
        buckets[idx]["avg_confidence"] += conf
    
    # Finalize buckets
    ece_numer = 0.0
    n = len(records)
    for b in buckets:
        if b["count"] > 0:
            b["avg_confidence"] = round(b["avg_confidence"] / b["count"], 4)
            b["accuracy"] = round(b["correct"] / b["count"], 4)
            b["gap"] = round(b["accuracy"] - b["avg_confidence"], 4)
            ece_numer += (b["count"] / n) * abs(b["accuracy"] - b["avg_confidence"])
    
    report.calibration_buckets = buckets
    report.expected_calibration_error = round(ece_numer, 4)
    
    # Latency stats
    latencies = sorted(r["latency_ms"] for r in records)
    report.latency_mean_ms = round(sum(latencies) / len(latencies), 1)
    report.latency_p50_ms = latencies[len(latencies) // 2]
    report.latency_p95_ms = latencies[int(len(latencies) * 0.95)]
    report.latency_p99_ms = latencies[min(int(len(latencies) * 0.99), len(latencies) - 1)]
    
    # Tokens per second
    total_gen_tokens = sum(r["generated_tokens"] for r in records)
    total_gen_time = sum(r["latency_ms"] for r in records) / 1000
    report.tokens_per_sec_mean = round(
        total_gen_tokens / total_gen_time if total_gen_time > 0 else 0, 1
    )
    
    # Failures
    report.failure_count = sum(1 for r in records if not r["is_correct"])
    report.parse_error_count = sum(1 for r in records if r["parse_method"] == "failed")
    
    # Per-register accuracy
    by_reg = defaultdict(list)
    for r in records:
        by_reg[r["register"]].append(r)
    for reg, bucket in by_reg.items():
        if len(bucket) >= 3:  # Only report registers with 3+ samples
            c = sum(1 for r in bucket if r["is_correct"])
            report.per_register_accuracy[reg] = {
                "total": len(bucket),
                "correct": c,
                "accuracy": round(c / len(bucket), 4),
            }
    
    return report


# ============================================================
# SECTION 7: REPORT FORMATTING
# ============================================================

def print_report(report: EvalReport, records: List[Dict]):
    """Human-readable console report."""
    w = 72
    hr = "=" * w
    
    print(f"\n{hr}")
    print(f" SENTINELEDGE EVALUATION REPORT")
    print(f"{hr}")
    print(f" Model:              {report.model_name}")
    print(f" Dataset:            {report.dataset_name}")
    print(f" Total in dataset:   {report.total_entries}")
    print(f" Evaluated:          {report.total_evaluated}")
    print(f" Duration:           {report.elapsed_seconds / 60:.1f} min "
          f"({report.elapsed_seconds:.0f}s)")
    print(f"{hr}")
    
    # --- Overall accuracy ---
    print(f"\n OVERALL ACCURACY")
    print(f" {'-' * 30}")
    print(f"   {report.overall_accuracy * 100:.2f}% "
          f"({report.total_evaluated - report.failure_count}/{report.total_evaluated})")
    print(f"   Macro F1:         {report.macro_f1:.4f}")
    print(f"   Parse errors:     {report.parse_error_count}")
    
    # --- Per-label ---
    print(f"\n PER-LABEL METRICS")
    print(f" {'-' * 60}")
    print(f"   {'Label':<12} {'Accuracy':>10} {'Precision':>11} {'Recall':>9} {'F1':>8}")
    for label in LABELS:
        acc = report.per_label_accuracy.get(label, 0)
        prec = report.per_label_precision.get(label, 0)
        rec = report.per_label_recall.get(label, 0)
        f1 = report.per_label_f1.get(label, 0)
        print(f"   {label:<12} {acc*100:>9.2f}% {prec:>11.4f} {rec:>9.4f} {f1:>8.4f}")
    
    # --- Per-source ---
    if report.per_source_accuracy:
        print(f"\n PER-SOURCE ACCURACY")
        print(f" {'-' * 30}")
        for src, acc in sorted(report.per_source_accuracy.items()):
            print(f"   {src:<12}: {acc * 100:.2f}%")
    
    # --- Confusion matrix ---
    print(f"\n CONFUSION MATRIX (rows=expected, cols=predicted)")
    print(f" {'-' * 60}")
    header = f"   {'':12}" + "".join(f"{p:>12}" for p in LABELS + ["UNKNOWN"])
    print(header)
    for exp in LABELS:
        row_vals = report.confusion_matrix.get(exp, {})
        row = f"   {exp:<12}" + "".join(
            f"{row_vals.get(p, 0):>12}" for p in LABELS + ["UNKNOWN"]
        )
        print(row)
    
    # --- Latency ---
    print(f"\n LATENCY (CPU inference)")
    print(f" {'-' * 30}")
    print(f"   Mean:              {report.latency_mean_ms:.0f} ms")
    print(f"   p50 (median):      {report.latency_p50_ms:.0f} ms")
    print(f"   p95:               {report.latency_p95_ms:.0f} ms")
    print(f"   p99:               {report.latency_p99_ms:.0f} ms")
    print(f"   Tokens/second:     {report.tokens_per_sec_mean:.1f}")
    
    # --- Calibration ---
    print(f"\n CONFIDENCE CALIBRATION")
    print(f" {'-' * 60}")
    print(f"   Expected Calibration Error (ECE): {report.expected_calibration_error:.4f}")
    print(f"   (lower is better; 0.05 is well-calibrated, 0.15+ is poor)")
    print(f"\n   {'Conf range':<12} {'Count':>8} {'AvgConf':>10} {'Accuracy':>10} {'Gap':>8}")
    for b in report.calibration_buckets:
        if b["count"] == 0:
            continue
        rng = f"{b['low']:.1f}-{b['high']:.1f}"
        gap_str = f"{b['gap']:+.3f}"
        print(f"   {rng:<12} {b['count']:>8} {b['avg_confidence']:>10.3f} "
              f"{b['accuracy']:>10.3f} {gap_str:>8}")
    
    # --- Top failing registers ---
    print(f"\n TOP 10 WORST-PERFORMING REGISTERS (>= 3 samples)")
    print(f" {'-' * 60}")
    sorted_regs = sorted(
        report.per_register_accuracy.items(),
        key=lambda kv: kv[1]["accuracy"],
    )[:10]
    for reg, stats in sorted_regs:
        print(f"   {reg:<40} {stats['accuracy'] * 100:>6.1f}% "
              f"({stats['correct']}/{stats['total']})")
    
    # --- Sample failures ---
    failures = [r for r in records if not r["is_correct"]][:5]
    if failures:
        print(f"\n SAMPLE FAILURES (first 5)")
        print(f" {'-' * 60}")
        for f in failures:
            print(f"\n   [{f['entry_id']}] expected={f['expected_label']} "
                  f"predicted={f['predicted_label']} ({f['source']}, {f['register']})")
            print(f"     Expected: {f['expected_reasoning'][:120]}")
            print(f"     Got:      {f['predicted_reasoning'][:120]}")
    
    print(f"\n{hr}\n")


def export_failures_csv(records: List[Dict], path: str):
    """Export only the failed predictions as CSV for manual review."""
    import csv
    failures = [r for r in records if not r["is_correct"]]
    if not failures:
        return
    
    fieldnames = [
        "entry_id", "expected_label", "predicted_label", "source",
        "function_code", "register", "source_ip", "expected_confidence",
        "predicted_confidence", "latency_ms", "parse_method",
        "expected_reasoning", "predicted_reasoning", "raw_output",
    ]
    
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in failures:
            writer.writerow(r)


def export_full_predictions(records: List[Dict], path: str):
    """Write every prediction to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def export_report_json(report: EvalReport, path: str):
    """Write the aggregate report to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2)


# ============================================================
# SECTION 8: PROGRESS BAR (simple, no external deps)
# ============================================================

class ProgressBar:
    def __init__(self, total: int, width: int = 40):
        self.total = total
        self.width = width
        self.start = time.time()
        self.last_print = 0
    
    def update(self, current: int, extra: str = ""):
        now = time.time()
        if now - self.last_print < 0.3 and current < self.total:
            return
        self.last_print = now
        
        frac = current / self.total if self.total else 1
        filled = int(self.width * frac)
        bar = "#" * filled + "-" * (self.width - filled)
        
        elapsed = now - self.start
        rate = current / elapsed if elapsed > 0 else 0
        eta = (self.total - current) / rate if rate > 0 else 0
        eta_str = f"{int(eta // 60):02d}:{int(eta % 60):02d}"
        
        sys.stdout.write(
            f"\r  [{bar}] {current}/{self.total} "
            f"({frac * 100:.1f}%) ETA {eta_str} {extra}"
        )
        sys.stdout.flush()
        
        if current >= self.total:
            sys.stdout.write("\n")


# ============================================================
# SECTION 9: MAIN EVALUATION LOOP
# ============================================================

def run_evaluation(
    model_path: str,
    dataset_path: str,
    output_dir: str,
    sample_size: Optional[int],
    workers: int,
    n_threads: int,
    n_ctx: int,
    use_mmap: bool,
    use_mlock: bool,
    resume: bool,
    seed: int,
) -> Tuple[EvalReport, List[Dict]]:
    """Main orchestration."""
    os.makedirs(output_dir, exist_ok=True)
    
    # ---- Load dataset ----
    print(f"[*] Loading dataset: {dataset_path}")
    dataset = load_dataset(dataset_path)
    all_entries = dataset["entries"]
    dataset_name = dataset.get("dataset_name", "unknown")
    
    print(f"    Dataset:      {dataset_name}")
    print(f"    Version:      {dataset.get('version', 'unknown')}")
    print(f"    Total entries: {len(all_entries)}")
    print(f"    Labels:       {dataset.get('label_distribution', {})}")
    
    # ---- Select entries to evaluate ----
    if sample_size and sample_size < len(all_entries):
        print(f"\n[*] Stratified sample of {sample_size} entries (seed={seed})")
        entries = stratified_sample(all_entries, sample_size, seed=seed)
        sampled_labels = Counter(e["label"] for e in entries)
        print(f"    Sampled distribution: {dict(sampled_labels)}")
    else:
        entries = all_entries
    
    # ---- Resume from checkpoint if requested ----
    model_name = os.path.basename(model_path)
    ckpt_path = checkpoint_path(model_name, output_dir)
    completed_ids = set()
    records: List[Dict] = []
    
    if resume:
        prior = load_checkpoint(ckpt_path)
        if prior:
            records = prior
            completed_ids = {r["entry_id"] for r in prior}
            print(f"\n[+] Resumed from checkpoint: {len(completed_ids)} entries already done")
    else:
        # Fresh run: wipe any old checkpoint for this model
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
    
    remaining = [e for e in entries if e["id"] not in completed_ids]
    if not remaining:
        print(f"[+] All entries already completed. Computing report from checkpoint.")
    else:
        print(f"[*] Remaining to evaluate: {len(remaining)}")
    
    tasks = build_tasks(remaining)
    
    # ---- Launch workers ----
    model_size_gb = os.path.getsize(model_path) / 1024**3
    print(f"\n[*] Model:     {model_name} ({model_size_gb:.2f} GB)")
    print(f"[*] Workers:   {workers}")
    print(f"[*] Threads per worker: {n_threads}")
    print(f"[*] Context:   {n_ctx}")
    print(f"[*] Memory:    mmap={use_mmap}, mlock={use_mlock}")
    
    if tasks:
        print(f"\n[*] Starting evaluation ({len(tasks)} tasks)...")
        pbar = ProgressBar(len(tasks))
        start_time = time.time()
        
        # Setup Ctrl+C handler: save and exit gracefully
        interrupted = [False]
        def sigint_handler(signum, frame):
            print("\n\n[!] Interrupted! Saving checkpoint and exiting...")
            interrupted[0] = True
        signal.signal(signal.SIGINT, sigint_handler)
        
        try:
            if workers == 1:
                # Single-process path - no mp overhead, easier to debug
                worker_init(model_path, n_ctx, n_threads, use_mmap, use_mlock)
                
                # --- PRE-FLIGHT CONTEXT CHECK ---
                # Tokenize a real prompt and verify it fits in n_ctx.
                # This catches the "Requested tokens exceed context window"
                # error in ~5 seconds instead of after the first task.
                print(f"\n[*] Pre-flight context check...")
                sample_content = tasks[0]["user_content"]
                ok, msg = preflight_context_check(sample_content, n_ctx)
                if not ok:
                    print(f"\n[!] {msg}")
                    raise SystemExit(2)
                print(f"    [+] {msg}")
                
                # --- Consecutive-failure circuit breaker ---
                consecutive_errors = 0
                MAX_CONSECUTIVE_ERRORS = 3
                
                for i, task in enumerate(tasks):
                    if interrupted[0]:
                        break
                    try:
                        rec = worker_predict(task)
                        consecutive_errors = 0  # reset on success
                    except Exception as e:
                        consecutive_errors += 1
                        err_str = str(e)
                        print(f"\n[!] Error on {task['entry_id']}: {err_str}")
                        
                        # Fail fast on context size errors - no point continuing
                        if "exceed context window" in err_str or "n_ctx" in err_str.lower():
                            print(f"\n[!!] ABORT: Context window error is fatal.")
                            print(f"     Re-run with a larger --ctx-size value.")
                            print(f"     Current: {n_ctx}. Try: --ctx-size 2048")
                            raise SystemExit(2)
                        
                        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                            print(f"\n[!!] ABORT: {MAX_CONSECUTIVE_ERRORS} consecutive errors.")
                            print(f"     Something is systematically wrong.")
                            print(f"     Last error: {err_str}")
                            raise SystemExit(2)
                        continue
                    records.append(rec)
                    append_checkpoint(ckpt_path, rec)
                    pbar.update(i + 1,
                                f"acc={sum(r['is_correct'] for r in records) / len(records) * 100:.1f}%")
            else:
                # Multi-worker path
                ctx = mp.get_context("spawn")
                with ctx.Pool(
                    processes=workers,
                    initializer=worker_init,
                    initargs=(model_path, n_ctx, n_threads, use_mmap, use_mlock),
                ) as pool:
                    # Multi-worker can't easily preflight (workers are in
                    # separate processes), but llama.cpp will raise the same
                    # clear error on the first task. The consecutive-failure
                    # check below still catches it fast.
                    consecutive_errors = 0
                    MAX_CONSECUTIVE_ERRORS = 3
                    
                    for i, rec in enumerate(pool.imap_unordered(worker_predict, tasks)):
                        if interrupted[0]:
                            pool.terminate()
                            break
                        # Multi-worker returns normally (exceptions inside
                        # worker_predict would propagate). We approximate
                        # "failure" here as parse_method == 'failed'.
                        if rec.get("parse_method") == "failed":
                            consecutive_errors += 1
                            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                                print(f"\n[!!] ABORT: {MAX_CONSECUTIVE_ERRORS} "
                                      f"consecutive parse failures in a row.")
                                print(f"     Model may be producing garbage. "
                                      f"Check one prediction manually.")
                                pool.terminate()
                                raise SystemExit(2)
                        else:
                            consecutive_errors = 0
                        records.append(rec)
                        append_checkpoint(ckpt_path, rec)
                        pbar.update(i + 1,
                                    f"acc={sum(r['is_correct'] for r in records) / len(records) * 100:.1f}%")
        finally:
            # Restore default SIGINT handler
            signal.signal(signal.SIGINT, signal.SIG_DFL)
        
        elapsed = time.time() - start_time
    else:
        elapsed = 0.0
    
    # ---- Build report ----
    print(f"\n[*] Computing metrics...")
    report = compute_report(
        records=records,
        model_name=model_name,
        dataset_name=dataset_name,
        total_entries=len(entries),
        elapsed=elapsed,
    )
    
    return report, records


# ============================================================
# SECTION 10: CLI
# ============================================================

def main():
    plat = detect_platform_defaults()
    
    parser = argparse.ArgumentParser(
        description="SentinelEdge test suite - evaluate GGUF model on full dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Detected platform:
  CPU:            {plat['cpu_model'][:55]}
  RAM:            {plat['ram_gb']:.1f} GB
  Physical cores: {plat['physical_cores']}
  Raspberry Pi:   {plat['is_rpi']}
  Defaults:       workers={plat['default_workers']}, threads={plat['default_threads']}, ctx={plat['default_ctx']}

Examples:
  # Quick 200-entry stratified sample (~15 min on laptop)
  %(prog)s --model sentineledge-gemma2-2b-q5_k_m.gguf \\
           --dataset sentineledge_dataset.json --sample 200

  # Full 1830 evaluation on laptop (~2 hours)
  %(prog)s --model sentineledge-gemma2-2b-q5_k_m.gguf \\
           --dataset sentineledge_dataset.json

  # Full eval on RPi5 with Q4_K_M (~4-6 hours)
  %(prog)s --model sentineledge-gemma2-2b-q4_k_m.gguf \\
           --dataset sentineledge_dataset.json \\
           --threads 4 --ctx-size 2048

  # Resume after Ctrl+C
  %(prog)s --model sentineledge-gemma2-2b-q5_k_m.gguf \\
           --dataset sentineledge_dataset.json --resume
        """
    )
    
    parser.add_argument("--model", required=True, help="Path to .gguf model file")
    parser.add_argument("--dataset", required=True,
                        help="Path to sentineledge_dataset.json")
    parser.add_argument("--output-dir", default="./test_results",
                        help="Directory for reports and checkpoints")
    parser.add_argument("--sample", type=int, default=None,
                        help="Run on a stratified sample of N entries (default: full dataset)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    parser.add_argument("--workers", type=int, default=plat["default_workers"],
                        help=f"Parallel workers (default: {plat['default_workers']})")
    parser.add_argument("--threads", type=int, default=plat["default_threads"],
                        help=f"Threads per worker (default: {plat['default_threads']})")
    parser.add_argument("--ctx-size", type=int, default=plat["default_ctx"],
                        help=f"Context window size (default: {plat['default_ctx']})")
    parser.add_argument("--no-mmap", action="store_true", help="Disable mmap")
    parser.add_argument("--no-mlock", action="store_true", help="Disable mlock")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint if present")
    parser.add_argument("--no-reports", action="store_true",
                        help="Skip writing report files (console only)")
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.model):
        print(f"[!] Model file not found: {args.model}")
        sys.exit(1)
    if not os.path.exists(args.dataset):
        print(f"[!] Dataset file not found: {args.dataset}")
        sys.exit(1)
    
    # Print platform info
    print("=" * 72)
    print(" SentinelEdge Test Suite")
    print("=" * 72)
    print(f" Platform:       {'Raspberry Pi' if plat['is_rpi'] else 'Laptop/Desktop'}")
    print(f" CPU:            {plat['cpu_model'][:55]}")
    print(f" RAM:            {plat['ram_gb']:.1f} GB")
    print(f" Physical cores: {plat['physical_cores']}")
    print("=" * 72)
    
    # Run evaluation
    report, records = run_evaluation(
        model_path=args.model,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        sample_size=args.sample,
        workers=args.workers,
        n_threads=args.threads,
        n_ctx=args.ctx_size,
        use_mmap=not args.no_mmap,
        use_mlock=not args.no_mlock,
        resume=args.resume,
        seed=args.seed,
    )
    
    # Print console report
    print_report(report, records)
    
    # Write artifacts
    if not args.no_reports:
        model_slug = re.sub(r'[^a-zA-Z0-9_-]', '_', os.path.basename(args.model))
        
        report_json = os.path.join(args.output_dir, f"report_{model_slug}.json")
        preds_json = os.path.join(args.output_dir, f"predictions_{model_slug}.json")
        failures_csv = os.path.join(args.output_dir, f"failures_{model_slug}.csv")
        
        export_report_json(report, report_json)
        export_full_predictions(records, preds_json)
        export_failures_csv(records, failures_csv)
        
        print(f" Output files:")
        print(f"   Report:      {report_json}")
        print(f"   Predictions: {preds_json}")
        print(f"   Failures:    {failures_csv}")
        
        ckpt = checkpoint_path(os.path.basename(args.model), args.output_dir)
        print(f"   Checkpoint:  {ckpt}")
        print()
    
    # Exit code reflects accuracy: fail the run if < 70%
    sys.exit(0 if report.overall_accuracy >= 0.70 else 1)


if __name__ == "__main__":
    main()
