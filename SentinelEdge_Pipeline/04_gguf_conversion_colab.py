"""
==========================================================================
 SentinelEdge GGUF Conversion - Pure llama.cpp (v5, NO UNSLOTH)
 
 Converts the fine-tuned SentinelEdge model (merged 16-bit HF format)
 to GGUF for CPU edge deployment on laptop and Raspberry Pi 5.
 
 Pipeline:
   1. Locate merged 16-bit model from 02_gemma2_finetune_colab.py
   2. Clone llama.cpp + build llama-quantize
   3. Convert HF safetensors -> GGUF F16
   4. Quantize F16 GGUF -> Q8_0, Q5_K_M, Q4_K_M, Q4_0
   5. Validate one quant by running inference
   6. Copy to Google Drive
 
 Quantization Targets:
   - Q8_0   (~2.68 GB)  Laptop, max quality
   - Q5_K_M (~1.82 GB)  Laptop, sweet spot
   - Q4_K_M (~1.57 GB)  RPi5, recommended
   - Q4_0   (~1.44 GB)  RPi5, fastest
 
 Run AFTER 02_gemma2_finetune_colab.py completes successfully.
==========================================================================
"""

# %%  ========== CELL 1: PREREQUISITES ==========
import os, sys, shutil, subprocess, time, json
import torch

print("=" * 65)
print(" SentinelEdge GGUF Conversion (pure llama.cpp)")
print("=" * 65)

# Locate trained model — we need the MERGED 16-bit version, not LoRA adapters
MERGED_DIR = "./sentineledge_gemma2_merged_16bit"
GDRIVE_MERGED = "/content/drive/MyDrive/sentineledge_gemma2_model/merged_16bit"

if not os.path.exists(MERGED_DIR):
    print("[*] Local merged model not found. Mounting Drive...")
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        if os.path.exists(GDRIVE_MERGED):
            print(f"  [+] Found on Drive: {GDRIVE_MERGED}")
            print("  [*] Copying to local storage...")
            shutil.copytree(GDRIVE_MERGED, MERGED_DIR)
            print(f"  [+] Copied to {MERGED_DIR}")
        else:
            raise FileNotFoundError(
                f"Merged model not found.\n"
                f"  Expected at: {MERGED_DIR}  or  {GDRIVE_MERGED}\n"
                "Run 02_gemma2_finetune_colab.py first (Cell 14 saves the merged model)."
            )
    except ImportError:
        raise RuntimeError(f"Not in Colab. Place merged model at {MERGED_DIR}")

print(f"  [+] Merged model: {MERGED_DIR}")

# Verify it has the safetensors files
files = os.listdir(MERGED_DIR)
has_safetensors = any(f.endswith(".safetensors") for f in files)
has_config = "config.json" in files
print(f"  [+] safetensors: {has_safetensors}")
print(f"  [+] config.json: {has_config}")
assert has_safetensors and has_config, "Merged model is incomplete"

if torch.cuda.is_available():
    print(f"  [+] GPU: {torch.cuda.get_device_name(0)}")
print("=" * 65)


# %%  ========== CELL 2: INSTALL TOOLS ==========
print("[*] Installing dependencies...")

# Python deps for llama.cpp's HF conversion script
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "sentencepiece", "protobuf", "gguf", "transformers", "torch",
    "numpy", "safetensors",
], capture_output=True)

# llama-cpp-python for validation
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "llama-cpp-python",
], capture_output=True)

# Build deps
subprocess.run(["apt-get", "install", "-y", "cmake", "build-essential"],
               capture_output=True)

print("[+] Dependencies ready")


# %%  ========== CELL 3: CLONE & BUILD llama.cpp ==========
LLAMACPP_DIR = "/content/llama.cpp"

if not os.path.exists(LLAMACPP_DIR):
    print(f"[*] Cloning llama.cpp...")
    result = subprocess.run([
        "git", "clone", "--depth=1",
        "https://github.com/ggerganov/llama.cpp.git",
        LLAMACPP_DIR,
    ], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"git clone failed: {result.stderr}")
    print(f"[+] Cloned to {LLAMACPP_DIR}")
else:
    print(f"[+] llama.cpp already cloned")

# Install Python requirements for the conversion script
req_file = os.path.join(LLAMACPP_DIR, "requirements", "requirements-convert_hf_to_gguf.txt")
if os.path.exists(req_file):
    print(f"[*] Installing llama.cpp Python requirements...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q", "-r", req_file,
    ], capture_output=True)

# Build the quantize tool
print(f"[*] Building llama-quantize (this takes ~2-3 minutes)...")
build_dir = os.path.join(LLAMACPP_DIR, "build")

# Configure
subprocess.run(
    ["cmake", "-B", "build", "-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA=OFF"],
    cwd=LLAMACPP_DIR, capture_output=True, text=True,
)

# Build only what we need
build_result = subprocess.run(
    ["cmake", "--build", "build", "--config", "Release",
     "-t", "llama-quantize", "-j", str(os.cpu_count() or 4)],
    cwd=LLAMACPP_DIR, capture_output=True, text=True,
)

# Find the quantize binary
QUANTIZE_BIN = None
for path in [
    os.path.join(build_dir, "bin", "llama-quantize"),
    os.path.join(build_dir, "llama-quantize"),
]:
    if os.path.exists(path):
        QUANTIZE_BIN = path
        break

if QUANTIZE_BIN is None:
    print(f"[!] Build output:\n{build_result.stdout[-2000:]}")
    print(f"[!] Build errors:\n{build_result.stderr[-2000:]}")
    raise RuntimeError("llama-quantize not found after build")

print(f"[+] llama-quantize built: {QUANTIZE_BIN}")


# %%  ========== CELL 4: HF -> F16 GGUF CONVERSION ==========
GGUF_OUTPUT_BASE = "./sentineledge_gguf"
os.makedirs(GGUF_OUTPUT_BASE, exist_ok=True)

F16_GGUF = os.path.join(GGUF_OUTPUT_BASE, "sentineledge-gemma2-2b-f16.gguf")

if not os.path.exists(F16_GGUF):
    print(f"[*] Converting HF safetensors -> GGUF F16...")
    print(f"    This takes ~3-5 minutes for a 2B model")
    
    convert_script = os.path.join(LLAMACPP_DIR, "convert_hf_to_gguf.py")
    
    start = time.time()
    result = subprocess.run([
        sys.executable, convert_script,
        MERGED_DIR,
        "--outfile", F16_GGUF,
        "--outtype", "f16",
    ], capture_output=True, text=True)
    
    elapsed = time.time() - start
    
    if result.returncode != 0:
        print(f"[!] Conversion stdout:\n{result.stdout[-1500:]}")
        print(f"[!] Conversion stderr:\n{result.stderr[-1500:]}")
        raise RuntimeError("HF -> GGUF conversion failed")
    
    f16_size = os.path.getsize(F16_GGUF) / 1024**3
    print(f"[+] F16 GGUF created: {f16_size:.2f} GB ({elapsed:.0f}s)")
else:
    f16_size = os.path.getsize(F16_GGUF) / 1024**3
    print(f"[+] F16 GGUF already exists: {f16_size:.2f} GB")


# %%  ========== CELL 5: QUANTIZE TO TARGET LEVELS ==========
QUANT_TARGETS = {
    "q8_0":   ("Q8_0",   "Laptop max quality",  "~2.68 GB"),
    "q5_k_m": ("Q5_K_M", "Laptop sweet spot",    "~1.82 GB"),
    "q4_k_m": ("Q4_K_M", "RPi5 recommended",     "~1.57 GB"),
    "q4_0":   ("Q4_0",   "RPi5 fastest",          "~1.44 GB"),
}

print("=" * 65)
print(" Quantization")
print("=" * 65)

gguf_files = {}

for method, (quant_type, desc, est_size) in QUANT_TARGETS.items():
    output_gguf = os.path.join(
        GGUF_OUTPUT_BASE, f"sentineledge-gemma2-2b-{method}.gguf"
    )
    
    if os.path.exists(output_gguf):
        size = os.path.getsize(output_gguf) / 1024**3
        gguf_files[method] = output_gguf
        print(f"  [+] {method.upper():8s} already exists: {size:.2f} GB")
        continue
    
    print(f"\n  [*] Quantizing to {quant_type} ({desc}, est {est_size})...")
    start = time.time()
    
    result = subprocess.run(
        [QUANTIZE_BIN, F16_GGUF, output_gguf, quant_type],
        capture_output=True, text=True,
    )
    
    if result.returncode == 0 and os.path.exists(output_gguf):
        size = os.path.getsize(output_gguf) / 1024**3
        elapsed = time.time() - start
        gguf_files[method] = output_gguf
        print(f"  [+] {quant_type}: {size:.2f} GB ({elapsed:.0f}s)")
    else:
        print(f"  [!] {quant_type} failed")
        print(f"      stderr: {result.stderr[-300:]}")

print(f"\n[+] {len(gguf_files)} GGUF files generated")


# %%  ========== CELL 6: VALIDATE GGUF ==========
print("\n" + "=" * 65)
print(" Validation (load smallest GGUF + test inference)")
print("=" * 65)

try:
    from llama_cpp import Llama
    
    test_method = "q4_0" if "q4_0" in gguf_files else next(iter(gguf_files))
    test_file = gguf_files[test_method]
    
    print(f"\n[*] Loading {test_method.upper()} for validation...")
    llm = Llama(
        model_path=test_file,
        n_ctx=2048,
        n_threads=4,
        n_gpu_layers=0,
        verbose=False,
    )
    
    # Realistic SentinelEdge prompt format
    test_prompt = """<bos><start_of_turn>user
You are SentinelEdge, an AI-powered Zero-Trust cybersecurity monitor for a municipal water treatment SCADA system. Your role is to classify incoming Modbus commands as SAFE, SUSPICIOUS, or THREAT.

OPERATIONAL SAFETY LIMITS:
  Chlorine residual: 0.20-1.00 mg/L | pH: 6.5-8.5 | Tank level: >15%

DEFINITIONS:
- SAFE: Operationally appropriate
- SUSPICIOUS: Within limits but contextually risky
- THREAT: Limit violation or sabotage

Respond with exactly:
CATEGORY: [SAFE|SUSPICIOUS|THREAT]
CONFIDENCE: [float 0.0-1.0]
REASONING: [one concise sentence]

[PLANT STATE]
  chlorine_residual = 0.65 mg/L
  ph = 7.3
  tank_level = 65%

[INCOMING MODBUS COMMAND]
  function_code = FC03 (Read Holding Registers)
  register_address = 40001
  source_ip = 10.10.2.20

Classify this command. Respond with CATEGORY, CONFIDENCE, and REASONING only.<end_of_turn>
<start_of_turn>model
"""
    
    print("[*] Running test inference...")
    start = time.time()
    
    output = llm(
        test_prompt,
        max_tokens=200,
        temperature=0.15,
        top_p=0.90,
        top_k=40,
        repeat_penalty=1.18,
        stop=["<end_of_turn>", "<eos>"],
    )
    
    latency = (time.time() - start) * 1000
    response = output["choices"][0]["text"].strip()
    tokens_gen = output["usage"]["completion_tokens"]
    tps = tokens_gen / (latency / 1000) if latency > 0 else 0
    
    print(f"\n  Response ({tokens_gen} tokens, {latency:.0f}ms, {tps:.1f} tok/s):")
    print(f"  {response[:300]}")
    
    if "CATEGORY" in response:
        print(f"\n  [+] GGUF validation PASSED")
    else:
        print(f"\n  [!] Output missing CATEGORY: keyword - check fine-tuning quality")
    
    del llm

except ImportError:
    print("[!] llama-cpp-python not available - skipping validation")
except Exception as e:
    print(f"[!] Validation error: {e}")


# %%  ========== CELL 7: SAVE TO DRIVE ==========
print("\n" + "=" * 65)
print(" Saving to Google Drive")
print("=" * 65)

GDRIVE_GGUF = "/content/drive/MyDrive/sentineledge_gemma2_model/gguf"

try:
    from google.colab import drive
    if not os.path.exists("/content/drive/MyDrive"):
        drive.mount('/content/drive')
    
    os.makedirs(GDRIVE_GGUF, exist_ok=True)
    
    for method, filepath in gguf_files.items():
        if not os.path.exists(filepath):
            continue
        size = os.path.getsize(filepath) / 1024**3
        dest = os.path.join(GDRIVE_GGUF, f"sentineledge-gemma2-2b-{method}.gguf")
        print(f"  [*] Copying {method.upper()} ({size:.2f} GB)...")
        shutil.copy2(filepath, dest)
        print(f"  [+] {dest}")
    
    # Metadata
    metadata = {
        "base_model": "google/gemma-2-2b-it",
        "fine_tuning": "QLoRA r=32 alpha=64 (pure HF + TRL stack)",
        "task": "SentinelEdge SCADA Modbus IDS",
        "output_format": "plain text (CATEGORY/CONFIDENCE/REASONING)",
        "context_length": 2048,
        "quantizations": {
            method: {
                "filename": f"sentineledge-gemma2-2b-{method}.gguf",
                "size_gb": round(os.path.getsize(fp) / 1024**3, 2),
                "description": QUANT_TARGETS[method][1],
            }
            for method, fp in gguf_files.items() if os.path.exists(fp)
        },
    }
    
    with open(os.path.join(GDRIVE_GGUF, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  [+] Metadata: {GDRIVE_GGUF}/model_metadata.json")

except Exception as e:
    print(f"[!] Drive save: {e}")


# %%  ========== CELL 8: DEPLOYMENT GUIDE ==========
print("\n" + "=" * 65)
print(" GGUF CONVERSION COMPLETE")
print("=" * 65)
print(f"\n  Files in: {GDRIVE_GGUF}\n")

for method, filepath in sorted(gguf_files.items()):
    if os.path.exists(filepath):
        size = os.path.getsize(filepath) / 1024**3
        _, desc, _ = QUANT_TARGETS[method]
        print(f"    sentineledge-gemma2-2b-{method}.gguf  ({size:.2f} GB)  {desc}")

print("""

==================================================
 LAPTOP (Ryzen 7 5700U, 16GB RAM)
==================================================
 Recommended: Q5_K_M
 
 Install (with BLAS for speedup):
   CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" \\
     pip install llama-cpp-python --force-reinstall --no-cache-dir
 
 Run:
   python 05_edge_inference_llamacpp.py \\
     --model sentineledge-gemma2-2b-q5_k_m.gguf \\
     --threads 16 --demo

==================================================
 RASPBERRY PI 5 (8GB RAM, Cortex-A76)
==================================================
 Recommended: Q4_K_M
 
 Install:
   sudo apt update && sudo apt install -y cmake g++ python3-pip
   CMAKE_ARGS="-DGGML_NATIVE=ON" \\
     pip install llama-cpp-python --break-system-packages
 
 Run:
   python 05_edge_inference_llamacpp.py \\
     --model sentineledge-gemma2-2b-q4_k_m.gguf \\
     --threads 4 --ctx-size 1024 --demo
 
 Performance tip: enable performance governor
   echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
""")
print("=" * 65)
