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


# %%  ========== CELL 4:SETUP FOR F16 TO GGUF CONVERSION ==========
import os
import shutil
from huggingface_hub import hf_hub_download

# Update this if you used a different base model (e.g., "google/gemma-2-2b-it")
BASE_MODEL_ID = "google/gemma-2-2b"
MERGED_DIR = "sentineledge_gemma2_merged_16bit"

print(f"[*] Downloading tokenizer.model from {BASE_MODEL_ID}...")
try:
    # Download the missing tokenizer.model from the original Hugging Face repo
    original_tokenizer_path = hf_hub_download(
        repo_id=BASE_MODEL_ID,
        filename="tokenizer.model"
    )

    # Copy it directly into your merged directory where llama.cpp is looking for it
    dest_path = os.path.join(MERGED_DIR, "tokenizer.model")
    shutil.copy(original_tokenizer_path, dest_path)

    print(f"[+] Successfully copied tokenizer.model to {dest_path}")
except Exception as e:
    print(f"[!] Error downloading tokenizer.model: {e}")
    print("    Please ensure you have accepted the Gemma terms on Hugging Face")
    print("    and are logged in using `huggingface-cli login` or notebook_login()")


# %%  ========== CELL 5: HF -> F16 GGUF CONVERSION ==========
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
 

# %%  ========== CELL 6: QUANTIZE TO TARGET LEVELS ==========
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
print("=" * 65)
