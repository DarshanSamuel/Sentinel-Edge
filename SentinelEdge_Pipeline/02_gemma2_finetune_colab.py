"""
==========================================================================
 SentinelEdge SCADA IDS — Gemma 2 2B-IT Fine-Tuning (v5 — PURE HF STACK)
 
 Dataset: SentinelEdge-SCADA-Gemma2-v4.x (1830 entries, HF messages format)
 Model:   google/gemma-2-2b-it
 Stack:   transformers + TRL + PEFT + bitsandbytes  (NO UNSLOTH)
 
 Why this is a complete rewrite:
   The Unsloth-based version had three independent failure modes that
   compounded into a broken pipeline:
     1. UnslothSFTTrainer doesn't auto-detect "messages" datasets
     2. Unsloth's "gemma2" chat template lacks {% generation %} tags,
        breaking SFTConfig(assistant_only_loss=True)
     3. train_on_responses_only string-matching produced labels that
        gave loss = 9.37 (near-random for 256k vocab)
   
   This version uses the pure HuggingFace stack (the same one used in
   the AI Engineering Academy / adithya-s-k Gemma fine-tuning guide):
     - transformers AutoModelForCausalLM with BitsAndBytesConfig
     - peft LoraConfig + get_peft_model
     - trl SFTTrainer + DataCollatorForCompletionOnlyLM
     - HF's NATIVE Gemma 2 chat template (handles assistant->model)
   
   Response-only loss masking is done with explicit token IDs derived
   from the actual chat template output (Cell 7), which is bulletproof
   against tokenizer whitespace quirks.
 
 Runtime: Google Colab T4 (16GB) / L4 (24GB) / A10G (24GB)
 Estimated time: ~30 min on T4, ~15 min on L4
==========================================================================
"""

# %%  ========== CELL 1: ENVIRONMENT SETUP ==========
# Pure HuggingFace stack — NO Unsloth.
# Versions verified working as of April 2026 on Colab.

import subprocess, sys, os

def pip_install(args, quiet=True):
    cmd = [sys.executable, "-m", "pip", "install"] + args
    flags = ["-q"] if quiet else []
    return subprocess.run(cmd + flags, capture_output=True, text=True).returncode == 0

print("[*] Installing pure HuggingFace fine-tuning stack...")
pip_install([
    "--upgrade",
    "transformers>=4.46.0",      # Gemma 2 support, stable chat template
    "trl>=0.13.0,<0.20.0",       # Modern SFTTrainer, but avoid bleeding edge
    "peft>=0.13.0",              # LoRA + QLoRA support
    "bitsandbytes>=0.44.0",      # 4-bit NF4 quantization
    "accelerate>=1.0.0",         # Multi-GPU + device map
    "datasets>=3.0.0",           # Modern dataset API
    "sentencepiece",             # Gemma 2 tokenizer dependency
    "protobuf",                  # SentencePiece dependency
])

print("[+] Installation complete\n")

# Verify versions
import importlib
print("Installed versions:")
for pkg in ["transformers", "trl", "peft", "bitsandbytes", "accelerate", "datasets"]:
    try:
        mod = importlib.import_module(pkg)
        print(f"  {pkg:18s} = {getattr(mod, '__version__', 'unknown')}")
    except ImportError:
        print(f"  {pkg:18s} = NOT INSTALLED")


# %%  ========== CELL 2: GPU DIAGNOSTICS ==========
import torch

print("=" * 60)
print(" GPU DIAGNOSTICS")
print("=" * 60)

if not torch.cuda.is_available():
    raise RuntimeError(
        "No GPU detected! Enable: Runtime > Change runtime type > T4 GPU"
    )

GPU_NAME = torch.cuda.get_device_name(0)
GPU_MEM_GB = torch.cuda.get_device_properties(0).total_mem / 1024**3
COMPUTE_CAP = torch.cuda.get_device_capability(0)
SUPPORTS_BF16 = COMPUTE_CAP[0] >= 8  # Ampere+ supports bf16

print(f" GPU:            {GPU_NAME}")
print(f" VRAM:           {GPU_MEM_GB:.1f} GB")
print(f" Compute Cap:    {COMPUTE_CAP[0]}.{COMPUTE_CAP[1]}")
print(f" Precision:      {'bfloat16' if SUPPORTS_BF16 else 'float16'}")
print(f" PyTorch:        {torch.__version__}")
print(f" CUDA:           {torch.version.cuda}")

# Auto-tune batch size
if GPU_MEM_GB >= 22:
    BATCH_SIZE, GRAD_ACCUM = 4, 4
    GPU_TIER = "HIGH (L4/A10G/A100)"
elif GPU_MEM_GB >= 14:
    BATCH_SIZE, GRAD_ACCUM = 2, 8
    GPU_TIER = "MEDIUM (T4)"
else:
    BATCH_SIZE, GRAD_ACCUM = 1, 16
    GPU_TIER = "LOW"

print(f" Tier:           {GPU_TIER}")
print(f" Batch size:     {BATCH_SIZE}")
print(f" Grad accum:     {GRAD_ACCUM}")
print(f" Effective bs:   {BATCH_SIZE * GRAD_ACCUM}")
print("=" * 60)


# %%  ========== CELL 3: LOAD GEMMA 2 2B-IT DIRECTLY (NO UNSLOTH) ==========
# Load google/gemma-2-2b-it directly with bitsandbytes 4-bit quantization.
# This is the standard HuggingFace pattern documented at
# https://huggingface.co/docs/transformers/main_classes/quantization

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

MODEL_ID = "google/gemma-2-2b-it"
MAX_SEQ_LENGTH = 2048

print(f"[*] Loading {MODEL_ID} with 4-bit NF4 quantization...")

# QLoRA standard config: NF4 + double quantization + bf16 compute
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if SUPPORTS_BF16 else torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# Gemma 2 uses <pad> for padding; if missing, use eos
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# IMPORTANT: padding_side must be "right" for causal LM training
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map={"": 0},
    attn_implementation="eager",  # Gemma 2 logit soft-cap is most reliable with eager
    torch_dtype=torch.bfloat16 if SUPPORTS_BF16 else torch.float16,
)

print(f"\n[+] Model loaded")
print(f"    BOS:              {tokenizer.bos_token!r} (id={tokenizer.bos_token_id})")
print(f"    EOS:              {tokenizer.eos_token!r} (id={tokenizer.eos_token_id})")
print(f"    PAD:              {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})")
print(f"    <start_of_turn>:  id={tokenizer.convert_tokens_to_ids('<start_of_turn>')}")
print(f"    <end_of_turn>:    id={tokenizer.convert_tokens_to_ids('<end_of_turn>')}")
print(f"    Vocab size:       {tokenizer.vocab_size}")
print(f"    VRAM after load:  {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# Verify HF's native chat template handles assistant->model conversion
# (Gemma 2's official template maps role='assistant' -> '<start_of_turn>model')
test_msgs = [
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "hi there"},
]
test_formatted = tokenizer.apply_chat_template(test_msgs, tokenize=False)
print(f"\n[*] Verifying Gemma 2 chat template (HF native):")
print(f"    {test_formatted!r}")
assert "<start_of_turn>user" in test_formatted, "Template missing user marker!"
assert "<start_of_turn>model" in test_formatted, "Template missing model marker (assistant->model failed)!"
print(f"    [+] HF chat template correctly maps assistant -> model")


# %%  ========== CELL 4: PREPARE FOR K-BIT TRAINING + ATTACH LoRA ==========
# Standard QLoRA setup:
#   1. prepare_model_for_kbit_training - enables grad checkpointing,
#      casts norms to fp32, freezes base weights
#   2. LoraConfig - LoRA hyperparameters
#   3. get_peft_model - injects trainable LoRA adapters

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

print("[*] Preparing model for k-bit training...")
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()
model.config.use_cache = False  # Required when gradient checkpointing is on

# LoRA config — same hyperparameters as before, but applied via PEFT directly
# instead of Unsloth's get_peft_model wrapper
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

trainable, total = 0, 0
for p in model.parameters():
    n = p.numel()
    total += n
    if p.requires_grad:
        trainable += n

print(f"\n[+] LoRA adapters attached")
print(f"    Trainable params:  {trainable:,} ({trainable/total*100:.4f}%)")
print(f"    Total params:      {total:,}")
print(f"    VRAM after LoRA:   {torch.cuda.memory_allocated() / 1024**3:.2f} GB")


# %%  ========== CELL 5: LOAD SENTINELEDGE DATASET + APPLY CHAT TEMPLATE ==========
# Load the JSON dataset, extract the messages list, and apply Gemma 2's
# native chat template to produce a "text" column.
#
# The dataset uses role='assistant' which is the HF standard convention.
# Gemma 2's chat template (shipped with google/gemma-2-2b-it) automatically
# converts 'assistant' -> 'model' when rendering, so the resulting text
# will contain <start_of_turn>model as expected by Gemma 2.

import json
from datasets import Dataset

# ---- Locate dataset ----
dataset_paths = [
    "sentineledge_dataset.json",
    "/content/sentineledge_dataset.json",
    "/content/drive/MyDrive/sentineledge_dataset.json",
]
DATASET_PATH = next((p for p in dataset_paths if os.path.exists(p)), None)

if DATASET_PATH is None:
    print("[!] Dataset not found locally. Mounting Drive...")
    from google.colab import drive
    drive.mount('/content/drive')
    DATASET_PATH = "/content/drive/MyDrive/sentineledge_dataset.json"
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Upload sentineledge_dataset.json or place at {DATASET_PATH}")

print(f"[*] Loading: {DATASET_PATH}")
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

print(f"\n[+] Dataset loaded:")
print(f"    Name:    {raw_data.get('dataset_name', 'unknown')}")
print(f"    Version: {raw_data.get('version', 'unknown')}")
print(f"    Total:   {raw_data.get('total_entries', len(raw_data.get('entries', [])))}")
print(f"\n  Label distribution:")
for label, count in raw_data.get("label_distribution", {}).items():
    pct = count / raw_data["total_entries"] * 100
    print(f"    {label:12s}: {count:4d} ({pct:.1f}%)")

# Detect the role names actually used in the dataset (assistant vs model)
roles_seen = set()
for entry in raw_data["entries"][:50]:
    for msg in entry["messages"]:
        roles_seen.add(msg["role"])
print(f"\n  Roles found in dataset: {roles_seen}")

# HF's Gemma 2 template accepts both 'assistant' and 'model'.
# If the dataset uses 'model', we leave it alone (template handles it).
# If it uses 'assistant', the template auto-converts to <start_of_turn>model.

# ---- Convert entries to HF Dataset ----
entries = raw_data["entries"]
hf_records = [{"messages": e["messages"]} for e in entries]
full_dataset = Dataset.from_list(hf_records)
print(f"\n[+] Built HF Dataset: {len(full_dataset)} rows")

# ---- Apply chat template to produce a "text" column ----
# We use HF's NATIVE Gemma 2 chat template (loaded with the tokenizer).
# Batched mapping is much faster and avoids per-example overhead.
print(f"\n[*] Applying Gemma 2 chat template (HF native)...")

def format_messages(examples):
    """Convert each conversation in messages -> formatted text string."""
    texts = [
        tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=False,
        )
        for convo in examples["messages"]
    ]
    return {"text": texts}

full_dataset = full_dataset.map(
    format_messages,
    batched=True,
    num_proc=2,
    remove_columns=full_dataset.column_names,
)

print(f"[+] Dataset formatted. Columns: {full_dataset.column_names}")

# ---- Sanity check the formatted output ----
sample_text = full_dataset[0]["text"]
print(f"\n[+] First formatted example (head 600 chars):")
print("-" * 60)
print(sample_text[:600] + "...")
print("-" * 60)

# Verify Gemma 2 control tokens and dataset content are present
required_tokens = ["<bos>", "<start_of_turn>user", "<start_of_turn>model", "<end_of_turn>"]
for tok in required_tokens:
    assert tok in sample_text, f"Missing {tok} in formatted text!"
assert "CATEGORY" in sample_text, "Missing CATEGORY in formatted text!"
print(f"\n    [+] All Gemma 2 control tokens present")
print(f"    [+] Response structure (CATEGORY) present")

# ---- Token length distribution ----
print("\n[*] Scanning token length distribution (sampling 100)...")
import random
random.seed(42)
sample_lengths = []
for idx in random.sample(range(len(full_dataset)), min(100, len(full_dataset))):
    text = full_dataset[idx]["text"]
    toks = tokenizer(text, add_special_tokens=False)["input_ids"]
    sample_lengths.append(len(toks))

print(f"    Min:    {min(sample_lengths)} tokens")
print(f"    Max:    {max(sample_lengths)} tokens")
print(f"    Mean:   {sum(sample_lengths) / len(sample_lengths):.0f} tokens")
print(f"    p95:    {sorted(sample_lengths)[int(len(sample_lengths) * 0.95)]} tokens")
print(f"    Headroom at MAX_SEQ_LENGTH={MAX_SEQ_LENGTH}: "
      f"{MAX_SEQ_LENGTH - max(sample_lengths)} tokens")


# %%  ========== CELL 6: TRAIN/EVAL SPLIT ==========
# 95/5 split. The dataset is already balanced (715 SAFE / 567 SUSPICIOUS / 548 THREAT)
# so a random shuffle gives a representative eval set.

split = full_dataset.train_test_split(test_size=0.05, seed=42, shuffle=True)
train_dataset = split["train"]
eval_dataset = split["test"]

print(f"[+] Dataset split:")
print(f"    Train: {len(train_dataset)} examples")
print(f"    Eval:  {len(eval_dataset)} examples")


# %%  ========== CELL 7: BUILD RESPONSE-ONLY LOSS COLLATOR ==========
# This is the bulletproof way to do response-only loss masking on Gemma 2.
#
# DataCollatorForCompletionOnlyLM works by finding a specific token sequence
# (the "response template") in each input_ids, and setting labels=-100 for
# all tokens BEFORE that sequence. This computes loss only on the model's
# response, not the user prompt.
#
# The challenge: tokenizing "<start_of_turn>model\n" in isolation may produce
# DIFFERENT token IDs than when it appears inside a real conversation,
# because of SentencePiece's whitespace handling.
#
# THE FIX: Derive the response_template token IDs from the ACTUAL output of
# the chat template applied to a real example. This guarantees the IDs
# match what appears in training inputs.

from trl import DataCollatorForCompletionOnlyLM

print("[*] Deriving response template token IDs from chat template output...")

# Build a minimal example with just a user message and use add_generation_prompt=True
# This produces text ending in the exact "<start_of_turn>model\n" sequence we want
# to match in real training inputs.
probe_messages = [{"role": "user", "content": "probe"}]
probe_ids = tokenizer.apply_chat_template(
    probe_messages,
    tokenize=True,
    add_generation_prompt=True,
)
print(f"    Probe input IDs (length={len(probe_ids)}): {probe_ids}")
print(f"    Probe decoded: {tokenizer.decode(probe_ids)!r}")

# The response template is the suffix of probe_ids starting from the LAST
# <start_of_turn> token (id 106 for Gemma 2). Everything from that token
# to the end is the marker that says "model is about to speak."
SOT_ID = tokenizer.convert_tokens_to_ids("<start_of_turn>")
EOT_ID = tokenizer.convert_tokens_to_ids("<end_of_turn>")
print(f"    <start_of_turn> id: {SOT_ID}")
print(f"    <end_of_turn> id:   {EOT_ID}")

# Find the position of the LAST <start_of_turn> in probe_ids
sot_positions = [i for i, t in enumerate(probe_ids) if t == SOT_ID]
assert len(sot_positions) >= 2, (
    f"Expected at least 2 <start_of_turn> tokens, found {len(sot_positions)}. "
    "Chat template may be malformed."
)
response_start_idx = sot_positions[-1]

# response_template_ids = the marker tokens that begin the model's turn
# For Gemma 2 this is typically [106, model_token_id, newline_id]
response_template_ids = probe_ids[response_start_idx:]
print(f"\n    Response template token IDs: {response_template_ids}")
print(f"    Decoded: {tokenizer.decode(response_template_ids)!r}")

# Sanity check: this template MUST appear in a full training example
sample_full_ids = tokenizer(train_dataset[0]["text"], add_special_tokens=False)["input_ids"]
def find_subsequence(haystack, needle):
    """Return the start index of needle in haystack, or -1 if not found."""
    n = len(needle)
    for i in range(len(haystack) - n + 1):
        if haystack[i:i+n] == needle:
            return i
    return -1

found_idx = find_subsequence(sample_full_ids, response_template_ids)
print(f"\n    Searching for template in real training example...")
print(f"    Found at position: {found_idx} (out of {len(sample_full_ids)} tokens)")
assert found_idx >= 0, (
    "Response template NOT found in real training example! "
    "DataCollatorForCompletionOnlyLM will fail. Check tokenizer + chat template."
)
print(f"    [+] Template matches — collator will work correctly")

# Build the collator with these explicit token IDs (NOT a string)
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template_ids,
    tokenizer=tokenizer,
    mlm=False,
)
print(f"\n[+] DataCollatorForCompletionOnlyLM built")


# %%  ========== CELL 8: SFTConfig — TRAINING HYPERPARAMETERS ==========
# Modern TRL API: SFTConfig replaces TrainingArguments for SFT-specific args.
# All dataset-related parameters live here now (max_seq_length, dataset_text_field,
# packing, etc.).
#
# We do NOT use assistant_only_loss=True because that requires {% generation %}
# tags in the chat template (Gemma 2's template doesn't have them). Instead,
# our DataCollatorForCompletionOnlyLM in Cell 7 handles response-only masking
# explicitly via token ID matching.

from trl import SFTConfig

steps_per_epoch = len(train_dataset) // (BATCH_SIZE * GRAD_ACCUM)
total_steps = steps_per_epoch * 3
warmup_steps = max(int(total_steps * 0.05), 5)

print(f"[*] Training plan:")
print(f"    Steps per epoch:  {steps_per_epoch}")
print(f"    Total steps:      {total_steps}")
print(f"    Warmup steps:     {warmup_steps}")

sft_config = SFTConfig(
    # ---- Output ----
    output_dir="./sentineledge_checkpoints",
    
    # ---- Dataset ----
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",       # We pre-formatted into "text" in Cell 5
    packing=False,                   # Required with response-only collator
    dataset_num_proc=2,
    
    # ---- Schedule ----
    num_train_epochs=3,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    
    # ---- Optimizer ----
    optim="paged_adamw_8bit",        # Same as medium post; saves VRAM
    learning_rate=2e-4,
    weight_decay=0.01,
    max_grad_norm=1.0,
    
    # ---- LR scheduler ----
    lr_scheduler_type="cosine",
    warmup_steps=warmup_steps,
    
    # ---- Precision ----
    bf16=SUPPORTS_BF16,
    fp16=not SUPPORTS_BF16,
    
    # ---- Memory ----
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    
    # ---- Logging ----
    logging_steps=5,
    logging_first_step=True,
    report_to="none",
    
    # ---- Eval ----
    eval_strategy="steps",
    eval_steps=25,
    
    # ---- Saving ----
    save_strategy="steps",
    save_steps=50,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    
    # ---- Reproducibility ----
    seed=42,
    data_seed=42,
)

print(f"\n[+] SFTConfig built")
print(f"    Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")
print(f"    Learning rate:        {sft_config.learning_rate}")
print(f"    LR scheduler:         {sft_config.lr_scheduler_type}")
print(f"    Precision:            {'bf16' if SUPPORTS_BF16 else 'fp16'}")


# %%  ========== CELL 9: BUILD SFTTrainer + VERIFY MASKING ==========
# Modern TRL API:
#   - args=SFTConfig (not TrainingArguments)
#   - processing_class=tokenizer (replaces deprecated tokenizer=)
#   - data_collator=our_response_only_collator

from trl import SFTTrainer

print("[*] Building SFTTrainer...")

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    data_collator=collator,
)

print(f"[+] SFTTrainer built")
print(f"    Train examples: {len(train_dataset)}")
print(f"    Eval examples:  {len(eval_dataset)}")

# ---- Verify masking by inspecting an actual collated batch ----
# This is the ground truth: we ask the trainer to give us the EXACT
# label tensor that CrossEntropyLoss will see during training.
print(f"\n[*] Verifying response-only masking on a real batch...")

# Get the first training example through the trainer's data pipeline
sample_dataset = trainer.train_dataset.select(range(min(2, len(trainer.train_dataset))))
sample_examples = [sample_dataset[i] for i in range(len(sample_dataset))]

# Run them through the collator (this is exactly what the training loop does)
batch = collator(sample_examples)
labels = batch["labels"][0].tolist()
input_ids = batch["input_ids"][0].tolist()

total_tokens = sum(1 for t in input_ids if t != tokenizer.pad_token_id)
trained_tokens = sum(1 for lbl in labels if lbl != -100)
masked_tokens = total_tokens - trained_tokens
trained_pct = trained_tokens / total_tokens * 100

print(f"    Total tokens:    {total_tokens}")
print(f"    Trained tokens:  {trained_tokens} ({trained_pct:.1f}%)")
print(f"    Masked tokens:   {masked_tokens} ({100-trained_pct:.1f}%)")

# Decode trained tokens — must contain CATEGORY
trained_token_ids = [t for t, l in zip(input_ids, labels) if l != -100]
if trained_token_ids:
    decoded_trained = tokenizer.decode(trained_token_ids, skip_special_tokens=False)
    print(f"\n    First 250 chars of trained tokens (should contain CATEGORY):")
    print(f"    >>> {decoded_trained[:250]!r}")
    
    if "CATEGORY" in decoded_trained:
        print(f"\n    [+] PASS — collator correctly identified the response region")
    else:
        raise RuntimeError(
            "FAIL: 'CATEGORY' not found in trained tokens. "
            "The response template is matching the wrong location."
        )
else:
    raise RuntimeError(
        "FAIL: zero trained tokens. The response template was not found "
        "in the input_ids — DataCollatorForCompletionOnlyLM produced "
        "an all-masked batch."
    )

# Healthy ratio: 5-15% trained for SentinelEdge prompts
if 0.02 <= trained_tokens / total_tokens <= 0.30:
    print(f"    [+] Mask ratio looks healthy ({trained_pct:.1f}%)")
else:
    print(f"    [!] Unusual mask ratio ({trained_pct:.1f}%) — investigate")


# %%  ========== CELL 10: EXPECTED LOSS TRAJECTORY ==========
EXPECTED_LOSS_TRAJECTORY = """
================================================================
 EXPECTED LOSS TRAJECTORY (response-only masking, pure HF stack)
================================================================

For 1738 train examples, effective batch size 16, 3 epochs:
  steps_per_epoch ~= 108
  total_steps     ~= 326
  warmup_steps    ~= 16

These numbers are for response-only loss (DataCollatorForCompletionOnlyLM
masks the user prompt; loss is computed only on the model's CATEGORY/
CONFIDENCE/REASONING output).

+---------+-------------+------------+----------------------+
|  Step   |  Train Loss |  Eval Loss |  What's happening    |
+---------+-------------+------------+----------------------+
|   0     |  3.5 - 4.5  |     -      | Cold start           |
|   5     |  3.0 - 3.8  |     -      | Warmup ramp          |
|  10     |  2.5 - 3.2  |     -      | Warmup ramp          |
|  15     |  2.0 - 2.6  |     -      | End of warmup        |
|  20     |  1.6 - 2.2  |     -      | Format learning      |
|  25     |  1.3 - 1.8  |  1.4 - 2.0 | First eval point     |
|  50     |  0.7 - 1.1  |  0.8 - 1.2 | Format locked in     |
|  75     |  0.5 - 0.8  |  0.6 - 0.9 |                      |
| 100     |  0.4 - 0.6  |  0.5 - 0.7 | End of epoch 1       |
| 150     |  0.25- 0.45 |  0.35- 0.55|                      |
| 200     |  0.18- 0.35 |  0.28- 0.45| End of epoch 2       |
| 250     |  0.13- 0.28 |  0.23- 0.40|                      |
| 300     |  0.10- 0.22 |  0.20- 0.36|                      |
| 325     |  0.08- 0.18 |  0.20- 0.35| Final step           |
+---------+-------------+------------+----------------------+

================================================================
 IF YOU SEE LOSS = 9 OR HIGHER AT STEP 25
================================================================
That means response-only masking is broken — model is being asked
to predict random plant state numbers (impossible task -> high loss).
Check Cell 9 output:
  - Trained tokens % should be 5-15% (not 0% and not 100%)
  - First trained tokens MUST contain "CATEGORY"
  - If either fails, DataCollatorForCompletionOnlyLM is not finding
    the response template token IDs in the input_ids
================================================================

KEY MILESTONES:

* Step 0: Loss should be 3.5-4.5
  - For Gemma 2 with 256k vocab, ln(256000) ~= 12.45 = pure random
  - Initial loss ~3-4 means LoRA init + good base model on a fresh task
  - If you see >5: tokenizer/template/masking is broken
  - If you see >9: response masking is completely wrong

* Step 25: First eval (~step 25)
  - Train ~1.3-1.8, Eval ~1.4-2.0
  - The model has learned the CATEGORY/CONFIDENCE/REASONING format
  - Eval slightly higher than train is normal

* Step 100 (epoch 1 end): ~0.4-0.6
  - Most of the learning happens here
  - Eval should track within 0.1-0.2 of train

* Step 200 (epoch 2 end): ~0.18-0.35
  - Refining factual associations (chemical interactions, attack patterns)

* Step 325 (final): ~0.08-0.18 train, ~0.20-0.35 eval
  - load_best_model_at_end recovers the best eval checkpoint

================================================================
 WARNING SIGNS
================================================================

[!] Loss > 5 at step 0:
    Tokenizer or chat template broken. Re-run Cell 5 and Cell 9.

[!] Loss > 8 at any point:
    Response template matching failed. Check Cell 7's "Found at
    position: X" output - if X = -1, the collator can't find the
    boundary and is masking everything (or nothing).

[!] Loss oscillates 2 -> 6 -> 1 -> 7:
    LR too high. Drop to 1e-4 in Cell 8.

[!] Eval >> Train (gap > 0.5):
    Overfitting. Reduce epochs to 2.

[!] NaN loss:
    Use bf16, not fp16. Check max_grad_norm=1.0 is set.

[!] OOM:
    BATCH_SIZE=1, GRAD_ACCUM=16. Already auto-tuned in Cell 2.
================================================================
"""
print(EXPECTED_LOSS_TRAJECTORY)
print("[*] Training starts in 5 seconds...")
import time
time.sleep(5)


# %%  ========== CELL 11: TRAIN ==========
print("=" * 60)
print(" STARTING FINE-TUNING (pure HuggingFace stack)")
print("=" * 60)
print(f" Model:          google/gemma-2-2b-it (NF4 4-bit QLoRA)")
print(f" Dataset:        SentinelEdge ({len(train_dataset)} train / {len(eval_dataset)} eval)")
print(f" Effective bs:   {BATCH_SIZE * GRAD_ACCUM}")
print(f" Epochs:         3")
print(f" Total steps:    {total_steps}")
print(f" GPU:            {GPU_NAME}")
print(f" Stack:          transformers + trl + peft + bitsandbytes")
print("=" * 60)
print()

start_time = time.time()
torch.cuda.reset_peak_memory_stats()

trainer_stats = trainer.train()

elapsed = time.time() - start_time
peak_vram = torch.cuda.max_memory_allocated() / 1024**3

print(f"\n{'=' * 60}")
print(f" TRAINING COMPLETE")
print(f"{'=' * 60}")
print(f" Duration:           {elapsed/60:.1f} min")
print(f" Final train loss:   {trainer_stats.training_loss:.4f}")
print(f" Peak VRAM:          {peak_vram:.2f} GB")
print(f" Samples/second:     {len(train_dataset) * 3 / elapsed:.1f}")
print(f"{'=' * 60}")


# %%  ========== CELL 12: FINAL EVALUATION ==========
print("[*] Running final evaluation on held-out set...")
eval_results = trainer.evaluate()

print(f"\n[+] Final Eval Results:")
print(f"    eval_loss:        {eval_results['eval_loss']:.4f}")
print(f"    eval_perplexity:  {2 ** eval_results['eval_loss']:.2f}")

if 0.20 <= eval_results['eval_loss'] <= 0.40:
    print(f"    [+] Within expected range (0.20-0.40)")
elif eval_results['eval_loss'] < 0.20:
    print(f"    [+] BETTER than expected!")
else:
    print(f"    [!] HIGHER than expected. Consider:")
    print(f"        - Train more epochs")
    print(f"        - Increase LoRA rank to 64")
    print(f"        - Lower learning rate to 1e-4")


# %%  ========== CELL 13: SANITY INFERENCE ==========
print("\n[*] Running sanity inference on 3 examples (one per category)...\n")

# Switch model to inference mode
model.eval()

# Pick one example from each category
test_examples = []
seen_labels = set()
for entry in raw_data["entries"]:
    label = entry["label"]
    if label not in seen_labels:
        test_examples.append(entry)
        seen_labels.add(label)
    if len(seen_labels) == 3:
        break

from transformers import GenerationConfig

gen_config = GenerationConfig(
    max_new_tokens=200,
    temperature=0.15,
    top_p=0.90,
    top_k=40,
    repetition_penalty=1.18,
    do_sample=True,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

correct_count = 0
for i, example in enumerate(test_examples):
    user_msg = example["messages"][0]["content"]
    expected_label = example["label"]
    
    # Build prompt with chat template + generation prompt
    input_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False,
        add_generation_prompt=True,
    )
    
    inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=False).to(model.device)
    
    with torch.no_grad():
        output = model.generate(**inputs, generation_config=gen_config)
    
    generated = output[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    
    # Parse predicted CATEGORY
    predicted = "UNKNOWN"
    for line in response.split("\n"):
        if line.upper().startswith("CATEGORY:"):
            predicted = line.split(":", 1)[1].strip().upper()
            # Strip non-alpha
            predicted = "".join(c for c in predicted if c.isalpha())
            break
    
    is_correct = predicted == expected_label
    correct_count += int(is_correct)
    status = "[OK]" if is_correct else "[X]"
    
    print(f"--- Test {i+1}: expected={expected_label} ---")
    print(f"  Predicted: {predicted} {status}")
    print(f"  Response:  {response[:300]}")
    print()

print(f"Sanity accuracy: {correct_count}/{len(test_examples)}")


# %%  ========== CELL 14: SAVE MODEL ==========
import shutil

print("\n[*] Saving model artifacts...\n")

LORA_DIR = "./sentineledge_gemma2_lora"
MERGED_DIR = "./sentineledge_gemma2_merged_16bit"

# Save LoRA adapters
trainer.model.save_pretrained(LORA_DIR)
tokenizer.save_pretrained(LORA_DIR)
print(f"[+] LoRA adapters saved: {LORA_DIR}")

# Save merged 16-bit model
print(f"\n[*] Merging LoRA into base model (16-bit)...")
from peft import PeftModel

# Load fresh base in fp16 (not 4-bit) for clean merge
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map={"": 0},
    low_cpu_mem_usage=True,
)
merged = PeftModel.from_pretrained(base_model, LORA_DIR)
merged = merged.merge_and_unload()
merged.save_pretrained(MERGED_DIR, safe_serialization=True)
tokenizer.save_pretrained(MERGED_DIR)
print(f"[+] Merged 16-bit model saved: {MERGED_DIR}")

# Save generation config
gen_config_dict = {
    "max_new_tokens": 200,
    "min_new_tokens": 15,
    "temperature": 0.15,
    "top_p": 0.90,
    "top_k": 40,
    "repetition_penalty": 1.18,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id,
}
with open(os.path.join(LORA_DIR, "generation_config_scada.json"), "w") as f:
    json.dump(gen_config_dict, f, indent=2)

# Copy to Drive
try:
    from google.colab import drive
    if not os.path.exists("/content/drive/MyDrive"):
        drive.mount('/content/drive')
    
    gdrive_root = "/content/drive/MyDrive/sentineledge_gemma2_model"
    os.makedirs(gdrive_root, exist_ok=True)
    
    for src, name in [(LORA_DIR, "lora_adapters"), (MERGED_DIR, "merged_16bit")]:
        dest = os.path.join(gdrive_root, name)
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        print(f"[+] Copied to Drive: {dest}")
        
except Exception as e:
    print(f"[!] Drive save skipped: {e}")


# %%  ========== CELL 15: EXPORT TRAINING METRICS ==========
log_history = trainer.state.log_history
metrics = {
    "train_loss_history": [
        {"step": x["step"], "loss": x["loss"]}
        for x in log_history if "loss" in x and "eval_loss" not in x
    ],
    "eval_loss_history": [
        {"step": x["step"], "eval_loss": x["eval_loss"]}
        for x in log_history if "eval_loss" in x
    ],
    "final_train_loss": trainer_stats.training_loss,
    "final_eval_loss": eval_results["eval_loss"],
    "total_steps": trainer.state.global_step,
    "epochs_completed": trainer.state.epoch,
    "training_minutes": elapsed / 60,
    "peak_vram_gb": peak_vram,
    "gpu": GPU_NAME,
    "effective_batch_size": BATCH_SIZE * GRAD_ACCUM,
    "stack": "pure HF (transformers+trl+peft+bitsandbytes)",
}

metrics_path = os.path.join(LORA_DIR, "training_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n[+] Training metrics saved: {metrics_path}")
print(f"\n  Loss progression:")
if metrics['train_loss_history']:
    print(f"    Initial loss: {metrics['train_loss_history'][0]['loss']:.3f}")
    mid = len(metrics['train_loss_history']) // 2
    print(f"    Mid loss:     {metrics['train_loss_history'][mid]['loss']:.3f}")
print(f"    Final train:  {metrics['final_train_loss']:.3f}")
print(f"    Final eval:   {metrics['final_eval_loss']:.3f}")


# %%  ========== CELL 16: DONE ==========
print(f"\n{'=' * 60}")
print(f" FINE-TUNING COMPLETE")
print(f"{'=' * 60}")
print(f"")
print(f"  Stack:            pure HuggingFace (no Unsloth)")
print(f"  LoRA adapters:    {LORA_DIR}/")
print(f"  Merged model:     {MERGED_DIR}/")
print(f"  Drive backup:     /content/drive/MyDrive/sentineledge_gemma2_model/")
print(f"")
print(f"  Final train loss: {trainer_stats.training_loss:.4f}")
print(f"  Final eval loss:  {eval_results['eval_loss']:.4f}")
print(f"")
print(f"  Next steps:")
print(f"    1. Run 04_gguf_conversion_colab.py to make edge GGUF files")
print(f"    2. Use 05_edge_inference_llamacpp.py to deploy")
print(f"")
print(f"{'=' * 60}")
