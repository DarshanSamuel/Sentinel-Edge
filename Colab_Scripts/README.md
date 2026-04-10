# Colab Scripts

These scripts run in **Google Colab** (or any environment with a GPU + internet access). They produce the dataset, the fine-tuned model, and the GGUF files for edge deployment.

## Run order

1. **`generate_dataset.py`** — Generate the training dataset (run locally OR in Colab).
2. **`gemma2_finetune_colab.py`** — Fine-tune Gemma 2 2B-IT on the dataset (Colab T4/L4).
3. **`gguf_conversion_colab.py`** — Convert merged 16-bit model to four GGUF quantizations (Colab CPU).

## 01 — Dataset generator

Pure-Python script (no GPU, no ML deps) that produces 1830 stratified training entries with 18 distinct physics-based scenarios spanning SAFE / SUSPICIOUS / THREAT classifications.

```bash
python generate_dataset.py --total 1830 --output sentineledge_dataset.json
```

Output: `sentineledge_dataset.json` (~9 MB) with the canonical SentinelEdge v4 format.

## 02 — Fine-tuning

Pure HuggingFace stack: `transformers` + `trl` + `peft` + `bitsandbytes`. **No Unsloth.** Uses QLoRA (NF4 4-bit) with `DataCollatorForCompletionOnlyLM` for response-only loss masking.

Run **cell by cell** in Colab. Cells 7 and 9 contain bulletproof masking checks that abort training before it starts if anything is wrong.

**Hyperparameters:**
- LoRA r=32, α=64, dropout=0.05, all 7 modules (q/k/v/o + gate/up/down)
- Learning rate: 2e-4 with cosine schedule
- 3 epochs, effective batch size 16
- `paged_adamw_8bit` optimizer

**Expected losses:**
- Step 0: ~3.5–4.5
- Step 25 (first eval): ~1.3–1.8 train, ~1.4–2.0 eval
- Step 100 (epoch 1): ~0.4–0.6
- Step 325 (final): ~0.08–0.18 train, ~0.20–0.35 eval

If you see step 25 loss > 5, something is wrong with response masking. Check Cell 7 output.

**Outputs** (saved to `/content/drive/MyDrive/sentineledge_gemma2_model/`):
- `lora_adapters/` — LoRA-only weights (~80 MB)
- `merged_16bit/` — Full merged 16-bit model (~5 GB)

## 04 — GGUF conversion

Clones llama.cpp, builds the `llama-quantize` binary, runs `convert_hf_to_gguf.py` on the merged 16-bit model from step 02, then quantizes to four levels:

| Quant | Size | Recommended for |
|---|---|---|
| Q8_0 | ~2.68 GB | PC, max quality |
| Q5_K_M | ~1.82 GB | PC, sweet spot |
| Q4_K_M | ~1.57 GB | Edge Device, balanced |
| Q4_0 | ~1.44 GB | Edge Device, fastest |

After conversion, download the `.gguf` files from Google Drive and use them with `model_inference.py`
