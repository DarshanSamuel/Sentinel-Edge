#!/usr/bin/env python3
"""
==========================================================================
 SentinelEdge Firebase Upload Service
 
 Simulates a real SCADA Edge AI pipeline by:
   1. Loading the codes.json (Modbus code registry) 
   2. Loading the sentineledge_dataset.json (full scenario corpus)
   3. Randomly picking a scenario from the dataset
   4. Looking up the Modbus code from codes.json  
   5. Using the dataset's ground-truth label as the "model inference"
   6. Printing a rich console summary of the command + inference
   7. Uploading the telemetry document to Firebase Firestore
   
 The uploaded document matches the exact schema expected by the
 Antigravity Flutter dashboard (DashboardProvider):
   - timestamp        (SERVER_TIMESTAMP)
   - flow_rate        (double)
   - pressure         (double)
   - modbus_code      (string, e.g. "MB007")
   - modbus_register_value (int)
   - is_anomalous     (bool)
   
 Additionally, it stores enriched metadata for deeper analysis:
   - classification   (SAFE | SUSPICIOUS | THREAT)
   - confidence       (float 0.0-1.0)
   - reasoning        (string)
   - register_name    (string)
   - function_code    (string, e.g. "FC06")
   - commanded_value  (the raw value from the Modbus command)
   - source_ip        (string)
   - entry_id         (string, dataset scenario ID)

 Prerequisites:
   pip install firebase-admin

 Usage (Simulation):
   # Simulation
   python firebase_upload_service.py

   # Upload a single random scenario
   python firebase_upload_service.py

   # Upload N scenarios with a delay between each
   python firebase_upload_service.py --count 10 --delay 3

   # Continuous mode (upload every 5 seconds until Ctrl+C)
   python firebase_upload_service.py --continuous --delay 5

   # Target specific label types for testing
   python firebase_upload_service.py --label THREAT --count 5
   python firebase_upload_service.py --label SAFE --count 5
   
 Usage (Real Time AI Inference Engine):
   # Real Edge AI Hardware Mode
  python firebase_upload_service.py --model models/sentineledge-gemma2-2b-q4_k_m.gguf --continuous --delay 10

==========================================================================
"""

import firebase_admin
from firebase_admin import credentials, firestore
import datetime
import json
import os
import sys
import time
import random
import argparse
import signal
import re


# ================================================================
# SECTION 0: LOCAL AI INFERENCE (OPTIONAL)
# ================================================================

GEMMA2_PROMPT = "<bos><start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n"

def init_llm(model_path: str):
    """Initializes the local LLM using llama_cpp for live inference."""
    if not model_path:
        return None
    if not os.path.exists(model_path):
        print(f"  ❌ Model file not found: {model_path}")
        sys.exit(1)
        
    try:
        from llama_cpp import Llama
        print(f"  🧠 Loading LLM Model for LIVE inference: {model_path}...")
        return Llama(
            model_path=model_path,
            n_ctx=2048, # Context window large enough for SCADA state + generation
            n_threads=os.cpu_count() or 4,
            n_gpu_layers=0, # Assuming CPU edge deployment
            verbose=False
        )
    except ImportError:
        print("  ❌ llama-cpp-python is required for live inference.")
        print("  Run: pip install llama-cpp-python")
        sys.exit(1)

def parse_response(raw: str):
    """Parses Category, Confidence, and Reasoning from LLM string output."""
    category = "UNKNOWN"
    confidence = 0.0
    reasoning = ""
    
    for line in [l.strip() for l in raw.strip().split("\n")]:
        up = line.upper()
        if up.startswith("CATEGORY:"):
            cat = "".join(c for c in line.split(":", 1)[1].strip().upper() if c.isalpha())
            if cat in ("SAFE", "SUSPICIOUS", "THREAT"): category = cat
        elif up.startswith("CONFIDENCE:"):
            m = re.search(r'([\d.]+)', line.split(":", 1)[1])
            if m: confidence = float(m.group(1))
        elif up.startswith("REASONING:"):
            reasoning = line.split(":", 1)[1].strip()
            
    if category == "UNKNOWN":
        cat_m = re.search(r'CATEGORY[:\s]+([A-Z]+)', raw, re.IGNORECASE)
        if cat_m and cat_m.group(1).upper() in ("SAFE", "SUSPICIOUS", "THREAT"):
            category = cat_m.group(1).upper()
            
    return category, confidence, reasoning


# ================================================================
# SECTION 1: FIREBASE SERVICE
# ================================================================

class FirebaseTelemetryService:
    """Handles authentication and document writes to Firestore."""

    def __init__(self, credential_path="service-account.json"):
        """
        Initializes the Firebase Admin SDK.
        Uses a Service Account key, which bypasses client-side 
        Security Rules — allowing the Edge Hardware to write to 
        the locked-down database directly.
        """
        if not os.path.exists(credential_path):
            raise FileNotFoundError(
                f"Missing '{credential_path}'.\n"
                f"Download it from: Firebase Console -> Project Settings -> Service Accounts -> Generate New Private Key"
            )

        # Initialize Firebase only once (idempotent)
        if not firebase_admin._apps:
            cred = credentials.Certificate(credential_path)
            firebase_admin.initialize_app(cred)

        self.db = firestore.client()
        self.main_collection = "scada_telemetry"
        print(f"  ✅ Firebase authenticated via Service Account")

    def push_telemetry(self, payload: dict, label: str) -> str:
        """
        Pushes telemetry to various collections based on label, and manages retention.
        """
        collections_to_push = [self.main_collection]
        if label == "SAFE":
            collections_to_push.append("commands_safe")
        elif label == "SUSPICIOUS":
            collections_to_push.append("commands_suspicious")
        elif label == "THREAT":
            collections_to_push.append("commands_threat")
            
        try:
            # Generate a common document ID across all collections for this telemetry entry
            doc_ref = self.db.collection(self.main_collection).document()
            doc_id = doc_ref.id
            
            batch = self.db.batch()
            for coll in collections_to_push:
                batch.set(self.db.collection(coll).document(doc_id), payload)
            batch.commit()
            
            # Handle retention
            if label == "SAFE":
                self._trim_safe_collection("commands_safe", 100)
            else:
                if label == "SUSPICIOUS":
                    self._trim_anomalous_collection("commands_suspicious", 1000)
                if label == "THREAT":
                    self._trim_anomalous_collection("commands_threat", 1000)
                
            return doc_id
        except Exception as e:
            print(f"  ❌ Firebase write failed: {e}")
            return ""

    def _trim_safe_collection(self, collection_name: str, limit: int):
        """
        Keep up to limit (100) entries. If the 101st arrives, delete the previous 100
        (keeping only the newest 1), both from the safe collection AND main collection.
        """
        try:
            docs = list(self.db.collection(collection_name).order_by("timestamp", direction=firestore.Query.DESCENDING).limit(limit + 1).stream())
            if len(docs) > limit:
                # Keep index 0 (newest), delete all the rest
                docs_to_delete = docs[1:]
                
                # Delete from safe collection and main collection using the same IDs
                batch = self.db.batch()
                for d in docs_to_delete:
                    batch.delete(d.reference) # Delete from commands_safe
                    batch.delete(self.db.collection(self.main_collection).document(d.id)) # Delete from main
                batch.commit()
                print(f"\n  🧹 Cleared {len(docs_to_delete)} old SAFE entries from '{collection_name}' and main directory")
        except Exception as e:
            print(f"  ❌ Trim failed for {collection_name}: {e}")

    def _trim_anomalous_collection(self, collection_name: str, limit: int):
        """
        Keep up to 'limit' entries for anomalous collections
        """
        try:
            docs = list(self.db.collection(collection_name).order_by("timestamp", direction=firestore.Query.DESCENDING).limit(limit + 1).stream())
            if len(docs) > limit:
                docs_to_delete = docs[limit:] # Keep the first 'limit' entries
                batch = self.db.batch()
                for d in docs_to_delete:
                    batch.delete(d.reference)
                batch.commit()
        except Exception:
            pass


# ================================================================
# SECTION 2: DATA LOADERS
# ================================================================

def load_codes(path="codes.json") -> dict:
    """Load the Modbus code registry."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing '{path}' — the Modbus code registry.")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  📋 Loaded {data['total_codes']} Modbus codes from codes.json")
    return data


def load_dataset(path="sentineledge_dataset.json") -> dict:
    """Load the SentinelEdge scenario dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing '{path}' — the SentinelEdge dataset.")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    total = data.get("total_entries", len(data.get("entries", [])))
    dist = data.get("label_distribution", {})
    print(f"  📊 Loaded {total} scenarios from dataset")
    print(f"     Distribution: SAFE={dist.get('SAFE',0)} | "
          f"SUSPICIOUS={dist.get('SUSPICIOUS',0)} | "
          f"THREAT={dist.get('THREAT',0)}")
    return data


def find_code_for_register(codes_data: dict, register_name: str) -> dict:
    """
    Look up a Modbus code entry from codes.json by register name.
    Returns the code dict, or an empty dict if not found.
    """
    for code_id, code_info in codes_data.get("codes_by_id", {}).items():
        if code_info.get("register", "").lower() == register_name.lower():
            return code_info
    return {}


# ================================================================
# SECTION 3: SCENARIO SELECTION
# ================================================================

def pick_random_scenario(dataset: dict, label_filter: str = None) -> dict:
    """
    Pick a random scenario entry from the dataset.
    Optionally filter by label (SAFE, SUSPICIOUS, THREAT).
    """
    entries = dataset["entries"]
    if label_filter:
        label_filter = label_filter.upper()
        filtered = [e for e in entries if e["label"] == label_filter]
        if not filtered:
            print(f"  ⚠️  No entries with label '{label_filter}', using full dataset")
            filtered = entries
        return random.choice(filtered)
    return random.choice(entries)


# ================================================================
# SECTION 4: CONSOLE DISPLAY
# ================================================================

LABEL_ICONS = {
    "SAFE": "✅",
    "SUSPICIOUS": "⚠️ ",
    "THREAT": "🚨",
    "UNKNOWN": "❓",
}

LABEL_COLORS = {
    "SAFE": "\033[92m",       # Green
    "SUSPICIOUS": "\033[93m", # Yellow
    "THREAT": "\033[91m",     # Red
}
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[96m"


def print_scenario_header(index: int, total: int):
    """Print a divider for a new scenario."""
    width = 72
    print(f"\n{'═' * width}")
    print(f" 📡 SCADA COMMAND #{index}/{total}")
    print(f"{'═' * width}")


def print_modbus_command(entry: dict, code_info: dict):
    """Print the incoming Modbus command details."""
    cmd = entry["metadata"]["modbus_command"]
    code_id = code_info.get("code", "UNKNOWN")
    
    print(f"\n  {BOLD}[INCOMING MODBUS COMMAND]{RESET}")
    print(f"  {'─' * 50}")
    print(f"  Code:           {CYAN}{code_id}{RESET}")
    print(f"  Register:       {cmd.get('reg', 'unknown')}")
    print(f"  Function:       {cmd.get('fc', '?')} ({cmd.get('fn', '?')})")
    print(f"  Address:        {cmd.get('addr', '?')}")
    print(f"  Value:          {cmd.get('val', '?')} {cmd.get('unit', '')}")
    print(f"  Source IP:      {cmd.get('ip', '?')}")
    print(f"  Destination UID: {cmd.get('uid', '?')}")
    
    # Show extra flags if present
    extra = cmd.get("extra", {})
    if extra:
        for key, val in extra.items():
            print(f"  ⚡ {key}: {val}")


def print_plant_state_summary(plant_state: dict):
    """Print a compact view of key plant parameters."""
    print(f"\n  {BOLD}[PLANT STATE SNAPSHOT]{RESET}")
    print(f"  {'─' * 50}")
    
    # Show the most critical parameters
    key_params = [
        ("flow_rate_L_min", "Flow Rate", "L/min"),
        ("distribution_pressure_PSI", "Pressure", "PSI"),
        ("chlorine_residual_mg_L", "Chlorine", "mg/L"),
        ("ph", "pH", ""),
        ("turbidity_treated_NTU", "Turbidity", "NTU"),
        ("pump_rpm", "Pump RPM", "RPM"),
        ("tank_level_pct", "Tank Level", "%"),
        ("valve_position_pct", "Valve Pos.", "%"),
    ]
    
    for key, label, unit in key_params:
        val = plant_state.get(key, "N/A")
        unit_str = f" {unit}" if unit else ""
        print(f"    {label:<16} = {val}{unit_str}")


def print_inference_result(entry: dict, code_id: str):
    """Print the AI model inference output."""
    label = entry["label"]
    confidence = entry["metadata"]["confidence"]
    reasoning = entry["metadata"]["reasoning"]
    icon = LABEL_ICONS.get(label, "❓")
    color = LABEL_COLORS.get(label, "")
    
    print(f"\n  {BOLD}[AI INFERENCE OUTPUT]{RESET}")
    print(f"  {'─' * 50}")
    print(f"  Classification: {color}{BOLD}{icon} {label}{RESET}")
    print(f"  Confidence:     {confidence:.0%}")
    print(f"  Reasoning:      {DIM}{reasoning}{RESET}")
    
    is_anomalous = label in ("SUSPICIOUS", "THREAT")
    if is_anomalous:
        print(f"\n  {LABEL_COLORS['THREAT']}{'▓' * 50}")
        print(f"  ▓  ANOMALY DETECTED — UPLOADING TO CLOUD  ▓")
        print(f"  {'▓' * 50}{RESET}")
    else:
        print(f"\n  {LABEL_COLORS['SAFE']}  ── Normal operation ──{RESET}")


def print_upload_result(doc_id: str, code_id: str, is_anomalous: bool):
    """Print the Firebase upload confirmation."""
    status = "🚨 ANOMALY" if is_anomalous else "✅ SECURE"
    if doc_id:
        print(f"\n  {BOLD}[FIREBASE UPLOAD]{RESET}")
        print(f"  Status:      {status}")
        print(f"  Code:        {code_id}")
        print(f"  Document ID: {DIM}{doc_id}{RESET}")
        print(f"  Collection:  scada_telemetry")
    else:
        print(f"\n  ❌ Upload failed — check Firebase connection")


# ================================================================
# SECTION 5: MAIN PIPELINE
# ================================================================

def build_firestore_payload(entry: dict, code_info: dict) -> dict:
    """
    Build the Firestore document payload that matches the
    Flutter DashboardProvider's expected schema.
    """
    metadata = entry["metadata"]
    plant = metadata["plant_state"]
    cmd = metadata["modbus_command"]
    label = entry["label"]
    code_id = code_info.get("code", "UNKNOWN")

    # Core fields — these are what the Flutter dashboard reads
    payload = {
        "timestamp": firestore.SERVER_TIMESTAMP,
        "flow_rate": plant.get("flow_rate_L_min", 0.0),
        "pressure": plant.get("distribution_pressure_PSI", 0.0),
        "modbus_code": code_id,
        "modbus_register_value": int(str(cmd.get("addr", "40000")).replace("40", "", 1)) if isinstance(cmd.get("addr"), str) else 0,
        "is_anomalous": label in ("SUSPICIOUS", "THREAT"),
    }

    # Enriched metadata — extra context for advanced dashboard views
    payload["classification"] = label
    payload["confidence"] = metadata.get("confidence", 0.0)
    payload["reasoning"] = metadata.get("reasoning", "")
    payload["register_name"] = cmd.get("reg", "unknown")
    payload["function_code"] = cmd.get("fc", "unknown")
    payload["commanded_value"] = str(cmd.get("val", ""))
    payload["commanded_unit"] = cmd.get("unit", "")
    payload["source_ip"] = cmd.get("ip", "unknown")
    payload["destination_uid"] = cmd.get("uid", 0)
    payload["entry_id"] = entry.get("id", "unknown")
    payload["source_type"] = entry.get("source", "unknown")
    payload["plant_state"] = plant # Added 21 plant parameters

    return payload


def run_single_scenario(
    firebase_service: FirebaseTelemetryService,
    dataset: dict,
    codes_data: dict,
    index: int,
    total: int,
    label_filter: str = None,
    llm = None,
):
    """Execute one full pipeline cycle: pick → display → upload."""

    # 1. Pick a random scenario
    entry = pick_random_scenario(dataset, label_filter)
    
    # 1.5. Live Model Inference (Optional)
    if llm:
        user_content = entry["messages"][0]["content"]
        prompt = GEMMA2_PROMPT.format(user=user_content)
        
        start_time = time.time()
        output = llm(
            prompt,
            max_tokens=200,
            temperature=0.15,
            stop=["<end_of_turn>", "<eos>"],
        )
        latency_ms = (time.time() - start_time) * 1000
        raw_text = output["choices"][0]["text"].strip()
        
        category, confidence, reasoning = parse_response(raw_text)
        
        # Override mock ground-truth with live LLM inferences
        entry["label"] = category
        entry["metadata"]["confidence"] = confidence
        entry["metadata"]["reasoning"] = reasoning
        entry["metadata"]["latency_ms"] = round(latency_ms, 1)

    cmd = entry["metadata"]["modbus_command"]
    register_name = cmd.get("reg", "unknown")

    # 2. Cross-reference with codes.json
    code_info = find_code_for_register(codes_data, register_name)
    if not code_info:
        # Fallback: try matching by function code + address combo
        for cid, cdata in codes_data.get("codes_by_id", {}).items():
            if (cdata.get("function_code") == cmd.get("fc") and
                cdata.get("address") == cmd.get("addr")):
                code_info = cdata
                break
    if not code_info:
        code_info = {"code": "UNKNOWN", "register": register_name}

    code_id = code_info.get("code", "UNKNOWN")

    # 3. Console output
    print_scenario_header(index, total)
    print_modbus_command(entry, code_info)
    print_plant_state_summary(entry["metadata"]["plant_state"])
    print_inference_result(entry, code_id)

    # 4. Build payload and upload
    payload = build_firestore_payload(entry, code_info)
    doc_id = firebase_service.push_telemetry(payload, entry["label"])
    is_anomalous = entry["label"] in ("SUSPICIOUS", "THREAT")
    print_upload_result(doc_id, code_id, is_anomalous)

    return entry["label"]


def main():
    parser = argparse.ArgumentParser(
        description="SentinelEdge Firebase Upload Service — Simulates edge AI telemetry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single upload
  python firebase_upload_service.py

  # Upload 10 scenarios, 3 seconds apart
  python firebase_upload_service.py --count 10 --delay 3

  # Continuous mode (Ctrl+C to stop)
  python firebase_upload_service.py --continuous --delay 5

  # Only upload THREAT scenarios
  python firebase_upload_service.py --label THREAT --count 5

  # Only upload SAFE scenarios  
  python firebase_upload_service.py --label SAFE --count 5

  # Mix of all labels, fast  
  python firebase_upload_service.py --count 20 --delay 1
        """
    )

    parser.add_argument("--count", type=int, default=1,
                        help="Number of scenarios to upload (default: 1)")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Seconds between uploads (default: 2.0)")
    parser.add_argument("--continuous", action="store_true",
                        help="Run forever until Ctrl+C")
    parser.add_argument("--label", type=str, default=None,
                        choices=["SAFE", "SUSPICIOUS", "THREAT"],
                        help="Filter scenarios by classification label")
    parser.add_argument("--credential", type=str, default="service-account.json",
                        help="Path to Firebase service account JSON (default: service-account.json)")
    parser.add_argument("--dataset", type=str, default="sentineledge_dataset.json",
                        help="Path to the SentinelEdge dataset JSON")
    parser.add_argument("--codes", type=str, default="codes.json",
                        help="Path to the Modbus codes JSON")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to GGUF model for LIVE inference (bypasses mock data)")

    args = parser.parse_args()

    # Banner
    print(f"\n{'═' * 72}")
    print(f" ╔═══════════════════════════════════════════════════════════════════╗")
    print(f" ║          SENTINELEDGE — Firebase Upload Service                  ║")
    print(f" ║          SCADA Edge AI → Cloud Telemetry Pipeline                ║")
    print(f" ╚═══════════════════════════════════════════════════════════════════╝")
    print(f"{'═' * 72}")

    # Load resources
    print(f"\n  Loading resources...")
    try:
        codes_data = load_codes(args.codes)
        dataset = load_dataset(args.dataset)
        firebase = FirebaseTelemetryService(args.credential)
        llm = init_llm(args.model)
    except FileNotFoundError as e:
        print(f"\n  ❌ {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n  ❌ Initialization failed: {e}")
        sys.exit(1)

    print(f"\n  {'─' * 50}")
    mode = "CONTINUOUS" if args.continuous else f"{args.count} scenario(s)"
    inference_mode = f"LIVE MODEL ({args.model})" if args.model else "SIMULATED (Dataset Labels)"
    print(f"  Mode:      {mode}")
    print(f"  Delay:     {args.delay}s between uploads")
    print(f"  Inference: {inference_mode}")
    if args.label:
        print(f"  Filter:    {args.label} only")
    print(f"  {'─' * 50}")

    # Graceful shutdown on Ctrl+C
    running = [True]
    def signal_handler(signum, frame):
        print(f"\n\n  🛑 Shutting down gracefully...")
        running[0] = False
    signal.signal(signal.SIGINT, signal_handler)

    # Upload loop
    stats = {"SAFE": 0, "SUSPICIOUS": 0, "THREAT": 0}
    index = 0

    try:
        if args.continuous:
            total_str = "∞"
            while running[0]:
                index += 1
                label = run_single_scenario(
                    firebase, dataset, codes_data, 
                    index, total_str, args.label, llm
                )
                stats[label] = stats.get(label, 0) + 1
                if running[0]:
                    time.sleep(args.delay)
        else:
            for i in range(args.count):
                if not running[0]:
                    break
                index += 1
                label = run_single_scenario(
                    firebase, dataset, codes_data,
                    index, args.count, args.label, llm
                )
                stats[label] = stats.get(label, 0) + 1
                if i < args.count - 1 and running[0]:
                    time.sleep(args.delay)
    except KeyboardInterrupt:
        pass

    # Summary
    print(f"\n{'═' * 72}")
    print(f" SESSION SUMMARY")
    print(f"{'═' * 72}")
    print(f"  Total uploaded:  {index}")
    print(f"  ✅ SAFE:         {stats.get('SAFE', 0)}")
    print(f"  ⚠️  SUSPICIOUS:  {stats.get('SUSPICIOUS', 0)}")
    print(f"  🚨 THREAT:       {stats.get('THREAT', 0)}")
    print(f"{'═' * 72}\n")


if __name__ == "__main__":
    main()
