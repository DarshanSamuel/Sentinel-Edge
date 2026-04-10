#!/usr/bin/env python3
"""
==========================================================================
 SentinelEdge Modbus Command Mapper
 
 One-time script that scans sentineledge_dataset.json to:
   1. Extract every unique register/command targeted in the dataset
   2. Assign each one a stable Modbus code (MB001, MB002, ...)
   3. Compute value ranges, units, function codes per register
   4. Index dataset entries by register for fast scenario lookup
   5. Add hand-curated friendly-name aliases for natural-language parsing
   6. Write modbus_codes.json that the inference script loads at startup
 
 The output JSON has O(1) dict lookups by:
   - code        (MB001 -> register info)
   - register    (chlorine_pump_speed -> code + info)
   - alias       (chlorine -> register + code)
 
 Usage:
   python map_commands.py
   python map_commands.py --dataset path/to/dataset.json --output codes.json
==========================================================================
"""

import argparse
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Any


# ============================================================
# HAND-CURATED FRIENDLY-NAME ALIASES
# ============================================================
# Maps natural-language terms a human might type in the terminal
# to the canonical register names used in the SentinelEdge dataset.
# Add more aliases here if you want richer parsing.

FRIENDLY_ALIASES: Dict[str, str] = {
    # Chemical dosing
    "chlorine":          "chlorine_pump_speed",
    "chlorine_pump":     "chlorine_pump_speed",
    "cl":                "chlorine_pump_speed",
    "cl2":               "chlorine_pump_speed",
    
    "alum":              "alum_dosing_rate",
    "alum_dosing":       "alum_dosing_rate",
    "coagulant":         "alum_dosing_rate",
    
    "fluoride":          "fluoride_dosing_rate",
    "f":                 "fluoride_dosing_rate",
    
    "polymer":           "polymer_dosing_rate",
    
    "ph":                "ph_correction_pump",
    "ph_correction":     "ph_correction_pump",
    "ph_pump":           "ph_correction_pump",
    
    "hypochlorite":      "hypochlorite_feed_rate",
    "naocl":             "hypochlorite_feed_rate",
    
    "chloramine":        "chloramine_feed_rate",
    "chloramine_feed":   "chloramine_feed_rate",
    
    "ozone":             "ozone_generator_setpoint",
    "ozone_gen":         "ozone_generator_setpoint",
    "o3":                "ozone_generator_setpoint",
    
    "uv":                "uv_intensity_setpoint",
    "uv_intensity":      "uv_intensity_setpoint",
    
    # Pumps & flow
    "pump":              "main_pump_rpm",
    "main_pump":         "main_pump_rpm",
    "rpm":               "main_pump_rpm",
    
    "flow":              "flow_setpoint",
    "flowrate":          "flow_setpoint",
    "flow_rate":         "flow_setpoint",
    
    # Valves
    "tank_inlet":        "tank_inlet_valve",
    "tank":              "tank_inlet_valve",
    "inlet":             "tank_inlet_valve",
    
    "bypass":            "filter_bypass_valve",
    "filter_bypass":     "filter_bypass_valve",
    
    "pressure_relief":   "pressure_relief_valve",
    "relief":            "pressure_relief_valve",
    "prv":               "pressure_relief_valve",
    
    "isolation_a":       "isolation_valve_A",
    "valve_a":           "isolation_valve_A",
    "iso_a":             "isolation_valve_A",
    
    "isolation_b":       "isolation_valve_B",
    "valve_b":           "isolation_valve_B",
    "iso_b":             "isolation_valve_B",
    
    "backwash":          "backwash_valve",
    
    "chloramine_valve":  "chloramine_feed_valve",
    
    # Alarms / setpoints
    "uv_alarm":          "uv_alarm_setpoint",
    "turbidity_alarm":   "turbidity_alarm_setpoint",
    "toc_alarm":         "toc_alarm_setpoint",
    "tds_alarm":         "tds_alarm_setpoint",
    "do_alarm":          "do_alarm_setpoint",
    "temp":              "temp_setpoint",
    "temperature":       "temp_setpoint",
    
    # Emergency / multi-register
    "shutdown":          "emergency_shutdown",
    "emergency":         "emergency_shutdown",
    "estop":             "emergency_shutdown",
    "e_stop":            "emergency_shutdown",
}


# ============================================================
# DATASET ANALYSIS
# ============================================================

def analyze_dataset(dataset_path: str) -> Dict[str, Any]:
    """Walk through the dataset and collect per-register statistics."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    entries = data["entries"]
    print(f"[*] Loaded {len(entries)} entries")
    
    # reg -> info bucket
    reg_data: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "values": [],
        "units": set(),
        "function_codes": set(),
        "function_names": set(),
        "addresses": set(),
        "label_counts": defaultdict(int),
        "entry_ids": [],          # for fast scenario lookup
        "sample_plant_state": None,
        "sample_reasoning": None,
    })
    
    for entry in entries:
        cmd = entry["metadata"]["modbus_command"]
        reg = cmd.get("reg", "unknown")
        
        bucket = reg_data[reg]
        val = cmd.get("val")
        if isinstance(val, (int, float)):
            bucket["values"].append(float(val))
        bucket["units"].add(cmd.get("unit", "") or "")
        bucket["function_codes"].add(cmd.get("fc", "") or "")
        bucket["function_names"].add(cmd.get("fn", "") or "")
        bucket["addresses"].add(cmd.get("addr", "") or "")
        bucket["label_counts"][entry["label"]] += 1
        bucket["entry_ids"].append(entry["id"])
        
        if bucket["sample_plant_state"] is None:
            bucket["sample_plant_state"] = entry["metadata"]["plant_state"]
            bucket["sample_reasoning"] = entry["metadata"]["reasoning"]
    
    # Compute value range stats
    for reg, bucket in reg_data.items():
        vals = bucket["values"]
        if vals:
            bucket["value_min"] = round(min(vals), 3)
            bucket["value_max"] = round(max(vals), 3)
            bucket["value_mean"] = round(sum(vals) / len(vals), 3)
        else:
            bucket["value_min"] = None
            bucket["value_max"] = None
            bucket["value_mean"] = None
    
    return {"entries_count": len(entries), "registers": dict(reg_data)}


# ============================================================
# CODE ASSIGNMENT
# ============================================================

def build_modbus_codes(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Convert raw analysis into the lookup-friendly modbus_codes.json structure."""
    # Sort registers alphabetically for stable code assignment
    sorted_regs = sorted(analysis["registers"].keys())
    
    codes_by_id: Dict[str, Dict[str, Any]] = {}
    code_for_register: Dict[str, str] = {}
    
    for idx, reg in enumerate(sorted_regs, start=1):
        code = f"MB{idx:03d}"
        info = analysis["registers"][reg]
        
        # Pick a representative function code (FC06 if available, else first)
        fcs = sorted(info["function_codes"])
        primary_fc = "FC06" if "FC06" in fcs else (fcs[0] if fcs else "FC06")
        
        # Pick a representative unit (skip empty)
        units = sorted(u for u in info["units"] if u)
        primary_unit = units[0] if units else ""
        
        # Pick a representative address
        addresses = sorted(a for a in info["addresses"] if a)
        primary_addr = addresses[0] if addresses else "40000"
        
        codes_by_id[code] = {
            "code": code,
            "register": reg,
            "function_code": primary_fc,
            "function_name": ("Write Multiple Registers" if primary_fc == "FC16"
                              else "Write Single Coil" if primary_fc == "FC05"
                              else "Read Holding Registers" if primary_fc == "FC03"
                              else "Write Single Register"),
            "address": primary_addr,
            "unit": primary_unit,
            "value_range": {
                "min": info["value_min"],
                "max": info["value_max"],
                "mean": info["value_mean"],
            },
            "label_distribution": dict(info["label_counts"]),
            "scenario_count": len(info["entry_ids"]),
            "scenario_entry_ids": info["entry_ids"],  # full list for scenario lookup
            "sample_plant_state": info["sample_plant_state"],
            "sample_reasoning": info["sample_reasoning"],
        }
        code_for_register[reg] = code
    
    # Build alias -> register lookup (only keep aliases that match a real register)
    alias_to_register: Dict[str, str] = {}
    unmatched_aliases: List[str] = []
    for alias, reg in FRIENDLY_ALIASES.items():
        if reg in code_for_register:
            alias_to_register[alias] = reg
        else:
            unmatched_aliases.append(alias)
    
    # Also auto-add the canonical register name itself as a self-alias
    for reg in code_for_register:
        alias_to_register[reg.lower()] = reg
    
    return {
        "version": "1.0.0",
        "total_codes": len(codes_by_id),
        "total_aliases": len(alias_to_register),
        "codes_by_id": codes_by_id,
        "code_for_register": code_for_register,
        "alias_to_register": alias_to_register,
        "unmatched_aliases": unmatched_aliases,
    }


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract unique Modbus commands from SentinelEdge dataset")
    parser.add_argument("--dataset", default="sentineledge_dataset.json",
                        help="Path to sentineledge_dataset.json")
    parser.add_argument("--output", default="modbus_codes.json",
                        help="Output JSON file")
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset):
        print(f"[!] Dataset not found: {args.dataset}")
        return 1
    
    print("=" * 65)
    print(" SentinelEdge Modbus Command Mapper")
    print("=" * 65)
    print(f"  Dataset: {args.dataset}")
    print(f"  Output:  {args.output}")
    print("=" * 65)
    
    analysis = analyze_dataset(args.dataset)
    print(f"\n[+] Found {len(analysis['registers'])} unique registers")
    
    codes = build_modbus_codes(analysis)
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(codes, f, indent=2, ensure_ascii=False)
    
    size_kb = os.path.getsize(args.output) / 1024
    print(f"\n[+] Wrote {args.output} ({size_kb:.1f} KB)")
    print(f"    Total codes:    {codes['total_codes']}")
    print(f"    Total aliases:  {codes['total_aliases']}")
    if codes["unmatched_aliases"]:
        print(f"    Unused aliases: {codes['unmatched_aliases']}")
    
    print(f"\n[+] First 10 codes:")
    print(f"    {'Code':<8} {'Register':<40} {'FC':<6} {'Unit':<10} {'Range'}")
    print(f"    {'-'*8} {'-'*40} {'-'*6} {'-'*10} {'-'*30}")
    for i, (code, info) in enumerate(list(codes["codes_by_id"].items())[:10]):
        rng = ""
        if info["value_range"]["min"] is not None:
            rng = f"{info['value_range']['min']}-{info['value_range']['max']}"
        print(f"    {code:<8} {info['register']:<40} "
              f"{info['function_code']:<6} {info['unit']:<10} {rng}")
    
    print(f"\n[+] Sample alias lookups:")
    for alias in ["chlorine", "alum", "pump", "ph", "ozone", "uv", "shutdown"]:
        reg = codes["alias_to_register"].get(alias, "(no match)")
        code = codes["code_for_register"].get(reg, "?") if reg != "(no match)" else "?"
        print(f"    '{alias}' -> {reg} -> {code}")
    
    print()
    print("[+] Ready. Run 08_inference_firestore.py to use this mapping.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
