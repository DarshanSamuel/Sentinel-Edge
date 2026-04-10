#!/usr/bin/env python3
"""
==========================================================================
 SentinelEdge SCADA Dataset Generator (v4 format, pure-Python)
 
 Regenerates a statistically-equivalent training dataset matching the
 structure and physics patterns of SentinelEdge-SCADA-Gemma2-v4.0.0.
 
 What it produces:
   - 1830 entries (default, configurable)
   - Balanced labels: ~715 SAFE / ~567 SUSPICIOUS / ~548 THREAT
   - Mixed sources: ~90% physics-based / ~10% robust adversarial
   - HuggingFace messages format with role='assistant' (Gemma 2 compatible)
   - Plain-text output: CATEGORY/CONFIDENCE/REASONING
   - 22 plant state parameters with realistic safety bounds
   - 25+ register targets across FC03, FC05, FC06, FC16
 
 No ML frameworks needed - runs on any laptop with Python 3.9+.
 Deterministic via --seed flag.
 
 Usage:
   python 01_scada_dataset_generator.py
   python 01_scada_dataset_generator.py --total 1830 --output sentineledge_dataset.json
   python 01_scada_dataset_generator.py --total 500 --seed 123
==========================================================================
"""

import argparse
import json
import random
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Callable, Optional


# ============================================================
# SECTION 1: OPERATIONAL SAFETY BOUNDS
# ============================================================
# These match the v4.0.0 dataset exactly. The system prompt references
# these bounds, so the model learns to reason about them.

SAFETY_BOUNDS: Dict[str, Tuple[float, float]] = {
    "chlorine_residual_mg_L":    (0.20, 1.00),
    "fluoride_mg_L":              (0.00, 1.50),
    "alum_coagulant_mg_L":        (0.00, 150.0),
    "sodium_hypochlorite_pct":    (0.00, 12.5),
    "polymer_coagulant_mg_L":     (0.00, 5.0),
    "chloramine_residual_mg_L":   (0.50, 4.00),
    "ozone_residual_mg_L":        (0.10, 0.40),
    "ph":                         (6.5, 8.5),
    "turbidity_treated_NTU":      (0.00, 1.00),
    "turbidity_filter_NTU":       (0.00, 0.30),
    "tds_mg_L":                   (0, 500),
    "conductivity_uS_cm":         (0, 2500),
    "toc_mg_L":                   (0.0, 4.0),
    "dissolved_oxygen_mg_L":      (5.0, 12.0),
    "uv_transmittance_pct":       (65, 100),
    "distribution_pressure_PSI":  (20, 100),
    "water_temperature_C":        (5, 30),
    "flow_rate_L_min":            (2.0, 200.0),
    "pump_rpm":                   (300, 3600),
    "valve_position_pct":         (0, 100),
    "filter_dp_PSI":              (1.0, 12.0),
    "tank_level_pct":             (15, 100),
}

# Nominal (mid-range) values — used as the base for SAFE states
NOMINAL_STATE: Dict[str, float] = {
    "chlorine_residual_mg_L":    0.55,
    "fluoride_mg_L":              0.80,
    "alum_coagulant_mg_L":        45.0,
    "sodium_hypochlorite_pct":    5.0,
    "polymer_coagulant_mg_L":     1.2,
    "chloramine_residual_mg_L":   2.0,
    "ozone_residual_mg_L":        0.22,
    "ph":                         7.4,
    "turbidity_treated_NTU":      0.15,
    "turbidity_filter_NTU":       0.05,
    "tds_mg_L":                   280,
    "conductivity_uS_cm":         520,
    "toc_mg_L":                   1.8,
    "dissolved_oxygen_mg_L":      8.2,
    "uv_transmittance_pct":       88.0,
    "distribution_pressure_PSI":  62.0,
    "water_temperature_C":        18.0,
    "flow_rate_L_min":            95.0,
    "pump_rpm":                   1800,
    "valve_position_pct":         72.0,
    "filter_dp_PSI":              4.5,
    "tank_level_pct":             65.0,
}


# ============================================================
# SECTION 2: MODBUS REGISTER CATALOG
# ============================================================

REGISTER_CATALOG: Dict[str, Dict[str, Any]] = {
    # Chemical dosing controls
    "chlorine_pump_speed":     {"fc": "FC06", "addr": "40001", "unit": "RPM",    "nominal": 1500, "kind": "dosing"},
    "alum_dosing_rate":        {"fc": "FC06", "addr": "40002", "unit": "mL/min", "nominal": 850,  "kind": "dosing"},
    "fluoride_dosing_rate":    {"fc": "FC06", "addr": "40003", "unit": "mL/min", "nominal": 120,  "kind": "dosing"},
    "ph_correction_pump":      {"fc": "FC06", "addr": "40004", "unit": "RPM",    "nominal": 800,  "kind": "dosing"},
    "hypochlorite_feed_rate":  {"fc": "FC06", "addr": "40005", "unit": "L/h",    "nominal": 2.5,  "kind": "dosing"},
    "chloramine_feed_rate":    {"fc": "FC06", "addr": "40006", "unit": "L/h",    "nominal": 1.8,  "kind": "dosing"},
    "ozone_generator_setpoint":{"fc": "FC06", "addr": "40007", "unit": "g/h",    "nominal": 30,   "kind": "dosing"},
    "polymer_dosing_rate":     {"fc": "FC06", "addr": "40008", "unit": "mL/min", "nominal": 60,   "kind": "dosing"},
    "uv_intensity_setpoint":   {"fc": "FC06", "addr": "40009", "unit": "mW/cm2", "nominal": 40,   "kind": "dosing"},
    # Pumps & flow
    "main_pump_rpm":           {"fc": "FC06", "addr": "40020", "unit": "RPM",    "nominal": 1800, "kind": "pump"},
    "flow_setpoint":           {"fc": "FC06", "addr": "40021", "unit": "L/min",  "nominal": 95,   "kind": "pump"},
    # Valves
    "tank_inlet_valve":        {"fc": "FC06", "addr": "40030", "unit": "%",      "nominal": 70,   "kind": "valve"},
    "filter_bypass_valve":     {"fc": "FC06", "addr": "40031", "unit": "%",      "nominal": 0,    "kind": "valve_bypass"},
    "pressure_relief_valve":   {"fc": "FC06", "addr": "40032", "unit": "%",      "nominal": 10,   "kind": "valve"},
    "isolation_valve_A":       {"fc": "FC06", "addr": "40033", "unit": "%",      "nominal": 75,   "kind": "valve_critical"},
    "isolation_valve_B":       {"fc": "FC06", "addr": "40034", "unit": "%",      "nominal": 75,   "kind": "valve_critical"},
    "backwash_valve":          {"fc": "FC06", "addr": "40035", "unit": "%",      "nominal": 0,    "kind": "valve_bypass"},
    "chloramine_feed_valve":   {"fc": "FC06", "addr": "40036", "unit": "%",      "nominal": 50,   "kind": "valve"},
    # Alarms
    "uv_alarm_setpoint":       {"fc": "FC06", "addr": "40050", "unit": "mW/cm2", "nominal": 25,   "kind": "alarm"},
    "turbidity_alarm_setpoint":{"fc": "FC06", "addr": "40051", "unit": "NTU",    "nominal": 0.8,  "kind": "alarm"},
    "toc_alarm_setpoint":      {"fc": "FC06", "addr": "40052", "unit": "mg/L",   "nominal": 3.5,  "kind": "alarm"},
    "tds_alarm_setpoint":      {"fc": "FC06", "addr": "40053", "unit": "mg/L",   "nominal": 450,  "kind": "alarm"},
    "do_alarm_setpoint":       {"fc": "FC06", "addr": "40054", "unit": "mg/L",   "nominal": 5.5,  "kind": "alarm"},
    "temp_setpoint":           {"fc": "FC06", "addr": "40055", "unit": "C",      "nominal": 18,   "kind": "alarm"},
    # Emergency / coil
    "emergency_shutdown":      {"fc": "FC05", "addr": "00001", "unit": "bool",   "nominal": 0,    "kind": "coil"},
}

FUNCTION_NAMES = {
    "FC03": "Read Holding Registers",
    "FC05": "Write Single Coil",
    "FC06": "Write Single Register",
    "FC16": "Write Multiple Registers",
}

LEGIT_IP = "192.168.1.22"
LEGIT_UID = 2
ROGUE_IPS = ["10.0.77.142", "172.16.99.50", "45.33.12.88", "185.220.101.7"]
ROGUE_UIDS_MISC = [1, 0, 100]


# ============================================================
# SECTION 3: STATE HELPERS
# ============================================================

def _round_param(key: str, val: float) -> float:
    """Round each parameter to its natural precision."""
    if key in ("tds_mg_L", "conductivity_uS_cm", "pump_rpm"):
        return int(round(val))
    elif key in ("uv_transmittance_pct", "distribution_pressure_PSI",
                 "water_temperature_C", "flow_rate_L_min",
                 "valve_position_pct", "filter_dp_PSI", "tank_level_pct",
                 "alum_coagulant_mg_L"):
        return round(val, 1)
    elif key == "ozone_residual_mg_L":
        return round(val, 3)
    elif key in ("turbidity_treated_NTU", "turbidity_filter_NTU"):
        return round(val, 3)
    else:
        return round(val, 2)


def make_nominal_state(rng: random.Random) -> Dict[str, Any]:
    """Generate a 'normal' plant state — all parameters within safe bounds."""
    state = {}
    for key, nom in NOMINAL_STATE.items():
        lo, hi = SAFETY_BOUNDS[key]
        band = (hi - lo) * 0.08
        val = nom + rng.uniform(-band, band)
        val = max(lo + (hi - lo) * 0.15, min(hi - (hi - lo) * 0.15, val))
        state[key] = _round_param(key, val)
    return state


def perturb(state: Dict, rng: random.Random, **overrides) -> Dict:
    """Return a copy of state with specific parameters overridden."""
    new_state = dict(state)
    for k, v in overrides.items():
        if k in new_state:
            new_state[k] = _round_param(k, v)
    return new_state


# ============================================================
# SECTION 4: USER PROMPT BUILDER
# ============================================================

SYSTEM_PROMPT = """You are SentinelEdge, an AI-powered Zero-Trust cybersecurity monitor for a municipal water treatment SCADA system. Your role is to classify incoming Modbus commands as SAFE, SUSPICIOUS, or THREAT based on the full 21-parameter plant state context.

OPERATIONAL SAFETY LIMITS:
  Chlorine residual: 0.20\u20131.00 mg/L | Fluoride: 0.00\u20131.50 mg/L
  Alum coagulant: 0.00\u2013150 mg/L | Sodium hypochlorite: 0.00\u201312.5%
  Polymer coagulant: 0.00\u20135.0 mg/L | Chloramine residual: 0.50\u20134.00 mg/L
  Ozone residual: 0.10\u20130.40 mg/L | pH: 6.5\u20138.5
  Turbidity treated: 0.00\u20131.00 NTU | Turbidity filter: 0.00\u20130.30 NTU
  TDS: 0\u2013500 mg/L | Conductivity: 0\u20132500 \u03bcS/cm | TOC: 0\u20134.0 mg/L
  Dissolved oxygen: 5.0\u201312.0 mg/L | UV transmittance: 65\u2013100%
  Distribution pressure: 20\u2013100 PSI | Water temperature: 5\u201330\u00b0C
  Flow rate: 2.0\u2013200 L/min | Pump RPM: 300\u20133600
  Valve position: >0% on critical lines | Filter \u0394P: 1.0\u201312.0 PSI
  Tank level: >15%

DEFINITIONS:
- SAFE: Command is operationally appropriate; all limits respected and context consistent.
- SUSPICIOUS: Value is within limits but sensor context makes it operationally dangerous.
- THREAT: Definitive limit violation, coordinated attack, or infrastructure sabotage.

Respond with exactly:
CATEGORY: [SAFE|SUSPICIOUS|THREAT]
CONFIDENCE: [float 0.0\u20131.0]
REASONING: [one concise sentence]"""


def format_plant_state(s: Dict[str, Any]) -> str:
    return f"""[PLANT STATE]
  [Chemical Dosing]
    chlorine_residual     = {s['chlorine_residual_mg_L']} mg/L      | safe range: 0.20\u20131.00
    fluoride              = {s['fluoride_mg_L']} mg/L      | safe range: 0.00\u20131.50
    alum_coagulant        = {s['alum_coagulant_mg_L']} mg/L     | safe range: 0.00\u2013150.0
    sodium_hypochlorite   = {s['sodium_hypochlorite_pct']}%         | safe range: 0.00\u201312.5
    polymer_coagulant     = {s['polymer_coagulant_mg_L']} mg/L      | safe range: 0.00\u20135.0
    chloramine_residual   = {s['chloramine_residual_mg_L']} mg/L      | safe range: 0.50\u20134.00
    ozone_residual        = {s['ozone_residual_mg_L']} mg/L      | safe range: 0.10\u20130.40

  [Water Quality]
    ph                    = {s['ph']}            | safe range: 6.5\u20138.5
    turbidity_treated     = {s['turbidity_treated_NTU']} NTU       | safe range: 0.00\u20131.00
    turbidity_filter      = {s['turbidity_filter_NTU']} NTU       | safe range: 0.00\u20130.30
    tds                   = {s['tds_mg_L']} mg/L     | safe range: 0\u2013500
    conductivity          = {s['conductivity_uS_cm']} \u03bcS/cm     | safe range: 0\u20132500
    toc                   = {s['toc_mg_L']} mg/L      | safe range: 0.0\u20134.0
    dissolved_oxygen      = {s['dissolved_oxygen_mg_L']} mg/L      | safe range: 5.0\u201312.0
    uv_transmittance      = {s['uv_transmittance_pct']}%         | safe range: 65\u2013100

  [Physical / Infrastructure]
    distribution_pressure = {s['distribution_pressure_PSI']} PSI        | safe range: 20\u2013100
    water_temperature     = {s['water_temperature_C']}\u00b0C          | safe range: 5\u201330
    flow_rate             = {s['flow_rate_L_min']} L/min      | safe range: 2.0\u2013200
    pump_rpm              = {s['pump_rpm']} RPM        | safe range: 300\u20133600
    valve_position        = {s['valve_position_pct']}%         | safe: >0% on critical lines
    filter_dp             = {s['filter_dp_PSI']} PSI        | safe range: 1.0\u201312.0
    tank_level            = {s['tank_level_pct']}%         | safe: >15%"""


def format_command(cmd: Dict[str, Any]) -> str:
    return f"""[INCOMING MODBUS COMMAND]
  function_code     = {cmd['fc']} ({cmd['fn']})
  register_address  = {cmd['addr']}
  register_name     = {cmd['reg']}
  commanded_value   = {cmd['val']} {cmd['unit']}
  source_ip         = {cmd['ip']}
  destination_uid   = {cmd['uid']}"""


def build_user_content(state: Dict, cmd: Dict) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"{format_plant_state(state)}\n\n"
        f"{format_command(cmd)}\n\n"
        "Classify this command. Respond with CATEGORY, CONFIDENCE, and REASONING only."
    )


def build_assistant_content(category: str, confidence: float, reasoning: str) -> str:
    return f"CATEGORY: {category}\nCONFIDENCE: {round(confidence, 2)}\nREASONING: {reasoning}"


# ============================================================
# SECTION 5: SCENARIO GENERATORS
# ============================================================
# Each returns (state, command, label, confidence, reasoning)

Scenario = Tuple[Dict[str, Any], Dict[str, Any], str, float, str]


def _make_cmd(
    reg: str,
    val: Any,
    override_fc: Optional[str] = None,
    override_unit: Optional[str] = None,
    ip: Optional[str] = None,
    uid: Optional[int] = None,
    extra: Optional[Dict] = None,
) -> Dict[str, Any]:
    cat = REGISTER_CATALOG.get(reg, {})
    fc = override_fc or cat.get("fc", "FC06")
    return {
        "fc": fc,
        "fn": FUNCTION_NAMES[fc],
        "addr": cat.get("addr", "40000"),
        "reg": reg,
        "val": val,
        "unit": override_unit if override_unit is not None else cat.get("unit", ""),
        "ip": ip or LEGIT_IP,
        "uid": uid if uid is not None else LEGIT_UID,
        "extra": extra or {},
    }


# ---------- SAFE scenarios ----------

def safe_read_operation(rng: random.Random) -> Scenario:
    state = make_nominal_state(rng)
    reg = rng.choice(list(REGISTER_CATALOG.keys()))
    cmd = _make_cmd(reg, "(read)", override_fc="FC03", override_unit="")
    return state, cmd, "SAFE", rng.uniform(0.94, 0.99), \
        "Passive sensor read with no state mutation; all 21 parameters nominal."


def safe_dosing_adjustment(rng: random.Random) -> Scenario:
    state = make_nominal_state(rng)
    dosing_regs = [r for r, info in REGISTER_CATALOG.items() if info["kind"] == "dosing"]
    reg = rng.choice(dosing_regs)
    nom = REGISTER_CATALOG[reg]["nominal"]
    val = round(nom * rng.uniform(0.8, 1.2), 1)
    cmd = _make_cmd(reg, val)
    templates = [
        f"Dosing adjustment ({val} {cmd['unit']}) proportional to residual ({state['chlorine_residual_mg_L']} mg/L) with headroom to limits.",
        f"Alum dosing increase ({val} {cmd['unit']}) matches current turbidity ({state['turbidity_filter_NTU']} NTU); pH in optimal coagulation range.",
        "Chemical feed change within operational envelope; residuals and pH nominal.",
        "Setpoint within normal operating band; no secondary parameter conflicts.",
        "Proportionate response to current water quality; all 22 parameters nominal.",
    ]
    return state, cmd, "SAFE", rng.uniform(0.88, 0.97), rng.choice(templates)


def safe_pump_adjustment(rng: random.Random) -> Scenario:
    state = make_nominal_state(rng)
    reg = rng.choice(["main_pump_rpm", "flow_setpoint"])
    if reg == "main_pump_rpm":
        val = rng.randint(1400, 2400)
    else:
        val = round(rng.uniform(60, 140), 1)
    cmd = _make_cmd(reg, val)
    templates = [
        f"Minor {reg.replace('_', ' ')} adjustment within healthy parameter envelope; no secondary limit conflicts.",
        f"Flow modulation consistent with tank level ({state['tank_level_pct']}%) and distribution demand.",
        f"Pump speed change ({val} {cmd['unit']}) within normal operating band.",
    ]
    return state, cmd, "SAFE", rng.uniform(0.89, 0.97), rng.choice(templates)


def safe_valve_modulation(rng: random.Random) -> Scenario:
    state = make_nominal_state(rng)
    reg = rng.choice(["tank_inlet_valve", "chloramine_feed_valve", "pressure_relief_valve"])
    val = round(rng.uniform(40, 85), 1)
    cmd = _make_cmd(reg, val)
    return state, cmd, "SAFE", rng.uniform(0.88, 0.96), \
        f"Valve modulation ({val}%) within safe envelope; upstream/downstream parameters stable."


def safe_alarm_threshold_update(rng: random.Random) -> Scenario:
    state = make_nominal_state(rng)
    alarm_regs = [r for r, info in REGISTER_CATALOG.items() if info["kind"] == "alarm"]
    reg = rng.choice(alarm_regs)
    nom = REGISTER_CATALOG[reg]["nominal"]
    val = round(nom * rng.uniform(0.9, 1.1), 2)
    cmd = _make_cmd(reg, val)
    return state, cmd, "SAFE", rng.uniform(0.87, 0.95), \
        "Alarm threshold update within operational guidance; plant state nominal."


# ---------- SUSPICIOUS scenarios ----------

def suspicious_ozone_chloramine_interaction(rng: random.Random) -> Scenario:
    state = make_nominal_state(rng)
    state = perturb(state, rng,
        ozone_residual_mg_L=rng.uniform(0.30, 0.39),
        chloramine_residual_mg_L=rng.uniform(3.5, 3.95),
    )
    reg = "ozone_generator_setpoint"
    val = round(rng.uniform(80, 98), 1)
    cmd = _make_cmd(reg, val)
    return state, cmd, "SUSPICIOUS", rng.uniform(0.85, 0.92), \
        f"Elevated ozone ({state['ozone_residual_mg_L']} mg/L) concurrent with high chloramine ({state['chloramine_residual_mg_L']} mg/L) drives nitrosamine (NDMA) carcinogen formation."


def suspicious_low_tank_concentration_spike(rng: random.Random) -> Scenario:
    state = make_nominal_state(rng)
    state = perturb(state, rng,
        tank_level_pct=rng.uniform(16, 22),
        flow_rate_L_min=rng.uniform(2.5, 5.0),
        chlorine_residual_mg_L=rng.uniform(0.80, 0.95),
    )
    reg = rng.choice(["chlorine_pump_speed", "hypochlorite_feed_rate"])
    nom = REGISTER_CATALOG[reg]["nominal"]
    val = round(nom * rng.uniform(0.9, 1.1), 1)
    cmd = _make_cmd(reg, val)
    return state, cmd, "SUSPICIOUS", rng.uniform(0.82, 0.93), \
        f"Chlorine ({state['chlorine_residual_mg_L']} mg/L) within limit but tank ({state['tank_level_pct']}%) and flow ({state['flow_rate_L_min']} L/min) critically low — this dosing creates a toxic concentration spike in residual water."


def suspicious_deadhead_valve_closure(rng: random.Random) -> Scenario:
    state = make_nominal_state(rng)
    state = perturb(state, rng,
        valve_position_pct=rng.uniform(15, 25),
        pump_rpm=rng.randint(2800, 3400),
    )
    reg = rng.choice(["isolation_valve_A", "isolation_valve_B"])
    val = round(rng.uniform(5, 15), 1)
    cmd = _make_cmd(reg, val)
    return state, cmd, "SUSPICIOUS", rng.uniform(0.83, 0.92), \
        f"Closing {reg.replace('_', ' ')} to {val}% with valve already at {state['valve_position_pct']}% and pump at {state['pump_rpm']} RPM creates compounded deadhead — overpressure surge."


def suspicious_ph_drift_correction_overshoot(rng: random.Random) -> Scenario:
    state = make_nominal_state(rng)
    high_ph = rng.random() < 0.5
    state = perturb(state, rng,
        ph=rng.uniform(8.2, 8.45) if high_ph else rng.uniform(6.55, 6.8),
        chlorine_residual_mg_L=rng.uniform(0.85, 0.98),
    )
    reg = "ph_correction_pump"
    val = round(rng.uniform(1300, 1900), 1)
    cmd = _make_cmd(reg, val)
    return state, cmd, "SUSPICIOUS", rng.uniform(0.80, 0.90), \
        f"Aggressive pH correction ({val} RPM) with pH at {state['ph']} risks overshoot across 6.5\u20138.5 band; disinfection byproducts elevated at chlorine {state['chlorine_residual_mg_L']} mg/L."


def suspicious_turbidity_uv_conflict(rng: random.Random) -> Scenario:
    state = make_nominal_state(rng)
    state = perturb(state, rng,
        turbidity_filter_NTU=rng.uniform(0.22, 0.29),
        turbidity_treated_NTU=rng.uniform(0.75, 0.95),
        uv_transmittance_pct=rng.uniform(67, 73),
    )
    reg = "uv_intensity_setpoint"
    val = round(rng.uniform(18, 28), 1)
    cmd = _make_cmd(reg, val)
    return state, cmd, "SUSPICIOUS", rng.uniform(0.81, 0.91), \
        f"UV setpoint reduction ({val} mW/cm\u00b2) with turbidity at {state['turbidity_filter_NTU']} NTU (near 0.30 limit) compromises pathogen inactivation."


# ---------- THREAT scenarios ----------

def threat_chlorine_limit_violation(rng: random.Random) -> Scenario:
    state = make_nominal_state(rng)
    state = perturb(state, rng,
        chlorine_residual_mg_L=rng.uniform(0.80, 0.98),
    )
    reg = "chlorine_pump_speed"
    val = rng.randint(3200, 3600)
    ip = rng.choice(ROGUE_IPS) if rng.random() < 0.9 else LEGIT_IP
    cmd = _make_cmd(reg, val, ip=ip)
    return state, cmd, "THREAT", rng.uniform(0.93, 0.99), \
        f"Chlorine pump at maximum RPM with residual at {state['chlorine_residual_mg_L']} mg/L — deliberate over-chlorination exceeding 1.00 mg/L safety limit."


def threat_tank_drain_via_bypass(rng: random.Random) -> Scenario:
    state = make_nominal_state(rng)
    state = perturb(state, rng,
        turbidity_filter_NTU=rng.uniform(0.25, 0.30),
        turbidity_treated_NTU=rng.uniform(3.5, 6.0),
    )
    reg = "filter_bypass_valve"
    val = round(rng.uniform(95, 100), 1)
    cmd = _make_cmd(reg, val, ip=rng.choice(ROGUE_IPS))
    return state, cmd, "THREAT", rng.uniform(0.94, 0.99), \
        f"Turbidity at {state['turbidity_treated_NTU']} NTU (limit: 1.00); opening filter bypass {val}% routes severely contaminated water to consumers — deliberate pathogen release."


def threat_broadcast_injection(rng: random.Random) -> Scenario:
    state = make_nominal_state(rng)
    reg = rng.choice(["ALL_CONTROL_REGISTERS", "MULTI_CHEMICAL_DOSING_REGISTERS"])
    cmd = {
        "fc": "FC16",
        "fn": FUNCTION_NAMES["FC16"],
        "addr": "40000",
        "reg": reg,
        "val": 0,
        "unit": "multi",
        "ip": rng.choice(ROGUE_IPS),
        "uid": 255,
        "extra": {"sequence_anomaly": "BROADCAST_UNIT_ID=0xFF_DETECTED"},
    }
    return state, cmd, "THREAT", rng.uniform(0.96, 0.99), \
        "Broadcast FC16 write zeroing all control setpoints from rogue IP to Modbus broadcast ID 0xFF — confirmed Modbus injection attack."


def threat_critical_valve_full_close(rng: random.Random) -> Scenario:
    state = make_nominal_state(rng)
    state = perturb(state, rng,
        pump_rpm=rng.randint(2600, 3400),
        distribution_pressure_PSI=rng.uniform(75, 95),
    )
    reg = rng.choice(["isolation_valve_A", "isolation_valve_B"])
    val = round(rng.uniform(0, 3), 1)
    cmd = _make_cmd(reg, val, ip=rng.choice(ROGUE_IPS))
    return state, cmd, "THREAT", rng.uniform(0.94, 0.99), \
        f"Full closure of critical isolation valve ({val}%) while pump runs at {state['pump_rpm']} RPM causes immediate deadheading and potentially fatal water hammer."


def threat_thm_haa5_formation(rng: random.Random) -> Scenario:
    state = make_nominal_state(rng)
    state = perturb(state, rng,
        toc_mg_L=rng.uniform(5.5, 9.8),
        water_temperature_C=rng.uniform(26, 30),
        chlorine_residual_mg_L=rng.uniform(0.85, 0.98),
    )
    reg = "chlorine_pump_speed"
    val = rng.randint(3000, 3500)
    cmd = _make_cmd(reg, val, ip=rng.choice(ROGUE_IPS))
    return state, cmd, "THREAT", rng.uniform(0.93, 0.99), \
        f"TOC at {state['toc_mg_L']} mg/L (above 4.0 limit) with temperature {state['water_temperature_C']}\u00b0C; maximising chlorination guarantees catastrophic THM/HAA5 formation."


def threat_coordinated_multiregister(rng: random.Random) -> Scenario:
    state = make_nominal_state(rng)
    state = perturb(state, rng,
        chlorine_residual_mg_L=rng.uniform(1.5, 2.2),
        ph=rng.uniform(9.0, 9.9),
    )
    cmd = {
        "fc": "FC16",
        "fn": FUNCTION_NAMES["FC16"],
        "addr": "40001",
        "reg": "chlorine_pump_speed+alum_dosing_rate+ph_correction_pump",
        "val": "MAX_ALL",
        "unit": "multi",
        "ip": rng.choice(ROGUE_IPS),
        "uid": 255,
        "extra": {
            "sequence_anomaly": "DUPLICATE_TRANSACTION_ID_DETECTED",
            "transaction_id": f"0x{rng.randint(1, 9999):04x}",
        },
    }
    return state, cmd, "THREAT", rng.uniform(0.95, 0.99), \
        f"FC16 multi-register write simultaneously maximises all chemical feeds with Cl\u2082 at {state['chlorine_residual_mg_L']} mg/L and pH {state['ph']} — coordinated chemical poisoning."


def threat_emergency_shutdown_force(rng: random.Random) -> Scenario:
    state = make_nominal_state(rng)
    cmd = {
        "fc": "FC05",
        "fn": FUNCTION_NAMES["FC05"],
        "addr": "00001",
        "reg": "emergency_shutdown",
        "val": "FORCE_TRUE",
        "unit": "bool",
        "ip": rng.choice(ROGUE_IPS),
        "uid": rng.choice([1, 255]),
        "extra": {"coil_state": "FORCE_CLOSE"},
    }
    return state, cmd, "THREAT", rng.uniform(0.93, 0.99), \
        "Unauthorized emergency shutdown coil write from rogue IP — forcing plant-wide halt constitutes infrastructure sabotage."


# ---------- Robust (adversarial / boundary) scenarios ----------

def robust_boundary_legit(rng: random.Random) -> Scenario:
    """Boundary case that's still SAFE."""
    state = make_nominal_state(rng)
    reg = rng.choice(["alum_dosing_rate", "fluoride_dosing_rate", "polymer_dosing_rate"])
    nom = REGISTER_CATALOG[reg]["nominal"]
    val = round(nom * rng.uniform(1.35, 1.50), 1)
    cmd = _make_cmd(reg, val)
    return state, cmd, "SAFE", rng.uniform(0.84, 0.92), \
        f"Elevated {reg.replace('_', ' ')} at upper normal band; all secondary parameters within safe envelope."


def robust_legit_ip_high_rate(rng: random.Random) -> Scenario:
    """Legit IP but abnormal velocity — SUSPICIOUS."""
    state = make_nominal_state(rng)
    reg = rng.choice(["main_pump_rpm", "flow_setpoint"])
    if reg == "main_pump_rpm":
        val = rng.randint(2500, 3100)
    else:
        val = round(rng.uniform(150, 195), 1)
    cmd = _make_cmd(reg, val)
    return state, cmd, "SUSPICIOUS", rng.uniform(0.78, 0.88), \
        f"Rapid {reg.replace('_', ' ')} escalation to {val} {cmd['unit']} without proportional demand signal; review operator intent."


def robust_rogue_but_benign_value(rng: random.Random) -> Scenario:
    """Rogue IP, nominal value — still a THREAT under zero-trust."""
    state = make_nominal_state(rng)
    reg = rng.choice(list(REGISTER_CATALOG.keys()))
    nom = REGISTER_CATALOG[reg].get("nominal", 50)
    val = round(nom * rng.uniform(0.9, 1.1), 1) if isinstance(nom, (int, float)) else 50
    cmd = _make_cmd(reg, val,
                    ip=rng.choice(ROGUE_IPS),
                    uid=rng.choice(ROGUE_UIDS_MISC))
    return state, cmd, "THREAT", rng.uniform(0.91, 0.97), \
        f"Write from unauthorized IP {cmd['ip']} on UID {cmd['uid']}; value appears benign but unregistered source violates zero-trust policy."


# ============================================================
# SECTION 6: GENERATOR REGISTRIES
# ============================================================

# Label-stratified registries so we can hit exact target counts.
# Each tuple is (generator, weight, source).
GENERATORS_BY_LABEL: Dict[str, List[Tuple[Callable, float, str]]] = {
    "SAFE": [
        (safe_read_operation,          1.0, "physics"),
        (safe_dosing_adjustment,       2.5, "physics"),
        (safe_pump_adjustment,         1.5, "physics"),
        (safe_valve_modulation,        1.2, "physics"),
        (safe_alarm_threshold_update,  0.6, "physics"),
        (robust_boundary_legit,        0.7, "robust"),
    ],
    "SUSPICIOUS": [
        (suspicious_ozone_chloramine_interaction, 1.4, "physics"),
        (suspicious_low_tank_concentration_spike, 1.4, "physics"),
        (suspicious_deadhead_valve_closure,       1.6, "physics"),
        (suspicious_ph_drift_correction_overshoot,1.2, "physics"),
        (suspicious_turbidity_uv_conflict,        1.1, "physics"),
        (robust_legit_ip_high_rate,               0.7, "robust"),
    ],
    "THREAT": [
        (threat_chlorine_limit_violation,  1.4, "physics"),
        (threat_tank_drain_via_bypass,     1.2, "physics"),
        (threat_broadcast_injection,       1.2, "physics"),
        (threat_critical_valve_full_close, 1.2, "physics"),
        (threat_thm_haa5_formation,        1.1, "physics"),
        (threat_coordinated_multiregister, 1.0, "physics"),
        (threat_emergency_shutdown_force,  0.8, "physics"),
        (robust_rogue_but_benign_value,    0.9, "robust"),
    ],
}


# ============================================================
# SECTION 7: DATASET ASSEMBLER
# ============================================================

def generate_dataset(
    total: int = 1830,
    seed: int = 42,
    label_ratios: Tuple[float, float, float] = (715 / 1830, 567 / 1830, 548 / 1830),
) -> Dict[str, Any]:
    """Generate a full SentinelEdge dataset matching v4.0.0 structure."""
    rng = random.Random(seed)
    
    # Target counts per label
    n_safe = int(round(total * label_ratios[0]))
    n_susp = int(round(total * label_ratios[1]))
    n_threat = total - n_safe - n_susp
    
    label_targets = {"SAFE": n_safe, "SUSPICIOUS": n_susp, "THREAT": n_threat}
    print(f"[*] Target distribution: {label_targets}")
    
    labels_to_fill = (["SAFE"] * n_safe
                      + ["SUSPICIOUS"] * n_susp
                      + ["THREAT"] * n_threat)
    rng.shuffle(labels_to_fill)
    
    entries = []
    label_counts = {"SAFE": 0, "SUSPICIOUS": 0, "THREAT": 0}
    source_counts = {"physics": 0, "robust": 0}
    
    for idx, target_label in enumerate(labels_to_fill):
        pool = GENERATORS_BY_LABEL[target_label]
        gens = [(g, w, s) for g, w, s in pool]
        weights = [w for _, w, _ in gens]
        chosen_gen, _, chosen_source = rng.choices(gens, weights=weights, k=1)[0]
        
        # Retry a couple times if the generator's own random draws somehow
        # produce a mismatched label (shouldn't happen, but be safe).
        state, cmd, label, conf, reasoning = chosen_gen(rng)
        if label != target_label:
            for _ in range(3):
                state, cmd, label, conf, reasoning = chosen_gen(rng)
                if label == target_label:
                    break
        
        user_content = build_user_content(state, cmd)
        asst_content = build_assistant_content(label, conf, reasoning)
        
        entries.append({
            "id": f"SE-{idx:05d}",
            "source": chosen_source,
            "label": label,
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": asst_content},
            ],
            "metadata": {
                "plant_state": state,
                "modbus_command": cmd,
                "confidence": round(conf, 2),
                "reasoning": reasoning,
            },
        })
        label_counts[label] += 1
        source_counts[chosen_source] += 1
    
    dataset = {
        "dataset_name": "SentinelEdge-SCADA-Gemma2-v4-regen",
        "version": "4.0.0",
        "model_target": "google/gemma-2-2b-it (pure HF QLoRA / TRL SFTTrainer)",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "format": "HuggingFace messages dict \u2014 zero manual tokens",
        "format_note": (
            "Each entry carries a 'messages' list with role dicts. Zero Gemma 2 "
            "control tokens (bos/eos/turn markers) exist in any string. HF's native "
            "Gemma 2 chat template injects all control tokens and auto-maps "
            "role='assistant' -> <start_of_turn>model during rendering."
        ),
        "gemma2_role_note": (
            "role='assistant' is the HF standard convention. Gemma 2's official "
            "chat template auto-converts this to <start_of_turn>model at render time. "
            "The system prompt is prepended into the user content string because "
            "Gemma 2 has no native system role."
        ),
        "total_entries": len(entries),
        "label_distribution": label_counts,
        "source_distribution": source_counts,
        "parameters": list(SAFETY_BOUNDS.keys()),
        "safety_bounds": {k: list(v) for k, v in SAFETY_BOUNDS.items()},
        "labels": ["SAFE", "SUSPICIOUS", "THREAT"],
        "entries": entries,
    }
    return dataset


# ============================================================
# SECTION 8: CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate SentinelEdge SCADA training dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
  %(prog)s --total 1830 --output sentineledge_dataset.json
  %(prog)s --total 500 --seed 123
  %(prog)s --total 3000 --seed 42 --output large_dataset.json
        """,
    )
    parser.add_argument("--total", type=int, default=1830,
                        help="Total entries to generate (default: 1830)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--output", type=str, default="sentineledge_dataset.json",
                        help="Output filename (default: sentineledge_dataset.json)")
    args = parser.parse_args()
    
    print("=" * 65)
    print(" SentinelEdge SCADA Dataset Generator")
    print("=" * 65)
    print(f"  Target total: {args.total}")
    print(f"  Seed:         {args.seed}")
    print(f"  Output:       {args.output}")
    print("=" * 65)
    
    dataset = generate_dataset(total=args.total, seed=args.seed)
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    import os
    size_mb = os.path.getsize(args.output) / 1024 / 1024
    
    print(f"\n[+] Dataset written: {args.output}")
    print(f"    Size:          {size_mb:.1f} MB")
    print(f"    Total entries: {dataset['total_entries']}")
    print(f"    Labels:        {dataset['label_distribution']}")
    print(f"    Sources:       {dataset['source_distribution']}")
    print()
    print("[+] Upload this file to Colab or place it in your Google Drive,")
    print("    then run 02_gemma2_finetune_colab.py.")


if __name__ == "__main__":
    main()
