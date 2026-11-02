"""Diagnose (Muse Spark), with every claim checked by code.

Spark gets the peak table, the features, the rule verdicts, the spectrum image and (when there is one) the
magnified clip, and returns JSON: fault, confidence, evidence [{part, frequency_hz, order}], a plain-words
explanation, fix steps and what to check next.

verify() then checks the claim against the measurement:
- each evidence item must match a measured peak of that part within one FFT bin, and its cited order must be the
  measured peak's order; otherwise it is dropped (kept in the card, marked dropped, with the reason);
- a matched item counts as support only when its order fits the claimed fault (imbalance 1x, misalignment 2x,
  looseness 0.5x or harmonics); a real peak cited for the wrong fault is shown as context and does not count;
- the claim must be supported: imbalance needs a verified 1x, misalignment a verified 2x, looseness a verified
  0.5x or a row of MIN_HARMONICS harmonics.
The card's headline is the rule verdict. When Spark disagrees with it, any evidence was dropped, or the claim is
unsupported, Spark's confidence is capped at "low" and both are shown side by side.
"""
import re

from . import fingerprint
from .meta_api import CreditExhausted, MetaAPI, image_block

FAULTS = ("none", "imbalance", "looseness", "misalignment", "unclear")
CONFIDENCES = ("high", "medium", "low")
NEEDS = {"imbalance": "a 1x peak", "misalignment": "a 2x peak",
         "looseness": f"a 0.5x peak or a row of {fingerprint.MIN_HARMONICS} harmonics"}
SYNONYMS = [("unbalanc", "imbalance"), ("imbalanc", "imbalance"), ("misalign", "misalignment"),
            ("loose", "looseness"), ("healthy", "none"), ("no fault", "none"), ("normal", "none"),
            ("unclear", "unclear"), ("unknown", "unclear")]

SCHEMA = {
    "type": "object",
    "properties": {
        "fault": {"type": "string", "enum": list(FAULTS)},
        "confidence": {"type": "string", "enum": list(CONFIDENCES)},
        "evidence": {"type": "array", "items": {
            "type": "object",
            "properties": {"part": {"type": "string"}, "frequency_hz": {"type": "number"}, "order": {"type": "string"}},
            "required": ["part", "frequency_hz", "order"], "additionalProperties": False}},
        "explanation": {"type": "string"},
        "fix": {"type": "array", "items": {"type": "string"}},
        "check_next": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["fault", "confidence", "evidence", "explanation", "fix", "check_next"],
    "additionalProperties": False,
}

PROMPT = """You are a vibration analyst helping a maintenance technician at a small plant who has no sensors.
A phone filmed a running {machine} in slow motion. Our code measured how each part vibrates (sub-pixel tracking, camera shake removed) and ran fixed rules on the spectrum. The attached image shows each part's spectrum with peaks labelled as orders of running speed{clip_note}.

{table}

Fault signatures: imbalance shows a dominant 1x; misalignment shows a strong 2x (often with 1x); looseness shows 0.5x and/or a row of 3 or more harmonics. Light flicker is not vibration.

Return JSON:
- fault: one of none, imbalance, looseness, misalignment, unclear.
- confidence: high, medium or low.
- evidence: the peaks your diagnosis rests on, copied from the table above: part name exactly as listed, frequency_hz exactly as listed, and its order label (for example "1x"). Cite only peaks that are in the table.
- explanation: 2 to 4 plain sentences for the technician. If you disagree with the rule verdict, say why.
- fix: 2 to 5 short, practical steps.
- check_next: 1 to 3 short things to check or measure next.
Do not use em dashes."""


def _fmt_amp(p, mm):
    return f"{p['amp_mm']:.4f} mm" if mm and p.get("amp_mm") is not None else f"{p['amp_px']:.4f} px"


def peak_table(result, fp, plan=None):
    """The measurement as compact text for the prompt."""
    rs = result["running_speed"]
    mm = result["scale"]["mm_per_px"]
    lines = [f"Running speed: {rs['hz']:.3f} Hz ({rs['rpm']:.0f} rpm, from {rs['source']}).",
             f"FFT bin: {result['bin_hz']:.4f} Hz over {result['seconds']:.2f} s at {result['fps']:g} fps; "
             f"Nyquist {result['nyquist_hz']:g} Hz; light flicker lands at {result['flicker_hz']:g} Hz.",
             f"Scale: {result['scale']['basis']}." if mm else "Scale: pixels (no reference length).", "",
             "Measured peaks (part | axis | frequency_hz | order | amplitude | SNR):"]
    for name, entry in result["parts"].items():
        for ax in "xy":
            for p in entry["axes"][ax]["peaks"]:
                lines.append(f"{name} | {ax} ({'horizontal' if ax == 'x' else 'vertical'}) | {p['hz']:.3f} | "
                             f"{p['label']} | {_fmt_amp(p, mm)} | {p['snr']}")
    lines += ["", "Features per part:"]
    for name, pf in fp["parts"].items():
        f = pf["features"]
        lines.append(f"{name}: velocity {f['velocity_rms']:.2f} {f['velocity_unit']} RMS; 1x energy share "
                     f"{f['e1_share']:.2f}; 2x/1x {f['r21']}; 3x/1x {f['r31']}; harmonics present "
                     f"{f['harmonics'] or 'none'}; 0.5x present {'yes' if f['half_order'] else 'no'} "
                     f"(0.5x/1x {f['half_ratio']}); horizontal/vertical at 1x {f['h_over_v_1x']}.")
        lines.append(f"  rule verdict for this part: {pf['verdict']} ({pf['confidence']}): {pf['reason']}")
    if fp.get("phase_1x_x_rad"):
        lines.append("Phase between parts at 1x, horizontal (rad): "
                     + "; ".join(f"{k}: {v}" for k, v in fp["phase_1x_x_rad"].items()))
    lines += ["", f"Official rule verdict (fixed rules on the part that vibrates most, {fp['lead_part']}): "
              f"{fp['verdict']} ({fp['confidence']} confidence): {fp['reason']}. Alert level "
              f"{fp['alert_level']:g} {fp['alert_unit']} RMS."]
    if plan and plan.get("rpm_range"):
        lines.append(f"Inspection plan: {plan.get('machine_type')}, expected {plan['rpm_range'][0]:.0f} to "
                     f"{plan['rpm_range'][1]:.0f} rpm.")
    return "\n".join(lines)


# ---------------------------------------------------------------- verification

def normalise_fault(f):
    s = str(f or "").strip().lower()
    if s in FAULTS:
        return s
    for key, val in SYNONYMS:
        if key in s:
            return val
    return "unclear"


def normalise_confidence(c):
    s = str(c or "").strip().lower()
    return s if s in CONFIDENCES else "low"


def parse_order(o):
    """1.0 for "1x", "1X", "1", "1.0x"; None for anything else."""
    m = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*[xX×]?\s*", str(o))
    return float(m.group(1)) if m else None


def _match_part(name, parts):
    n = str(name or "").strip().lower()
    for p in parts:
        if p.lower() == n:
            return p
    cand = [p for p in parts if n and (n in p.lower() or p.lower() in n)]
    return cand[0] if len(cand) == 1 else None


def no_em_dash(s):
    return re.sub(r"\s*\u2014\s*", ", ", str(s))


def check_item(ev, result, need):
    """One evidence item: status verified, context or dropped, with the reason and the matched peak."""
    bin_hz = result["bin_hz"]
    item = {"part": ev.get("part"), "frequency_hz": ev.get("frequency_hz"), "order": ev.get("order")}
    part = _match_part(ev.get("part"), result["parts"])
    try:
        hz = float(ev.get("frequency_hz"))
    except (TypeError, ValueError):
        return {**item, "status": "dropped", "reason": "no frequency given"}
    if part is None:
        return {**item, "status": "dropped", "reason": f"no measured part named {ev.get('part')!r}"}
    peaks = [(ax, p) for ax in "xy" for p in result["parts"][part]["axes"][ax]["peaks"]]
    near = [(ax, p) for ax, p in peaks if abs(p["hz"] - hz) <= bin_hz + 1e-9]
    if not near:
        return {**item, "status": "dropped",
                "reason": f"no measured peak of {part} within one bin ({bin_hz:.3f} Hz) of {hz:g} Hz"}
    ax, p = min(near, key=lambda q: (abs(q[1]["hz"] - hz), -q[1]["amp_px"]))
    item["matched"] = {"part": part, "axis": ax, "hz": p["hz"], "label": p["label"], "amp_px": p["amp_px"],
                       "snr": p["snr"]}
    if p["label"] == "flicker":
        return {**item, "status": "dropped", "reason": f"the peak at {p['hz']:g} Hz is light flicker, not vibration"}
    cited = parse_order(ev.get("order"))
    if cited is not None and (p["order"] is None or abs(cited - p["order"]) > 1e-6):
        return {**item, "status": "dropped", "reason": f"cited as {ev.get('order')} but the measured peak is {p['label']}"}
    if p["label"] in need:
        return {**item, "status": "verified", "reason": f"matches the measured {p['label']} peak"}
    return {**item, "status": "context", "reason": f"a real {p['label']} peak, but it is not what this fault rests on"}


def supported(fault, items):
    v = {i["matched"]["label"] for i in items if i["status"] == "verified"}
    if fault == "imbalance":
        return "1x" in v
    if fault == "misalignment":
        return "2x" in v
    if fault == "looseness":
        return "0.5x" in v or len(v & set(fingerprint.HARMONIC_ORDERS)) >= fingerprint.MIN_HARMONICS
    return True


def verify(claim, result, fp):
    """The diagnosis card: rule verdict as the headline, Spark's claim checked and capped beside it."""
    fault = normalise_fault(claim.get("fault"))
    need = fingerprint.FAULT_ORDERS.get(fault, set())
    items = [check_item(ev, result, need) for ev in claim.get("evidence") or [] if isinstance(ev, dict)]
    counts = {s: sum(i["status"] == s for i in items) for s in ("verified", "context", "dropped")}
    ok = supported(fault, items)
    reasons = []
    if fault != fp["verdict"]:
        reasons.append(f"disagrees with the rule verdict ({fp['verdict']})")
    if counts["dropped"]:
        reasons.append(f"{counts['dropped']} evidence item(s) dropped")
    if not ok:
        reasons.append(f"{fault} needs {NEEDS[fault]}, and no verified evidence shows one")
    reported = normalise_confidence(claim.get("confidence"))
    spark = {"fault": fault, "fault_reported": claim.get("fault"), "confidence": "low" if reasons else reported,
             "confidence_reported": reported, "capped": bool(reasons) and reported != "low", "cap_reasons": reasons,
             "agrees": fault == fp["verdict"], "supported": ok, "evidence": items, "counts": counts,
             "explanation": no_em_dash(claim.get("explanation", "")),
             "fix": [no_em_dash(s) for s in claim.get("fix") or []],
             "check_next": [no_em_dash(s) for s in claim.get("check_next") or []]}
    headline = {"fault": fp["verdict"], "confidence": fp["confidence"], "reason": fp["reason"],
                "lead_part": fp["lead_part"], "evidence": fp["evidence"], "source": "fixed rules"}
    return {"headline": headline, "spark": spark, "show_both": bool(reasons)}


# ---------------------------------------------------------------- the call

def diagnose(api: MetaAPI, result, fp, spectrum_png=None, magnified_clip=None, magnified_file_id=None, plan=None,
             label="diagnose"):
    """Ask Spark, verify its claim, and return the card (with cost). A failed Spark call still returns the
    rule verdict as the headline; only a 402 propagates."""
    machine = (plan or {}).get("machine_type") or "machine"
    if magnified_clip and not magnified_file_id:
        magnified_file_id = api.upload(magnified_clip, label=f"{label}:magnified")
    clip_note = (", and the attached video shows the original and a magnified copy side by side"
                 if magnified_file_id else "")
    text = PROMPT.format(machine=machine, clip_note=clip_note, table=peak_table(result, fp, plan))
    content = [{"type": "input_text", "text": text}]
    if spectrum_png:
        content.append(image_block(spectrum_png))
    if magnified_file_id:
        content.append({"type": "input_file", "file_id": magnified_file_id})
    try:
        reply = api.spark(content, label=label, schema=SCHEMA, schema_name="diagnosis")
        claim = reply.json()
    except CreditExhausted:
        raise
    except Exception as e:  # incomplete, unparseable or failed: keep the rule verdict
        card = verify({"fault": "unclear", "confidence": "low", "evidence": []}, result, fp)
        card["spark"] = {"error": str(e)}
        card["show_both"] = True
        card["usd"] = round(api.ledger.total(op="spark", label=label), 6)
        return card
    card = verify(claim, result, fp)
    card["spark_raw"] = claim
    card["usd"] = round(reply.usd, 6)
    card["usage"] = {k: reply.usage.get(k) for k in ("input_tokens", "output_tokens")}
    return card
