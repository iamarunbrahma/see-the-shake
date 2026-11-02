"""Throwaway spike: injected vs measured, per clip."""
import json
from pathlib import Path

out = Path(__file__).parent / "out"
for fault in ("healthy", "imbalance", "loose"):
    t = json.loads((out / f"{fault}_truth.json").read_text())
    m = json.loads((out / f"{fault}_measured.json").read_text())
    print(f"\n{fault}: x velocity {m['axes']['x']['velocity_rms_mm_s']} mm/s rms, noise floor {m['axes']['x']['noise_floor_px']:.4f} px")
    print("  injected:", ", ".join(f"{c['hz']:.2f} Hz {c['x_amp_px']:.3f}px" for c in t["components"]))
    print("  measured:", ", ".join(f"{p['hz']:.2f} Hz {p['amp_px']:.3f}px" for p in m["axes"]["x"]["peaks"]))
