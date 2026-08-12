"""Guarded scoring — turns judge results into the paper's main-table row.

    python -m omni_persona.score results/qwen3b_base/judge_results.jsonl \
        --name "Qwen2.5-Omni-3B (base)" --json-out r.json --csv-out r.csv

Guarded rule (the paper's headline definition, x100):
  answerable   correct  <=>  NOT is_abstain(pred) AND judge_verdict == "CORRECT"
  unanswerable correct  <=>  is_abstain(pred)
  Ans   = ans_correct / n_answerable
  Unans = unans_abstain / n_unanswerable
  Cal   = (ans_correct + unans_abstain) / n_total        (count-weighted)
  1-FA  = answerable items where the model did NOT abstain / n_answerable
  TA    = unans_abstain / n_unanswerable
  Avg   = (1-FA + TA) / 2

Reference values are expected approximate values, not exact targets: the LLM
judge is non-deterministic and moves Ans/Cal by roughly +-1 point between runs.
1-FA and TA are derived from the keyword abstention rule only and are
deterministic — they should match a reference row exactly.
"""

import argparse
import csv
import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .judge import is_abstain

GROUP = {
    "visual_identity": "I2I",
    "voice_identity": "A2A",
    "same_modal_semantic": "T2T",
    "cross_modal_semantic_bridge": "T2Any",
}
GROUPS = ("I2I", "A2A", "T2T", "T2Any")


def _pct(num: int, den: int) -> Optional[float]:
    return 100.0 * num / den if den else None


def compute_guarded(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Guarded overall + per-scenario metrics from judge_results rows."""
    g = {k: {"ac": 0, "uc": 0, "na": 0, "nu": 0, "att": 0} for k in GROUPS}
    for r in rows:
        grp = GROUP.get(r.get("scenario"))
        if grp is None:
            continue
        abstained = is_abstain(r.get("model_prediction", "") or "")
        if r.get("answerable_from_context"):
            g[grp]["na"] += 1
            if not abstained:
                g[grp]["att"] += 1
                if r.get("judge_verdict") == "CORRECT":
                    g[grp]["ac"] += 1
        else:
            g[grp]["nu"] += 1
            if abstained:
                g[grp]["uc"] += 1

    tot = {"ac": 0, "uc": 0, "na": 0, "nu": 0, "att": 0}
    scen: Dict[str, Dict[str, Any]] = {}
    for grp in GROUPS:
        d = g[grp]
        for k in tot:
            tot[k] += d[k]
        scen[grp] = {
            "n": d["na"] + d["nu"],
            "n_answerable": d["na"],
            "n_unanswerable": d["nu"],
            "Ans": _pct(d["ac"], d["na"]),
            "Unans": _pct(d["uc"], d["nu"]),
            "Cal": _pct(d["ac"] + d["uc"], d["na"] + d["nu"]),
        }

    one_fa = _pct(tot["att"], tot["na"])
    ta = _pct(tot["uc"], tot["nu"])
    overall = {
        "n": tot["na"] + tot["nu"],
        "n_answerable": tot["na"],
        "n_unanswerable": tot["nu"],
        "Ans": _pct(tot["ac"], tot["na"]),
        "Unans": ta,
        "Cal": _pct(tot["ac"] + tot["uc"], tot["na"] + tot["nu"]),
        "1-FA": one_fa,
        "TA": ta,
        "Avg": (one_fa + ta) / 2 if (one_fa is not None and ta is not None) else None,
    }
    return {"overall": overall, "scenario": scen, "counts": {"overall": tot, "scenario": g}}


def load_judge_results(path: str) -> List[Dict[str, Any]]:
    """Load a judge_results .jsonl, or the newest one inside a directory."""
    p = Path(path)
    if p.is_dir():
        hits = sorted(glob.glob(os.path.join(str(p), "judge_results*.jsonl")))
        if not hits:
            raise FileNotFoundError(f"no judge_results*.jsonl in {p}")
        p = Path(hits[-1])
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _f(v: Optional[float], w: int = 6) -> str:
    return f"{'-':>{w}}" if v is None else f"{v:{w}.1f}"


def print_report(res: Dict[str, Any], name: str) -> None:
    o, s = res["overall"], res["scenario"]
    print()
    print(f"Omni-Persona v2.2 — guarded metrics   model: {name}")
    print(f"  N={o['n']}  answerable={o['n_answerable']}  unanswerable={o['n_unanswerable']}")
    print()
    print("  Main table row")
    print("  " + "-" * 46)
    print(f"  {'Ans':>7}{'Cal':>7}{'1-FA':>7}{'TA':>7}{'Avg':>7}")
    print(f"  {_f(o['Ans'], 7)}{_f(o['Cal'], 7)}{_f(o['1-FA'], 7)}{_f(o['TA'], 7)}{_f(o['Avg'], 7)}")
    print()
    print("  Per scenario")
    print("  " + "-" * 46)
    print(f"  {'group':<7}{'N':>5}{'Ans':>8}{'Unans':>8}{'Cal':>8}")
    for grp in GROUPS:
        d = s[grp]
        print(f"  {grp:<7}{d['n']:>5}{_f(d['Ans'], 8)}{_f(d['Unans'], 8)}{_f(d['Cal'], 8)}")
    print()


def _flat_row(res: Dict[str, Any], name: str) -> Dict[str, Any]:
    o = res["overall"]
    row: Dict[str, Any] = {"model": name, "n": o["n"]}
    for k in ("Ans", "Cal", "1-FA", "TA", "Avg"):
        row[k] = None if o[k] is None else round(o[k], 1)
    for grp in GROUPS:
        for k in ("Ans", "Unans", "Cal"):
            v = res["scenario"][grp][k]
            row[f"{grp}_{k}"] = None if v is None else round(v, 1)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Omni-Persona v2.2 guarded scoring")
    ap.add_argument("results", help="judge_results .jsonl, or a directory containing one")
    ap.add_argument("--name", default=None, help="label for the table row")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--csv-out", default=None)
    ap.add_argument("--expect-n", type=int, default=0,
                    help="warn if the item count differs (750 for the full benchmark)")
    args = ap.parse_args()

    rows = load_judge_results(args.results)
    name = args.name or (rows[0].get("model_name") if rows else "unknown")
    res = compute_guarded(rows)

    if args.expect_n and res["overall"]["n"] != args.expect_n:
        print(f"[warn] scored {res['overall']['n']} items, expected {args.expect_n}")

    print_report(res, name)
    flat = _flat_row(res, name)

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"model": name, **res, "row": flat}, f, indent=2)
        print(f"[done] json → {args.json_out}")
    if args.csv_out:
        Path(args.csv_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv_out, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(flat))
            w.writeheader()
            w.writerow(flat)
        print(f"[done] csv  → {args.csv_out}")


if __name__ == "__main__":
    main()
