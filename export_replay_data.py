#!/usr/bin/env python3
"""Export the data the replay viewer needs (prototype spec section 7). Computes no new result.

It reproduces the out-of-fold predictions that the primary and exploratory runs held in memory (same code, features,
seed and repetitions from advisor_common.py), then CHECKS them against the saved results files. If any check fails
nothing is written. Reads only the frozen task tables (hash-checked), Scrouting's released text-head probabilities,
and the two results files. Never reads post_* columns.

Usage:
  python export_replay_data.py <tasks_pro266.csv> <superscout-clone> --primary-results results-primary.md \
        --exploratory-results results-exploratory.md --out replay_data.json
Refuses to overwrite an existing output file.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss

import advisor_common as AC
import features as F

CUTS = (0.3, 0.5, 0.7)


def main():
    flags, rest, it = {}, [], iter(sys.argv[1:])
    for a in it:
        if a in ("--primary-results", "--exploratory-results", "--out"):
            flags[a] = next(it)
        else:
            rest.append(a)
    if len(rest) != 2 or len(flags) != 3:
        sys.exit("Expected 2 arguments and --primary-results, --exploratory-results, --out.\n" + __doc__)
    out = Path(flags["--out"])
    if out.exists():
        sys.exit(f"{out} already exists. Move it away deliberately to repeat.")
    prim = Path(flags["--primary-results"]).read_text(encoding="utf-8")
    expl = Path(flags["--exploratory-results"]).read_text(encoding="utf-8")
    d = AC.load_table(rest[0])
    N = len(d["ids"])
    Xp, keep, _ = AC.feature_matrix(d["text"])
    ks, os_ = d["ks"], d["os"]
    neither = ((ks == 0) & (os_ == 0)).astype(int)

    # ---- reproduce the in-memory predictions ----
    P, TH, _ = AC.repeated_oof(Xp, ks, d, True)                      # Advisor, Kimi success
    cats = sorted(set(d["repo"]))
    PE, _, _ = AC.repeated_oof(AC.repo_matrix(d["repo"], cats), ks, d, False)
    H, _, _ = AC.repeated_oof(Xp, neither, d, True)                 # hold-flag classifier
    pv = json.load(open(Path(rest[1]) / "data/receipts/routing_scenarios.json", encoding="utf-8"))["probability_vectors"]
    pt = np.array([pv[t]["kimi-k2-5"]["p_text"] for t in d["ids"]])
    sD, sE, sH = P.mean(0), PE.mean(0), H.mean(0)

    # ---- checks against the saved results ----
    fails = []
    a1, a2, _, _ = AC.delong(ks, sD, sE)
    need = f"| Advisor vs repository-only | {a1:.3f} | {a2:.3f} |"
    if need not in prim:
        fails.append(f"primary: expected the line '{need}'")
    brier = float(np.mean([brier_score_loss(ks, P[r]) for r in range(P.shape[0])]))
    if f"Advisor {brier:.4f}" not in prim:
        fails.append(f"primary: Brier score {brier:.4f} not found")
    pols = AC.policies(d, {"D": (P < TH).mean(0)})
    spend, solved = pols["D"][0].sum(), pols["D"][1].sum()
    if f"| D Advisor | ${spend:.2f} | {solved:.1f} |" not in prim:
        fails.append(f"primary: expected '| D Advisor | ${spend:.2f} | {solved:.1f} |'")
    base = neither.mean()
    if f"Base rate of 'neither model solves': {base:.1%} ({neither.sum()} of {N} tasks)" not in expl:
        fails.append("exploratory: base rate line differs")
    h1, h2, _, _ = AC.delong(neither, sH, AC.repeated_oof(AC.repo_matrix(d["repo"], cats), neither, d, False)[0].mean(0))
    if f"Advisor-form classifier {h1:.3f}, repository-only {h2:.3f}" not in expl:
        fails.append(f"exploratory: hold-flag AUCs {h1:.3f}/{h2:.3f} not found")
    Ccost = d["ck"] + (1 - ks) * d["co"]
    for cut in CUTS:
        f = (H >= cut).mean(0)
        nf, tp = f.sum(), (f * neither).sum()
        prec = tp / nf if nf > 0 else float("nan")
        if f"| {cut} | {nf:.1f} | {prec:.3f} |" not in expl:
            fails.append(f"exploratory: hold-flag row for cutoff {cut} ({nf:.1f} flagged, precision {prec:.3f}) not found")
    if fails:
        sys.exit("Checks failed; nothing written:\n- " + "\n- ".join(fails))
    print("all checks against results-primary.md and results-exploratory.md passed")

    # ---- price-ratio table, copied from the saved exploratory results (not recomputed) ----
    ratio_rows, inside = [], False
    for line in expl.splitlines():
        if line.startswith("| Ratio r"):
            inside = True
            continue
        if inside and line.startswith("|---"):
            continue
        if inside and line.startswith("|"):
            ratio_rows.append([c.strip() for c in line.strip("|").split("|")])
        elif inside:
            break
    if not ratio_rows:
        sys.exit("Could not find the price-ratio table in the exploratory results; nothing written.")

    # ---- per-task records; outcomes kept in a separate block so the viewer can hold them back until reveal ----
    tasks, outcomes = [], {}
    for i, tid in enumerate(d["ids"]):
        raw = F.features(d["text"][i])
        tasks.append({
            "task_id": tid, "repo": str(d["repo"][i]), "issue_text": d["text"][i],
            "features": {k: raw[k] for k, _ in AC.FEATURE_SPEC if k in keep},
            "p_advisor": round(float(sD[i]), 4), "p_text_head": round(float(pt[i]), 4),
            "hold_score": round(float(sH[i]), 4),
        })
        outcomes[tid] = {"kimi_solved": int(ks[i]), "kimi_cost": round(float(d["ck"][i]), 4),
                         "opus_solved": int(os_[i]), "opus_cost": round(float(d["co"][i]), 4),
                         "neither": int(neither[i])}
    theta = float(d["ck"].sum() / d["co"].sum())
    pa = AC.policies(d, {"D": (P < TH).mean(0)})
    totals = {k: {"spend": round(float(v[0].sum()), 2), "solved": round(float(v[1].sum()), 1),
                  "cost_per_solved": round(float(v[0].sum() / v[1].sum()), 3)} for k, v in pa.items()}
    data = {
        "note": "Recorded runs replayed. Predictions are out-of-fold averages over 20 repetitions of stratified 5-fold, seed 42; "
                "no task is predicted by a model that saw it. Outcomes are in the separate 'outcomes' block.",
        "n_tasks": N, "cost_ratio_actual": round(theta, 4), "base_rate_neither": round(float(base), 4),
        "policy_labels": {"A": "Frontier for all", "B": "Cheaper for all", "C": "Cheaper first, escalate once",
                          "D": "Advisor (handcrafted)", "H": "Perfect advance choice (ceiling)"},
        "policy_totals": totals,
        "hold_cutoffs": list(CUTS),
        "price_ratio_table": {"header": ["ratio", "advisor_share_routed", "advisor_cost_vs_cheaper_first", "advisor_interval",
                                          "solve_rate_diff_pp", "head_share_routed", "head_cost_vs_cheaper_first", "head_interval",
                                          "oracle_vs_cheaper_first"], "rows": ratio_rows},
        "tasks": tasks, "outcomes": outcomes,
    }
    out.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"written {out}: {N} tasks")


if __name__ == "__main__":
    main()
