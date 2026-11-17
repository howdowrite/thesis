#!/usr/bin/env python3
"""Hypothesis tests and calibration evidence for the Delegation Advisor (protocol v1.0, section 12 entry dated 13 November 2026).

Puts the decision rules already frozen in protocol section 9 into formal hypothesis-test form, and adds the evidence
for the Advisor's predicted chance (the 'confidence value'). Nothing about the features, model, label, seed,
repetitions or resamples differs from the primary analysis. All tests are ONE-SIDED, paired, and use the same
10,000 bootstrap resamples of the 266 tasks (seed 42).

  T1  Cost: H0 cost per solved task of the Advisor >= that of cheaper-first (policy C); H1 lower.  Run with margin 0
      (the frozen rule) and with a minimum meaningful saving (default 5% of cheaper-first's cost per solved task).
  T2  Solves (non-inferiority): H0 Advisor solve rate is 2 or more percentage points below C; H1 within 2 points.
  T3  Ranking: H0 AUC of the Advisor <= AUC of the comparator; H1 greater. Comparators: repository-only; Scrouting text head.
  T4  Ranking: H0 AUC of the Advisor <= 0.5 (no better than chance); H1 greater.
  T5  Confidence value (calibration): H0 Brier score of the Advisor >= Brier score of the base rate; H1 lower.
      H0 calibration slope <= 0 (the predicted chance carries no information); H1 positive. Reliability table and ECE.
  T6  Policy: H0 cost per solved task of cheaper-first (C) >= frontier for all (A); H1 lower.
Decision level: one-sided alpha = 0.05 for every test (adviser's request, written confirmation pending; 0.05 one-sided equals a 90% two-sided interval, not the 95% of §9);
p-values are shown so a reader can also apply 0.025 (the frozen level). Bootstrap p-value = (1 + resamples on the H0 side) / (resamples + 1).
No adjustment for multiple tests: T1 and T2 must BOTH pass for the primary claim (intersection rule); T3 to T6 are supporting.

DEFAULT MODE is a leakage check: the Advisor is trained on SHUFFLED labels and scored against the real outcomes, so no test
should reject. REAL MODE: --real --confirm-reviewed --primary-results results-primary.md --out <file>. Refuses to overwrite.

Usage:
  python analysis_hypothesis_tests.py <tasks_pro266.csv> <superscout-clone>                       (shuffled check)
  python analysis_hypothesis_tests.py <tasks_pro266.csv> <superscout-clone> --real --confirm-reviewed \
        --primary-results results-primary.md --out results-hypothesis-tests.md [--margin-rel 0.05]
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import advisor_common as AC

ALPHA = 0.05


def p_ge(x, thr):          # H0: statistic >= thr
    return (1 + int((x >= thr).sum())) / (len(x) + 1)


def p_le(x, thr):          # H0: statistic <= thr
    return (1 + int((x <= thr).sum())) / (len(x) + 1)


def dec(p):
    return "reject H0" if p < ALPHA else "fail to reject H0"


def one_sided(diff, p_two):    # DeLong two-sided p to a one-sided p for H1: first AUC greater
    return p_two / 2 if diff > 0 else 1 - p_two / 2


def slope_fit(s, y):
    z = np.log(np.clip(s, 1e-4, 1 - 1e-4) / (1 - np.clip(s, 1e-4, 1 - 1e-4)))[:, None]
    m = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
    return float(m.coef_[0][0]), float(m.intercept_[0])


def main():
    flags, rest, it = {}, [], iter(sys.argv[1:])
    for a in it:
        if a in ("--primary-results", "--out", "--margin-rel"):
            flags[a] = next(it)
        elif a.startswith("--"):
            flags[a] = True
        else:
            rest.append(a)
    if len(rest) != 2:
        sys.exit("Expected 2 arguments.\n" + __doc__)
    real_mode = "--real" in flags
    margin_rel = float(flags.get("--margin-rel", 0.05))
    if real_mode:
        if "--confirm-reviewed" not in flags or "--out" not in flags or "--primary-results" not in flags:
            sys.exit("Real mode needs --confirm-reviewed, --primary-results and --out.")
        if not Path(flags["--primary-results"]).is_file():
            sys.exit("The primary results file does not exist.")
        if Path(flags["--out"]).exists():
            sys.exit(f"{flags['--out']} already exists. Run once; move it away deliberately to repeat.")
    d = AC.load_table(rest[0])
    N, B = len(d["ks"]), AC.B
    Xp, keep, _ = AC.feature_matrix(d["text"])
    ks, os_ = d["ks"], d["os"]
    y_train = ks if real_mode else np.random.default_rng(AC.SEED).permutation(ks)

    P, TH, _ = AC.repeated_oof(Xp, y_train, d, True)
    PE, _, _ = AC.repeated_oof(AC.repo_matrix(d["repo"], sorted(set(d["repo"]))), y_train, d, False)
    sD, sE = P.mean(0), PE.mean(0)
    pv = json.load(open(Path(rest[1]) / "data/receipts/routing_scenarios.json", encoding="utf-8"))["probability_vectors"]
    pt = np.array([pv[t]["kimi-k2-5"]["p_text"] for t in d["ids"]])
    pols = AC.policies(d, {"D": (P < TH).mean(0)})

    # consistency with the frozen primary results (real mode only)
    res = AC.bootstrap(pols)
    ch, ni, v = AC.verdict(res["D"])
    if real_mode:
        prim = Path(flags["--primary-results"]).read_text(encoding="utf-8")
        a1, a2, _, _ = AC.delong(ks, sD, sE)
        need = [f"Verdict under the protocol rule: **{v}**", f"| Advisor vs repository-only | {a1:.3f} | {a2:.3f} |",
                f"| D Advisor | ${pols['D'][0].sum():.2f} | {pols['D'][1].sum():.1f} |"]
        miss = [n for n in need if n not in prim]
        if miss:
            sys.exit("Checks against results-primary.md failed; nothing written:\n- " + "\n- ".join(miss))
        print("consistency checks against results-primary.md passed")

    idx = np.random.default_rng(AC.SEED).integers(0, N, size=(B, N))
    cps, sr = {}, {}
    for k in ("A", "C", "D"):
        c, s = pols[k]
        ss = s[idx].sum(1)
        cps[k] = c[idx].sum(1) / np.where(ss > 0, ss, np.nan)
        sr[k] = ss / N
    obs = {k: pols[k][0].sum() / pols[k][1].sum() for k in ("A", "C", "D")}
    L, rows = [], []
    w = L.append

    # T1
    dlt, d_obs = cps["D"] - cps["C"], obs["D"] - obs["C"]
    margin = margin_rel * obs["C"]
    p1a, p1b = p_ge(dlt, 0.0), p_ge(dlt, -margin)
    # T2
    dsr, ds_obs = (sr["D"] - sr["C"]) * 100, (pols["D"][1].sum() - pols["C"][1].sum()) / N * 100
    p2 = p_le(dsr, -AC.NONINF_PP)
    # T3, T4
    a1, a2, dd, p_two = AC.delong(ks, sD, sE)
    b1, b2, dd2, p_two2 = AC.delong(ks, sD, pt)
    p3a, p3b = one_sided(dd, p_two), one_sided(dd2, p_two2)
    aucs = np.array([roc_auc_score(ks[b], sD[b]) if 0 < ks[b].sum() < N else np.nan for b in idx])
    aucs = aucs[~np.isnan(aucs)]
    p4 = p_le(aucs, 0.5)
    # T5
    eD, eB = (sD - ks) ** 2, (ks.mean() - ks) ** 2
    bdiff = (eD - eB)[idx].mean(1)
    p5a = p_ge(bdiff, 0.0)
    bss = 1 - eD.mean() / eB.mean()
    slope, icpt = slope_fit(sD, ks)
    rng = np.random.default_rng(AC.SEED + 1)
    sl = []
    for _ in range(2000):
        b = rng.integers(0, N, N)
        if 0 < ks[b].sum() < N:
            sl.append(slope_fit(sD[b], ks[b])[0])
    sl = np.array(sl)
    p5b = p_le(sl, 0.0)
    # T6
    dAC = cps["C"] - cps["A"]
    p6 = p_ge(dAC, 0.0)
    ci = lambda x: AC._ci(x[~np.isnan(x)])

    tests = [
        ("T1a", "Cost per solved: Advisor vs cheaper-first (frozen rule, margin 0)", "Advisor >= cheaper-first", "Advisor < cheaper-first", f"{d_obs:+.5f} $", p1a),
        ("T1b", f"Cost per solved: Advisor saves at least {margin_rel:.0%} (${margin:.3f}) vs cheaper-first", f"Advisor saves < ${margin:.3f}", f"Advisor saves >= ${margin:.3f}", f"{d_obs:+.5f} $", p1b),
        ("T2", "Solve rate: Advisor not worse than cheaper-first by 2 points", "Advisor 2 or more points below", "Advisor within 2 points", f"{ds_obs:+.3f} pp", p2),
        ("T3a", "Ranking: Advisor AUC vs repository-only", "AUC Advisor <= repo-only", "AUC Advisor > repo-only", f"{a1:.3f} vs {a2:.3f}", p3a),
        ("T3b", "Ranking: Advisor AUC vs Scrouting text head", "AUC Advisor <= text head", "AUC Advisor > text head", f"{b1:.3f} vs {b2:.3f}", p3b),
        ("T4", "Ranking: Advisor AUC vs chance (0.5)", "AUC <= 0.5", "AUC > 0.5", f"{a1:.3f}", p4),
        ("T5a", "Confidence value: Brier score vs base rate", "Brier Advisor >= base rate", "Brier Advisor < base rate", f"{eD.mean():.4f} vs {eB.mean():.4f}", p5a),
        ("T5b", "Confidence value: calibration slope", "slope <= 0 (no information)", "slope > 0", f"{slope:.3f}", p5b),
        ("T6", "Policy: cheaper-first vs frontier for all, cost per solved", "C >= frontier for all", "C < frontier for all", f"{obs['C'] - obs['A']:+.3f} $", p6),
    ]
    w("# Hypothesis tests and calibration (protocol v1.0; section 12 entry 13 November 2026)\n")
    if not real_mode:
        w("**SHUFFLED-LABEL LEAKAGE CHECK. The Advisor was trained on shuffled labels; no test should reject. These numbers mean nothing.**\n")
    w(f"{N} tasks, features {keep}, {AC.REPS} repetitions of stratified 5-fold, seed {AC.SEED}, {B} paired bootstrap resamples. "
      f"One-sided; alpha = {ALPHA} (adviser's request, written confirmation pending; 0.05 one-sided = 90% two-sided, not 95% of §9); "
      f"frozen alpha = 0.025 (95% two-sided of §9). My arithmetic.\n")
    w("## 1. Summary\n")
    w("| Test | Question | H0 | H1 | Observed | One-sided p | Decision at 0.05 | Decision at 0.025 (frozen) |")
    w("|---|---|---|---|---|---|---|---|")
    for t in tests:
        p = t[5]
        w(f"| {t[0]} | {t[1]} | {t[2]} | {t[3]} | {t[4]} | {p:.4f} | {'reject H0' if p < 0.05 else 'fail to reject H0'} | {'reject H0' if p < 0.025 else 'fail to reject H0'} |")
    primary_pass = (p1a < ALPHA) and (p2 < ALPHA)
    w(f"\n**Frozen protocol verdict on the primary claim** (cost interval wholly below zero AND non-inferior solve rate): **{v}** "
      f"(cost interval upper bound {res['D']['d_ci'][1]:+.5f}; solve-rate lower bound {res['D']['dsolve_ci'][0]:+.2f} pp).")
    w(f"Formal form: T1a and T2 must both reject H0 for the primary claim. Here T1a {'rejects' if p1a < ALPHA else 'does not reject'} and T2 {'rejects' if p2 < ALPHA else 'does not reject'}.\n")
    w(f"T1a and T1b together show why a margin is needed: when the Advisor and cheaper-first differ by only ${abs(d_obs):.5f} per solved task, "
      "a test against zero can reject H0 on a difference too small to matter, or fail to, depending on rounding of rare partly-routed tasks. "
      f"T1b asks for a saving of at least {margin_rel:.0%}. The margin is a practical-significance addition and is not part of the frozen rule.\n")
    w("## 2. Details\n")
    w(f"- Cost per solved task: frontier for all ${obs['A']:.3f}, cheaper-first ${obs['C']:.3f}, Advisor ${obs['D']:.3f}.")
    w(f"- T1 difference (Advisor minus cheaper-first): {d_obs:+.5f}, 95% interval {ci(dlt)[0]:+.5f} to {ci(dlt)[1]:+.5f}.")
    w(f"- T2 solve-rate difference: {ds_obs:+.3f} pp, 95% interval {ci(dsr)[0]:+.3f} to {ci(dsr)[1]:+.3f} (margin -{AC.NONINF_PP:.0f} pp).")
    w(f"- T3 AUC difference vs repository-only {dd:+.3f}; vs Scrouting text head {dd2:+.3f} (DeLong, two-sided p {p_two:.3f} and {p_two2:.3f}, halved to one-sided when the Advisor is higher).")
    w(f"- T4 AUC {a1:.3f}, 95% interval {ci(aucs)[0]:.3f} to {ci(aucs)[1]:.3f}.")
    w(f"- T6 difference (cheaper-first minus frontier for all): {obs['C'] - obs['A']:+.3f} per solved task, 95% interval {ci(dAC)[0]:+.3f} to {ci(dAC)[1]:+.3f}; "
      f"solved {pols['C'][1].sum():.0f} against {pols['A'][1].sum():.0f}.")
    w("\n## 3. Evidence for the confidence value (the predicted chance of cheaper-model success)\n")
    w(f"- Brier score: Advisor {eD.mean():.4f}, base rate {eB.mean():.4f}; Brier skill score {bss:+.3f} (0 means no better than the base rate; 1 is perfect). "
      f"Paired difference {np.mean(eD - eB):+.5f}, 95% interval {ci(bdiff)[0]:+.5f} to {ci(bdiff)[1]:+.5f}.")
    w(f"- Calibration slope {slope:.3f} (1 is ideal; below 1 means the predictions are too extreme; 0 means no information), 95% interval {ci(sl)[0]:.3f} to {ci(sl)[1]:.3f} "
      f"(2,000 resamples); intercept {icpt:+.3f}.\n")
    w("| Predicted-chance group (fifths of tasks, low to high) | Tasks | Mean predicted chance | Observed share solved by the cheaper model |")
    w("|---|---|---|---|")
    order = np.argsort(sD, kind="stable")
    ece = 0.0
    for g, ix in enumerate(np.array_split(order, 5), 1):
        pr, ob = sD[ix].mean(), ks[ix].mean()
        ece += len(ix) / N * abs(pr - ob)
        w(f"| {g} | {len(ix)} | {pr:.3f} | {ob:.3f} |")
    w(f"\nExpected calibration error over these five groups: {ece:.3f}. With about 53 tasks per group, each observed share has an uncertainty of roughly plus or minus 0.13, so only large miscalibration is detectable.\n")
    w("## 4. Limits\n")
    w("- The tests treat the 266 tasks as independent draws from three Python projects and do not cover other projects; the hold-out check showed the signal does not carry across repositories.")
    w("- Out-of-fold scores from one repeated cross-validation are resampled by task, which does not capture variation from retraining; intervals are somewhat optimistic.")
    w("- Outcomes are one recorded run per task, with a 50-call cap; run-to-run variation and false solves are not covered.")
    w("- No adjustment for multiple tests. T1a and T2 form the primary claim; T3 to T6 are supporting and descriptive.")
    text = "\n".join(L) + "\n"
    if real_mode:
        Path(flags["--out"]).write_text(text, encoding="utf-8")
    print(text)
    print(f"written to {flags['--out']}" if real_mode else "(shuffled-label check; nothing written)")


if __name__ == "__main__":
    main()
