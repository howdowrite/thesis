# Hypothesis tests and calibration (protocol v1.0; section 12 entry 13 November 2026)

266 tasks, features ['n_chars', 'n_code_blocks', 'has_trace_or_error', 'n_code_refs', 'n_headings', 'mentions_tests', 'n_list_items'], 20 repetitions of stratified 5-fold, seed 42, 10000 paired bootstrap resamples. One-sided; alpha = 0.05 (adviser's request, written confirmation pending; 0.05 one-sided = 90% two-sided, not 95% of §9); frozen alpha = 0.025 (95% two-sided of §9). My arithmetic.

## 1. Summary

| Test | Question | H0 | H1 | Observed | One-sided p | Decision at 0.05 | Decision at 0.025 (frozen) |
|---|---|---|---|---|---|---|---|
| T1a | Cost per solved: Advisor vs cheaper-first (frozen rule, margin 0) | Advisor >= cheaper-first | Advisor < cheaper-first | -0.00004 $ | 0.3696 | fail to reject H0 | fail to reject H0 |
| T1b | Cost per solved: Advisor saves at least 5% ($0.038) vs cheaper-first | Advisor saves < $0.038 | Advisor saves >= $0.038 | -0.00004 $ | 1.0000 | fail to reject H0 | fail to reject H0 |
| T2 | Solve rate: Advisor not worse than cheaper-first by 2 points | Advisor 2 or more points below | Advisor within 2 points | +0.000 pp | 0.0001 | reject H0 | reject H0 |
| T3a | Ranking: Advisor AUC vs repository-only | AUC Advisor <= repo-only | AUC Advisor > repo-only | 0.600 vs 0.489 | 0.0094 | reject H0 | reject H0 |
| T3b | Ranking: Advisor AUC vs Scrouting text head | AUC Advisor <= text head | AUC Advisor > text head | 0.600 vs 0.523 | 0.0583 | fail to reject H0 | fail to reject H0 |
| T4 | Ranking: Advisor AUC vs chance (0.5) | AUC <= 0.5 | AUC > 0.5 | 0.600 | 0.0027 | reject H0 | reject H0 |
| T5a | Confidence value: Brier score vs base rate | Brier Advisor >= base rate | Brier Advisor < base rate | 0.2392 vs 0.2464 | 0.0918 | fail to reject H0 | fail to reject H0 |
| T5b | Confidence value: calibration slope | slope <= 0 (no information) | slope > 0 | 0.917 | 0.0025 | reject H0 | reject H0 |
| T6 | Policy: cheaper-first vs frontier for all, cost per solved | C >= frontier for all | C < frontier for all | -0.509 $ | 0.0001 | reject H0 | reject H0 |

**Frozen protocol verdict on the primary claim** (cost interval wholly below zero AND non-inferior solve rate): **not supported** (cost interval upper bound +0.00000; solve-rate lower bound +0.00 pp).
Formal form: T1a and T2 must both reject H0 for the primary claim. Here T1a does not reject and T2 rejects.

T1a and T1b together show why a margin is needed: when the Advisor and cheaper-first differ by only $0.00004 per solved task, a test against zero can reject H0 on a difference too small to matter, or fail to, depending on rounding of rare partly-routed tasks. T1b asks for a saving of at least 5%. The margin is a practical-significance addition and is not part of the frozen rule.

## 2. Details

- Cost per solved task: frontier for all $1.274, cheaper-first $0.765, Advisor $0.765.
- T1 difference (Advisor minus cheaper-first): -0.00004, 95% interval -0.00013 to +0.00000.
- T2 solve-rate difference: +0.000 pp, 95% interval +0.000 to +0.000 (margin -2 pp).
- T3 AUC difference vs repository-only +0.112; vs Scrouting text head +0.078 (DeLong, two-sided p 0.019 and 0.117, halved to one-sided when the Advisor is higher).
- T4 AUC 0.600, 95% interval 0.533 to 0.666.
- T6 difference (cheaper-first minus frontier for all): -0.509 per solved task, 95% interval -0.631 to -0.403; solved 188 against 158.

## 3. Evidence for the confidence value (the predicted chance of cheaper-model success)

- Brier score: Advisor 0.2392, base rate 0.2464; Brier skill score +0.029 (0 means no better than the base rate; 1 is perfect). Paired difference -0.00713, 95% interval -0.01761 to +0.00344.
- Calibration slope 0.917 (1 is ideal; below 1 means the predictions are too extreme; 0 means no information), 95% interval 0.285 to 1.590 (2,000 resamples); intercept +0.016.

| Predicted-chance group (fifths of tasks, low to high) | Tasks | Mean predicted chance | Observed share solved by the cheaper model |
|---|---|---|---|
| 1 | 54 | 0.428 | 0.463 |
| 2 | 53 | 0.516 | 0.453 |
| 3 | 53 | 0.566 | 0.585 |
| 4 | 53 | 0.611 | 0.642 |
| 5 | 53 | 0.686 | 0.660 |

Expected calibration error over these five groups: 0.035. With about 53 tasks per group, each observed share has an uncertainty of roughly plus or minus 0.13, so only large miscalibration is detectable.

## 4. Limits

- The tests treat the 266 tasks as independent draws from three Python projects and do not cover other projects; the hold-out check showed the signal does not carry across repositories.
- Out-of-fold scores from one repeated cross-validation are resampled by task, which does not capture variation from retraining; intervals are somewhat optimistic.
- Outcomes are one recorded run per task, with a 50-call cap; run-to-run variation and false solves are not covered.
- No adjustment for multiple tests. T1a and T2 form the primary claim; T3 to T6 are supporting and descriptive.
