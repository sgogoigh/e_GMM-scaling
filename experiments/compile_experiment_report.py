"""
Experiment report compiler: experiments/results/exp_0*.json -> FINAL-EXPERIMENT-REPORT.md
==========================================================================================

Rewritten 2026-11-25 (docs/review/2026-11-25-REVIEW.md). Every verdict is computed from the result
files with the rule in experiments/HYPOTHESES-AND-RESEARCH-DESIGN.md Section 3, implemented in
src/e_gmm/stats.py: paired by seed, exact sign-flip test, p < 0.05, practical threshold 0.01 nats per
dimension, at least 10 seeds. Nothing is hard-coded: a missing result is reported as "Not run".
"""

import json
import pathlib
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

EXP_DIR = pathlib.Path(__file__).resolve().parent
ROOT = EXP_DIR.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(EXP_DIR))

from e_gmm.stats import equivalence_verdict, paired_comparison, verdict  # noqa: E402
from exp_logging import RESULTS_ROOT, tee_experiment_logs  # noqa: E402

THRESHOLD = 0.01  # nats per dimension
MIN_SEEDS = 10
NOT_RUN = "Not run"


def _load(name: str) -> Optional[Dict[str, Any]]:
    p = RESULTS_ROOT / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _f(x, nd=4, signed=False) -> str:
    if x is None:
        return "-"
    try:
        if not np.isfinite(float(x)):
            return "nan"
    except (TypeError, ValueError):
        return str(x)
    return f"{float(x):+.{nd}f}" if signed else f"{float(x):.{nd}f}"


def _p(x) -> str:
    return "-" if x is None else (f"{x:.4f}" if x >= 1e-4 else f"{x:.1e}")


def claim_from_comparison(cmp: Dict[str, Any], flip: bool, threshold: float = THRESHOLD) -> Dict[str, Any]:
    """
    cmp = paired_comparison(method, reference). For the claim "reference beats method", flip=True
    (diff = reference - method = -mean_diff). For "method beats reference", flip=False.
    """
    if not cmp or "mean_diff" not in cmp:
        return {"verdict": "Inconclusive (no paired result)", "diff": None, "p": None, "n": None}
    diff = -cmp["mean_diff"] if flip else cmp["mean_diff"]
    v = verdict(diff, cmp["p_value"], threshold, min_n=MIN_SEEDS, n=cmp["n"])
    return {"verdict": v, "diff": diff, "p": cmp["p_value"], "n": cmp["n"]}


# ---------------------------------------------------------------------------------------------


def section_exp00(e0, lines: List[str]) -> None:
    lines += ["## Exp 00: environment", ""]
    if e0 is None:
        lines += [f"{NOT_RUN}.", ""]
        return
    hw = e0.get("hardware", {})
    lines += [
        f"- CPU cores: {hw.get('cpu_count')}; torch installed: {hw.get('torch_installed')}; CUDA: {hw.get('cuda_available')}; "
        f"Triton: {hw.get('triton_available')}; scikit-learn: {hw.get('sklearn_available')}",
        f"- Data directory: {e0.get('data_dir')}; datasets audited: {len(e0.get('dataset_audit') or [])}",
        "",
    ]
    if not hw.get("torch_installed"):
        lines += ["Every torch/Triton/GPU claim in this repository is untested on this machine.", ""]


def section_exp01(e1, lines: List[str], verdicts: List[Dict[str, Any]]) -> None:
    lines += ["## Exp 01: scaling-law sweep (H1, H2)", ""]
    if e1 is None or "alpha_laws" not in e1:
        lines += [f"{NOT_RUN} (or produced by the pre-2026-11-25 script, whose output is not comparable).", ""]
        verdicts += [{"id": "H1", "claim": "a d term lowers LODO alpha_0 regret", "verdict": NOT_RUN},
                     {"id": "H2", "claim": "the law transfers with <= 0.01 nats/dim regret", "verdict": NOT_RUN}]
        return
    seeds = len(e1.get("grid", {}).get("seeds", []))
    g = e1.get("grid", {})
    c = e1["censoring"]
    prelim = f" (preliminary: {seeds} seeds per curve, design asks >= {MIN_SEEDS})" if seeds < MIN_SEEDS else ""
    lines += [
        f"- Data: {e1.get('source')}; backend {e1.get('backend')}; mode {e1.get('mode')}; {seeds} seeds averaged per curve point.",
        f"- Axes: d={g.get('dims')}, k={g.get('ks')}, n={g.get('ns')}, B={g.get('batch_sizes')}, T={g.get('epochs')}.",
        f"- Censored alpha_0 optima (excluded from fits): {c['alpha_censored']}/{c['alpha_total']}; "
        f"censored knees: {c['knee_censored']}/{c['knee_total']}.",
        "",
        "| Law | c | a_k | a_B | a_n | a_T | a_d | unidentified | R² (in-sample) | LODO regret (nats/dim) | LOSO regret (nats/dim) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, law in e1["alpha_laws"].items():
        if "error" in law:
            lines.append(f"| {name} | not fitted: {law['error']} |||||||||| ")
            continue
        f = law["fit"]
        lines.append(
            f"| {name} | {_f(f['c_alpha'])} | {_f(f['a_k'], 3)} | {_f(f['a_b'], 3)} | {_f(f['a_n'], 3)} | {_f(f['a_t'], 3)} | "
            f"{_f(f['a_d'], 3)} | {', '.join(f['unidentified']) or '-'} | {_f(f['r2_log'], 3)} | "
            f"{_f(law['lodo_d'].get('lodo_regret_nats_per_dim_mean'), 5)} | {_f(law['loso_n'].get('loso_regret_nats_per_dim_mean'), 5)} |"
        )
    lines += ["", "R² is shown for reference only; H1 and H2 are decided by held-out NLL regret.", ""]

    primary = e1.get("primary_d_law", "ambient_d")
    cmp = e1.get("lodo_paired_regret", {}).get(f"{primary}_vs_no_d", {})
    if "mean_diff" in cmp:
        # Units here are held-out configurations, not seeds: no min-seed rule, but flag the curve seeds.
        v = verdict(cmp["mean_diff"], cmp["p_value"], THRESHOLD)
        lines += [f"**H1 (LODO, {primary} vs no d):** mean regret change {_f(cmp['mean_diff'], 5, True)} nats/dim over "
                  f"{int(cmp['n'])} held-out configurations, p = {_p(cmp['p_value'])}, d-law better on {int(cmp['wins'])}. "
                  f"Verdict: **{v}**{prelim}.",
                  "",
                  "_Caveat: the p-value treats held-out configurations as independent. They share fitted laws, folds and "
                  "data draws, so it is optimistic; read the sign and size of the mean change, not the p-value alone._", ""]
        verdicts.append({"id": "H1", "claim": f"{primary} lowers LODO regret by >= {THRESHOLD} nats/dim", "verdict": v + prelim})
    else:
        verdicts.append({"id": "H1", "claim": "a d term lowers LODO regret", "verdict": "Inconclusive (no paired LODO result)"})

    ood = e1.get("ood", {})
    rows = ood.get("configs", [])
    if rows:
        lines += ["| OOD config | alpha* | censored | " + " | ".join(f"regret {k}" for k in rows[0]["regret_nats_per_dim"]) + " |",
                  "|---|---|---|" + "---|" * len(rows[0]["regret_nats_per_dim"])]
        for r in rows:
            lines.append(f"| {r['config']} | {_f(r['alpha_star'])} | {r['censored']} | "
                         + " | ".join(_f(v, 5) for v in r["regret_nats_per_dim"].values()) + " |")
        regrets = [r["regret_nats_per_dim"].get(primary) for r in rows if r["regret_nats_per_dim"].get(primary) is not None]
        med = float(np.median(regrets)) if regrets else None
        if med is None:
            v2 = "Inconclusive (no OOD regret)"
        elif len(regrets) < 3:
            v2 = f"Inconclusive (only {len(regrets)} OOD configurations; median regret {med:.5f})"
        else:
            v2 = "Supported" if med <= THRESHOLD else "Contradicted"
        lines += ["", f"**H2 (OOD transfer, {primary} law):** median regret {_f(med, 5)} nats/dim over {len(regrets)} "
                  f"configurations. Verdict: **{v2}**. No unseen *dataset* was tested in this run.", ""]
        verdicts.append({"id": "H2", "claim": f"law transfers with median regret <= {THRESHOLD} nats/dim", "verdict": v2})
    else:
        verdicts.append({"id": "H2", "claim": "law transfers", "verdict": NOT_RUN})

    kl = e1.get("knee_laws", {})
    lines += ["Batch-knee laws (uncensored knees only, alpha from the " + str(e1.get("knee_law_alpha_source")) + " law):", ""]
    for name, law in kl.items():
        if "error" in law:
            lines.append(f"- {name}: not fitted ({law['error']})")
        else:
            f = law["fit"]
            lines.append(f"- {name}: c_B={_f(f['c_b'])}, b_k={_f(f['b_k'], 3)}, b_n={_f(f['b_n'], 3)}, b_T={_f(f['b_t'], 3)}, "
                         f"b_d={_f(f['b_d'], 3)}, unidentified={f['unidentified'] or '-'}, n_used={f['n_used']}")
    lines.append("")


def section_exp02(e2, lines: List[str], verdicts: List[Dict[str, Any]]) -> None:
    lines += ["## Exp 02: tuned, matched, paired benchmark (H0, H3, H4)", ""]
    if e2 is None or "suites" not in e2 or not all("paired_vs_reference" in s for s in e2["suites"].values()):
        lines += [f"{NOT_RUN} (or produced by the pre-2026-11-25 script).", ""]
        for h in ("H0", "H3", "H4"):
            verdicts.append({"id": h, "claim": "see design doc", "verdict": NOT_RUN})
        return
    lines += [f"Backend {e2['backend']} (tested locally: {e2.get('backend_tested_locally')}); data {e2['data']}; "
              f"seeds {e2['seeds']}. diff = e-GMM − baseline in nats/dim; negative means e-GMM is better.", ""]
    for sname, res in e2["suites"].items():
        setup = res["setup"]
        lines += [f"### {sname}", "",
                  f"cov={setup['cov_type']}, n_train data from {res['data'].get('source')}, d={res.get('dim')}, k={setup['k']}, "
                  f"B={setup['batch_size']}, epochs={setup['epochs']}. Skipped: {res.get('skipped_methods') or '-'}.", "",
                  "| Method | test NLL/d | dead | tuned setting | e-GMM − method | p | e-GMM wins | verdict: e-GMM better |",
                  "|---|---|---|---|---|---|---|---|"]
        summ, pairs, tuning = res["summary"], res["paired_vs_reference"], res["tuning"]
        claims = {}
        for m, s in summ.items():
            edge = tuning[m].get("at_grid_edge") or []
            tuned = f"{tuning[m]['best_params']}" + (f" (grid edge: {', '.join(edge)})" if edge else "")
            if m == setup["reference"]:
                lines.append(f"| **{m}** (reference) | {_f(s['test_nll_per_dim_mean'])} | {_f(s['dead_count_mean'], 1)} | {tuned} | | | | |")
                continue
            cl = claim_from_comparison(pairs.get(m, {}), flip=True)
            claims[m] = cl
            wins = pairs.get(m, {})
            w = f"{int(wins['n'] - wins['wins'] - wins['ties'])}/{int(wins['n'])}" if "wins" in wins else "-"
            lines.append(f"| {m} | {_f(s['test_nll_per_dim_mean'])} | {_f(s['dead_count_mean'], 1)} | {tuned} | "
                         f"{_f(cl['diff'], 5, True)} | {_p(cl['p'])} | {w} | {cl['verdict']} |")
        lines.append("")

        standard = [m for m in ("online_em", "emeans_then_em3", "fullbatch_em_emeans", "fullbatch_em_random", "sklearn_gmm") if m in claims]
        if standard:
            best = min(standard, key=lambda m: summ[m]["test_nll_per_dim_mean"] if summ[m]["test_nll_per_dim_mean"] is not None else np.inf)
            v0 = claims[best]["verdict"]
            verdicts.append({"id": "H0", "claim": f"[{sname}] e-GMM beats the best tuned standard baseline ({best})",
                             "verdict": ("H0 rejected: " if v0 == "Supported" else "H0 not rejected: ") + v0})
        for m in ("online_em", "fullbatch_em_emeans", "fullbatch_em_random"):
            if m in claims:
                verdicts.append({"id": "H3", "claim": f"[{sname}] e-GMM beats {m} at epochs={setup['epochs']}", "verdict": claims[m]["verdict"]})
        for m in ("emeans_posthoc", "emeans_then_em3"):
            if m in claims:
                verdicts.append({"id": "H4", "claim": f"[{sname}] e-GMM beats {m}", "verdict": claims[m]["verdict"]})
        if "egmm_no_revive" in claims:
            verdicts.append({"id": "H0/H5", "claim": f"[{sname}] spectral revival beats no revival (both alpha_0 tuned)",
                             "verdict": claims["egmm_no_revive"]["verdict"]})
    lines += ["H3 in the design doc is stated for T ∈ {1, 2, 3}; the epochs actually used are listed per suite.", ""]


def section_exp03(e3, lines: List[str], verdicts: List[Dict[str, Any]]) -> None:
    lines += ["## Exp 03: paired ablations (H0, H5, H6, H7)", ""]
    if e3 is None or "unit" not in e3:
        lines += [f"{NOT_RUN} (or produced by the pre-2026-11-25 script).", ""]
        for h in ("H5", "H6a", "H6b", "H7"):
            verdicts.append({"id": h, "claim": "see design doc", "verdict": NOT_RUN})
        return
    lines += [f"Seeds {e3['seeds']}; alpha_0 tuning seeds {e3.get('tuning_seeds')}. diff columns are variant − reference (nats/dim).", ""]

    def table(block: Dict[str, Any], title: str, cols=("test_nll_per_dim", "dead_count", "floor_hit_rate")) -> None:
        ref = block["reference"]
        lines.extend([f"**{title}** (reference: {ref})", "",
                      "| Variant | " + " | ".join(cols) + " | diff vs ref | p |", "|---|" + "---|" * (len(cols) + 2)])
        for v, s in block["summary"].items():
            cmp = block["paired_vs_reference"].get(v, {})
            lines.append(f"| {v} | " + " | ".join(_f(s.get(c), 4) for c in cols)
                         + f" | {_f(cmp.get('mean_diff'), 5, True) if cmp else 'ref'} | {_p(cmp.get('p_value')) if cmp else ''} |")
        lines.append("")

    rev = e3.get("revival")
    if rev:
        for a, block in rev["by_alpha"].items():
            table(block, f"Revival at fixed alpha_0 = {a}")
        table(rev["tuned"], f"Revival at tuned alpha_0 {rev['tuned_alpha']}")
        tried = rev.get("tuning_val_nll_per_dim", {})
        at_edge = {}
        for s, a in rev["tuned_alpha"].items():
            grid = sorted(float(x) for x in tried.get(s, rev["by_alpha"]))
            if len(grid) > 1 and a in (grid[0], grid[-1]):
                at_edge[s] = f"{a} (tried {grid})"
        if at_edge:
            lines += [f"_Tuned alpha_0 still at the edge of its (extended) grid, so the optimum may lie outside it: {at_edge}_", ""]
        ref = rev["tuned"]["reference"]
        for v, cmp in rev["tuned"]["paired_vs_reference"].items():
            cl = claim_from_comparison(cmp, flip=True)
            hid = "H0/H5" if v.startswith("none/") else "H5"
            verdicts.append({"id": hid, "claim": f"{ref} beats {v} (tuned alpha_0)", "verdict": cl["verdict"]})
        a35 = rev["by_alpha"].get("0.35", {}).get("paired_vs_reference", {}).get("none/count")
        if a35:
            cl = claim_from_comparison(a35, flip=True)
            verdicts.append({"id": "H5 (context)", "claim": f"{ref} beats no revival at the old hand-set alpha_0 = 0.35",
                             "verdict": cl["verdict"]})
    else:
        verdicts.append({"id": "H5", "claim": "spectral revival beats alternatives", "verdict": NOT_RUN})

    fl = e3.get("floor")
    if fl:
        table(fl, "Covariance floor lambda_0", cols=("test_nll_per_dim", "floor_hit_rate", "max_condition_number"))

    st = e3.get("structure")
    if st:
        table(st, "Covariance structure", cols=("test_nll_per_dim", "train_bic", "test_bic", "n_params"))
        lines += [f"_{st.get('note', '')}_", ""]
        per = st["per_seed"]
        lr = [v for v in per if v.startswith("low_rank")]
        for v in lr:
            cl = claim_from_comparison(st["paired_vs_reference"][v], flip=False)
            verdicts.append({"id": "H6a", "claim": f"{v} beats diag on test NLL", "verdict": cl["verdict"]})
            if "train_bic" in per[v] and "full" in per:
                cmp = paired_comparison(per[v]["train_bic"], per["full"]["train_bic"])
                vb = verdict(cmp["mean_diff"], cmp["p_value"], 0.0, min_n=MIN_SEEDS, n=cmp["n"])
                verdicts.append({"id": "H6a", "claim": f"{v} has lower training-set BIC than full (diff {cmp['mean_diff']:+.1f})",
                                 "verdict": vb})
    tm = e3.get("topm")
    if tm:
        table(tm, "Top-m responsibilities", cols=("test_nll_per_dim",))
        per = tm["per_seed"]
        for v in per:
            if v == tm["reference"]:
                continue
            diffs = np.asarray(per[v]["test_nll_per_dim"]) - np.asarray(per[tm["reference"]]["test_nll_per_dim"])
            eq = equivalence_verdict(diffs, THRESHOLD, min_n=MIN_SEEDS)
            verdicts.append({"id": "H6b", "claim": f"{v} within ±{THRESHOLD} nats/dim of dense "
                                                   f"(95% CI {_f(eq.get('ci95_low'), 4, True)}..{_f(eq.get('ci95_high'), 4, True)})",
                             "verdict": eq["verdict"]})
    fin = e3.get("finalization")
    if fin:
        for label, block in fin.items():
            table(block, f"Finalization, {label}", cols=("train_nll_per_dim", "test_nll_per_dim"))
            cl = claim_from_comparison(block["paired_vs_reference"].get("slow_ema", {}), flip=False)
            verdicts.append({"id": "H7", "claim": f"[{label}] slow-EMA beats last-epoch on test NLL", "verdict": cl["verdict"]})


def section_exp04(e4, lines: List[str]) -> None:
    lines += ["## Exp 04: CPU E-step timing and float32 precision (no GPU claims)", ""]
    if e4 is None or "numpy_gemm_expansion_vs_naive" not in e4:
        lines += [f"{NOT_RUN} (or produced by the pre-2026-11-25 script).", ""]
        return
    g, w = e4["numpy_gemm_expansion_vs_naive"], e4["numpy_woodbury_vs_cholesky"]
    lines += [
        f"- [x², x] GEMM expansion vs naive broadcast (NumPy, CPU, median): {g['gemm_expansion_ms_median']} ms vs "
        f"{g['naive_broadcast_ms_median']} ms ({g['speedup_vs_naive']}×), max |Δ log p| {g['max_abs_log_diff']:.1e}.",
        f"- Woodbury vs Cholesky (NumPy, CPU, median): {w['woodbury_ms_median']} ms vs {w['cholesky_ms_median']} ms "
        f"({w['speedup_vs_cholesky']}×), max |Δ log p| {w['max_abs_log_diff']:.1e}.",
        f"- torch: {e4['torch_bench']}. Not measured: {', '.join(e4.get('not_measured', []))}.",
        "", "| data mean offset | float32 error, centered | float32 error, uncentered |", "|---|---|---|",
    ]
    for r in e4["float32_precision_vs_mean_offset"]:
        lines.append(f"| {r['mean_offset']:g} | {r['float32_centered_max_abs_err']:.1e} | {r['float32_uncentered_max_abs_err']:.1e} |")
    lines.append("")


def main():
    with tee_experiment_logs("compile_experiment_report"):
        e0 = _load("exp_00_preflight.json")
        e1 = _load("exp_01_phase_a_scaling_laws.json")
        e2 = _load("exp_02_phase_b_benchmarks.json")
        e3 = _load("exp_03_phase_c_ablations.json")
        e4 = _load("exp_04_kernel_bench.json")
        lines = [
            "# Experiment report (e_GMM-scaling)",
            "",
            f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by experiments/compile_experiment_report.py from experiments/results/*.json.",
            "",
            f"Verdict rule (experiments/HYPOTHESES-AND-RESEARCH-DESIGN.md §3): paired by seed, exact sign-flip test, p < 0.05, "
            f"practical threshold {THRESHOLD} nats/dim, at least {MIN_SEEDS} seeds. Smoke runs are small CPU runs on synthetic "
            "data unless stated otherwise; they are evidence about the method at that scale only.",
            "",
        ]
        verdicts: List[Dict[str, Any]] = []
        body: List[str] = []
        section_exp00(e0, body)
        section_exp01(e1, body, verdicts)
        section_exp02(e2, body, verdicts)
        section_exp03(e3, body, verdicts)
        section_exp04(e4, body)

        counts: Dict[str, int] = {}
        for v in verdicts:
            key = v["verdict"].split(" (")[0].split(":")[0]
            counts[key] = counts.get(key, 0) + 1
        lines += ["## Verdicts", "", f"Totals: {counts}", "", "| ID | Claim | Verdict |", "|---|---|---|"]
        lines += [f"| {v['id']} | {v['claim']} | {v['verdict']} |" for v in verdicts]
        lines += [""] + body
        out = RESULTS_ROOT / "FINAL-EXPERIMENT-REPORT.md"
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Wrote {out} ({len(verdicts)} verdicts: {counts})")


if __name__ == "__main__":
    main()
