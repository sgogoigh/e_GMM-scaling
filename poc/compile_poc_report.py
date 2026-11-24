"""
PoC Result Compiler (stdlib + e_gmm.stats): aggregate PoC 1-4 telemetry into poc/results/COMPILED-POC-REPORT.md
==============================================================================================================

Every verdict is computed from the JSON, never written by hand:
  * PoC 2 and PoC 3 verdicts come from `e_gmm.stats.verdict(mean_diff, p_value, threshold=0.01 nats/dim, min_n=10)`
    applied to the paired comparisons stored in the JSON. mean_diff is oriented "claimed-better method minus other"
    (negative = the claim holds).
  * Scaling-law claims are judged by paired leave-one-d-out / out-of-distribution regret of the d-aware law against
    the no-d law. R^2 is not used as evidence.
  * A key absent from the JSON gives "Not run"; censored or unidentified fits are reported as such.
  * PoC 1 identities are deterministic checks against a stated tolerance; PoC 4 is single-seed, so its claims are
    labelled Untested (no paired statistics exist in that JSON).
"""

from datetime import datetime
import json
import math
import pathlib
import sys

POC_DIR = pathlib.Path(__file__).resolve().parent
ROOT = POC_DIR.parent
RESULTS_DIR = POC_DIR / "results"
sys.path.insert(0, str(ROOT / "src"))

from e_gmm.poc_logging import tee_poc_logs  # noqa: E402
from e_gmm.stats import verdict  # noqa: E402

THRESHOLD = 0.01  # nats per dimension: smallest difference worth acting on
MIN_N = 10        # minimum paired observations for a Supported/Contradicted verdict
IDENTITY_TOL = 1e-10
NOT_RUN = "Not run"


def _load_json(name: str):
    p = RESULTS_DIR / name
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def _get(obj, *path):
    for key in path:
        if not isinstance(obj, dict) or key not in obj:
            return None
        obj = obj[key]
    return obj


def _fmt(x, nd=4):
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def _verdict_from(cmp, flip=False):
    """Verdict for 'the claimed method beats the other'. `cmp` is a stored paired_comparison (method - reference).
    flip=True when the claim is about the reference (claimed = reference), so mean_diff must be negated."""
    if not isinstance(cmp, dict) or "mean_diff" not in cmp:
        return NOT_RUN if cmp is None else "Inconclusive (no paired comparison)"
    md = cmp["mean_diff"]
    md = -md if flip else md
    return verdict(md, cmp["p_value"], THRESHOLD, min_n=MIN_N, n=cmp.get("n"))


def _cmp_cells(cmp, flip=False):
    if not isinstance(cmp, dict) or "mean_diff" not in cmp:
        return "n/a", "n/a", "n/a"
    md = -cmp["mean_diff"] if flip else cmp["mean_diff"]
    return f"{md:+.4f}", _fmt(cmp["p_value"], 4), f"{int(cmp['n'])}"


def _poc1(p1, lines, matrix):
    ident = p1["identities"]
    f, w = ident["flash_gemm_max_log_diff"], ident["woodbury_max_log_diff"]
    v11 = "Supported (deterministic check)" if max(f, w) < IDENTITY_TOL else "Contradicted (tolerance exceeded)"
    v12 = ("Supported (deterministic check)"
           if ident["rank1_batch_min_eig"] >= ident["rank1_batch_floor"] * (1 - 1e-6) else "Contradicted")
    matrix.append(f"| H1.1 | 2d-GEMM Flash-GMM and Woodbury determinant lemma match naive log-densities (tol {IDENTITY_TOL:.0e}) | "
                  f"PoC 1 (`{f:.2e}`, `{w:.2e}`) | **{v11}** |")
    matrix.append(f"| H1.2 | Additive floor keeps min eigenvalue >= lambda_0 on a rank-1 batch | "
                  f"PoC 1 (`{ident['rank1_batch_min_eig']:.3e}` vs `{ident['rank1_batch_floor']:.1e}`) | **{v12}** |")
    wall = p1["full_cov_wall_k65k_d1024"]
    lines += [
        "## PoC 1: Mathematical identities, PSD floor and memory arithmetic", "",
        f"- Flash-GMM 2d-GEMM max |delta ln p|: `{f:.3e}`; Woodbury max |delta ln p|: `{w:.3e}`; tolerance `{IDENTITY_TOL:.0e}`",
        f"- Rank-1 batch min eigenvalue `{ident['rank1_batch_min_eig']:.6e}` (floor `{ident['rank1_batch_floor']:.1e}`)",
        f"- Spectral split mean conservation error `{ident['spectral_split_mean_diff']:.3e}`",
        f"- Full-covariance state at k=65536, d=1024: `{wall['GiB']} GiB` (arithmetic, not a measurement)", "",
        "| Preset | Cov | k | d | B | State (GiB) | Naive BxK HBM (GiB) | Tiled HBM (GiB) |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in p1["scale_profiles"]:
        lines.append(f"| `{s['preset']}` | `{s['cov_type']}` | {s['k']} | {s['d']} | {s['B']} | "
                     f"{s['state_fp32_GiB']} | {s['naive_bk_HBM_GiB']} | {s['tiled_HBM_GiB']} |")
    lines += ["", "The memory table is closed-form arithmetic on the stated presets. Nothing here was run on a GPU, and no claim "
              "about speed or about real-data accuracy follows from it (untested).", ""]


def _poc2(p2, lines, matrix):
    regimes = p2.get("regimes") or {}
    cfg = p2.get("config", {})
    lines += [
        "---", "", "## PoC 2: Matched tuned baselines and alpha_0 sensitivity", "",
        f"Setup: d={cfg.get('dim')}, k={cfg.get('num_components')}, B={cfg.get('batch_size')}, epochs={cfg.get('epochs')}, "
        f"train/val/test = {cfg.get('n_train')}/{cfg.get('n_val')}/{cfg.get('n_test')}, seeds = {len(p2.get('seeds', []))}, full covariance, "
        "floor 1e-4 for every method, each method tuned on validation with a separate tuning seed. Unit: test NLL in nats/dim. "
        f"Paired difference column is (egmm_spectral - baseline): negative means e-GMM is better. Verdict uses threshold {THRESHOLD} nats/dim, "
        f"min_n={MIN_N}, alpha 0.05, no multiple-comparison correction ({sum(len(_get(r, 'benchmark', 'paired_vs_reference') or {}) for r in regimes.values())} "
        "benchmark comparisons plus the sensitivity block).", "",
    ]
    for rname in ("zipf", "overlap"):
        reg = regimes.get(rname)
        if reg is None:
            lines += [f"### {rname}: {NOT_RUN}", ""]
            matrix.append(f"| H2.{rname} | e-GMM spectral beats every tuned baseline on {rname} | PoC 2 | **{NOT_RUN}** |")
            continue
        bench = reg["benchmark"]
        summ = bench["summary"]
        lines += [
            f"### Regime `{rname}` ({json.dumps(reg['generator'])})", "",
            "| Baseline | n | egmm_spectral - baseline (nats/dim) | p (sign-flip) | baseline mean dead | Verdict: e-GMM better |",
            "| :--- | ---: | ---: | ---: | ---: | :--- |",
        ]
        ref = bench["setup"]["reference"]
        lines.append(f"| `{ref}` (reference) | {summ[ref]['n_finite']} | - | - | {_fmt(summ[ref]['dead_count_mean'], 1)} | - |")
        for m, c in bench["paired_vs_reference"].items():
            # stored diff = baseline - egmm_spectral; the claim "egmm better" has diff egmm - baseline = -stored
            md, p, n = _cmp_cells(c, flip=True)
            lines.append(f"| `{m}` | {n} | {md} | {p} | {_fmt(summ[m]['dead_count_mean'], 1)} | {_verdict_from(c, flip=True)} |")
            matrix.append(f"| H2.{rname}.{m} | e-GMM spectral better than tuned `{m}` on {rname} | PoC 2 (diff {md}, p {p}, n {n}) | "
                          f"**{_verdict_from(c, flip=True)}** |")
        fails = {k: v for k, v in (bench.get("failures") or {}).items() if v}
        skipped = bench.get("skipped_methods") or {}
        if fails:
            lines += ["", f"Fit failures (excluded from pairs): `{json.dumps(fails)}`"]
        if skipped:
            lines += ["", f"Methods not run: `{json.dumps(skipped)}`"]
        lines += ["", "Tuned settings: " + "; ".join(f"`{m}`={t.get('best_params')}" for m, t in bench["tuning"].items()), ""]

        sens = reg.get("alpha_sensitivity")
        if sens is None:
            lines += [f"alpha_0 sensitivity: {NOT_RUN}", ""]
            continue
        lines += [
            f"#### alpha_0 sensitivity on `{rname}`: is revival only needed because alpha_0 is too large?", "",
            "| alpha_0 | spectral NLL | no-revive NLL | spectral - no-revive | p | dead (spectral / none) | Verdict: revival helps |",
            "| ---: | ---: | ---: | ---: | ---: | :---: | :--- |",
        ]
        helps_small = None
        for a, r in sens.items():
            c = r["no_revive_vs_spectral"]
            md, p, n = _cmp_cells(c, flip=True)
            v = _verdict_from(c, flip=True)
            lines.append(f"| {a} | {_fmt(c.get('reference_mean'))} | {_fmt(c.get('method_mean'))} | {md} | {p} | "
                         f"{r['dead_spectral_mean']:.1f} / {r['dead_no_revive_mean']:.1f} | {v} |")
            matrix.append(f"| H2.{rname}.alpha{a} | spectral revival beats no revival at alpha_0={a} on {rname} | PoC 2 (diff {md}, p {p}) | **{v}** |")
            if a == min(sens, key=float):
                helps_small = v
        lines += ["", f"At the smallest alpha_0 the revival verdict is **{helps_small}**. "
                  "If revival only helps at large alpha_0 the H0 (revival compensates for a too-large step) is not rejected for that regime.", ""]
    lines.append("")


def _poc3(p3, lines, matrix):
    lines += ["---", "", "## PoC 3: Does a dimension term improve alpha_0 / B_knee prediction? (synthetic, small grid)", ""]
    grid = p3.get("grid", {})
    cens = p3.get("censoring", {})
    lines += [
        f"- Grid: dims={grid.get('dims')}, ks={grid.get('ks')}, ns={grid.get('ns')}, B={grid.get('batch_sizes')}, T={grid.get('epochs')}, "
        f"seeds={grid.get('seeds')}. {p3.get('shrunk_note', '')}",
        f"- Alpha-ladder optima censored (on the ladder edge): **{cens.get('alpha_censored', 'n/a')} / {cens.get('alpha_total', 'n/a')}**; "
        f"knee optima censored: **{cens.get('knee_censored', 'n/a')} / {cens.get('knee_total', 'n/a')}**. "
        f"Skipped configs: {len(p3.get('skipped_configs') or [])}.",
        f"- Intrinsic dims (TwoNN): `{json.dumps(p3.get('intrinsic_dims'))}`", "",
        "| Law | n used | censored excluded | exponents (k, B, n, T, d) | unidentified | LOSO-n | LODO-d |",
        "| :--- | ---: | ---: | :--- | :--- | :--- | :--- |",
    ]
    for name, law in (p3.get("alpha_laws") or {}).items():
        if "fit" not in law:
            lines.append(f"| `{name}` | - | - | error: {law.get('error')} | - | - | - |")
            continue
        f = law["fit"]
        ex = ", ".join(_fmt(f.get(k), 2) for k in ("a_k", "a_b", "a_n", "a_t", "a_d"))
        lines.append(f"| `{name}` | {f.get('n_used')} | {f.get('n_censored_excluded')} | {ex} | {f.get('unidentified') or 'none'} | "
                     f"`{json.dumps(law.get('loso_n'))}` | `{json.dumps(law.get('lodo_d'))}` |")
    unid = [(n, l["fit"]["unidentified"]) for n, l in (p3.get("alpha_laws") or {}).items() if l.get("fit", {}).get("unidentified")]
    if unid:
        lines += ["", f"**Unidentified exponents**: {unid}. Any claim about those exponents is not supported by this sweep."]
    lines.append("")

    # Judged by paired regret, never R^2.
    lodo = p3.get("lodo_paired_regret")
    ood_paired = _get(p3, "ood", "paired_vs_no_d")
    lines += ["### Paired regret of d-aware law vs no-d law (nats/dim; negative = d term helps)", "",
              "| Test | Comparison | n | mean regret diff | p | Verdict: d term beats no-d |", "| :--- | :--- | ---: | ---: | ---: | :--- |"]
    for label, src in (("LODO (held-out d)", lodo), ("OOD configs", ood_paired)):
        if src is None:
            lines.append(f"| {label} | all | - | - | - | {NOT_RUN} |")
            matrix.append(f"| H3.{label.split()[0]} | d term lowers alpha_0 regret | PoC 3 | **{NOT_RUN}** |")
            continue
        for comp, c in src.items():
            md, p, n = _cmp_cells(c)
            v = _verdict_from(c)
            lines.append(f"| {label} | `{comp}` | {n} | {md} | {p} | {v} |")
            matrix.append(f"| H3.{label.split()[0]}.{comp} | d-aware alpha_0 law has lower paired regret than no-d | PoC 3 (diff {md}, p {p}, n {n}) | **{v}** |")
    lines += ["", "Caveats: LODO points are held-out (config, d) cells pooled over the grid, so they are not independent draws and the p-value is "
              "optimistic. `verdict` labels a significant wrong-direction difference Contradicted even when it is far below the "
              f"{THRESHOLD} nats/dim practical threshold, so read the size column. For OOD, n counts configs (not seeds), "
              "hence the min_n rule makes it Inconclusive."]
    ood_rows = _get(p3, "ood", "configs") or []
    if ood_rows:
        lines += ["", "OOD configs (regret in nats/dim at each law's predicted alpha_0; `extrapolated` means the prediction fell outside the measured ladder):", "",
                  "| config | alpha* | censored | " + " | ".join(sorted(ood_rows[0].get("regret_nats_per_dim", {}))) + " |",
                  "| :--- | ---: | :---: | " + " | ".join("---:" for _ in ood_rows[0].get("regret_nats_per_dim", {})) + " |"]
        for r in ood_rows:
            names = sorted(r.get("regret_nats_per_dim", {}))
            lines.append(f"| `{json.dumps(r['config'])}` | {_fmt(r.get('alpha_star'))} | {r.get('censored')} | "
                         + " | ".join(_fmt(r['regret_nats_per_dim'][n_]) + ("*" if (r.get('extrapolated') or {}).get(n_) else "") for n_ in names) + " |")
        lines.append("")
        lines.append("`*` = extrapolated beyond the measured alpha ladder.")
    lines.append("")

    klaw = p3.get("knee_laws")
    lines += ["### B_knee law", ""]
    if klaw is None:
        lines += [f"{NOT_RUN}", ""]
        matrix.append(f"| H3.knee | B_knee scales with d | PoC 3 | **{NOT_RUN}** |")
    else:
        bits = []
        for name, kl in klaw.items():
            f = kl.get("fit")
            bits.append(f"`{name}`: " + (f"n_used={f.get('n_used')}, censored excluded={f.get('n_censored_excluded')}, unidentified={f.get('unidentified') or 'none'}"
                                          if f else f"error {kl.get('error')}"))
        lines += ["; ".join(bits), "", f"Knee skip reason: `{p3.get('knee_skip_reason')}`.",
                  "Knee optima that are censored or unidentified carry no claim about B_knee scaling. No paired regret comparison is stored for the knee law, so it is **Untested**.", ""]
        matrix.append("| H3.knee | B_knee depends on d | PoC 3 (no paired regret stored) | **Untested** |")


def _poc4(p4, lines, matrix):
    lines += [
        "---", "", "## PoC 4: Top-m sparsity, low-rank Woodbury, finalization (single seed, descriptive)", "",
        "This JSON holds one run per setting, with no seeds and no paired statistics, so none of its claims can receive a verdict. "
        "The numbers are shown for inspection; every claim below is **Untested**.", "",
        "| Top-m | M-step work | Test NLL | Fit ms |", "| :---: | :---: | :---: | ---: |",
    ]
    for r in p4.get("top_m_ablation", []):
        lines.append(f"| `{r['top_m']}` | {r['m_step_work_ratio']}x | {r['test_nll']:.4f} | {r['fit_ms']} |")
    lines += ["", "| Cov mode | params | Test NLL | Test BIC |", "| :--- | ---: | ---: | ---: |"]
    for r in p4.get("covariance_mode_ablation", []):
        lines.append(f"| `{r['cov_type']}` | {r['n_params']} | {r['test_nll']:.4f} | {r['test_bic']:.1f} |")
    lines += ["", "| Finalizer | Train NLL | Test NLL |", "| :--- | :---: | :---: |"]
    for r in p4.get("finalization_ablation", []):
        lines.append(f"| `{r['finalizer']}` | {r['train_nll']:.4f} | {r['test_nll']:.4f} |")
    lines += ["", "Torch parity: " + ("recorded" if p4.get("torch_parity") else "**Not run** (no torch result in the JSON)."), ""]
    for hid, claim in (("H4.1", "Top-m sparse E-step preserves accuracy at m/k M-step cost"),
                       ("H4.2", "Low-rank+diag has the best BIC on rotated embeddings"),
                       ("H4.3", "last_epoch wins train NLL while slow_ema wins test NLL")):
        matrix.append(f"| {hid} | {claim} | PoC 4 (single seed) | **Untested** |")


def main():
    with tee_poc_logs("compile_report"):
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        p = {i: _load_json(f"poc_0{i}_results.json") for i in (1, 2, 3, 4)}
        completed = [f"PoC {i}" for i, o in p.items() if o is not None]
        header = [
            "# Compiled Proof-of-Concept Evidence Report (`e_GMM-scaling`)", "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
            f"**Completed Tiers**: {', '.join(completed) if completed else 'None yet'}  ",
            f"**Verdict rule**: `e_gmm.stats.verdict`, threshold {THRESHOLD} nats/dim, min_n {MIN_N}, alpha 0.05, exact paired sign-flip p. "
            "Synthetic data only; nothing here supports a claim about real embeddings.", "", "---", "",
        ]
        matrix = []
        body = []
        for i, fn in ((1, _poc1), (2, _poc2), (3, _poc3), (4, _poc4)):
            if p[i] is None:
                body += ["---", "", f"## PoC {i}: {NOT_RUN}", ""]
                matrix.append(f"| H{i}.* | all PoC {i} claims | - | **{NOT_RUN}** |")
            else:
                fn(p[i], body, matrix)
        table = ["## Hypothesis verdict matrix", "", "| ID | Claim | Evidence | Verdict |", "| :--- | :--- | :--- | :--- |"] + matrix + ["", "---", ""]
        out_md = RESULTS_DIR / "COMPILED-POC-REPORT.md"
        out_md.write_text("\n".join(header + table + body), encoding="utf-8")
        print(f"Compiled PoC report -> {out_md}")


if __name__ == "__main__":
    main()
