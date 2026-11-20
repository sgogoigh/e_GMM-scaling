"""
PoC Result Compiler (Any Machine, Python Stdlib Only): Aggregate PoC 1-4 Telemetry
===================================================================================
"""

from datetime import datetime
import json
import pathlib
import sys

POC_DIR = pathlib.Path(__file__).resolve().parent
ROOT = POC_DIR.parent
RESULTS_DIR = POC_DIR / "results"
sys.path.insert(0, str(ROOT / "src"))

from e_gmm.poc_logging import tee_poc_logs  # noqa: E402


def _load_json(name: str):
    p = RESULTS_DIR / name
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def main():
    with tee_poc_logs("compile_report"):
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        p1 = _load_json("poc_01_results.json")
        p2 = _load_json("poc_02_results.json")
        p3 = _load_json("poc_03_results.json")
        p4 = _load_json("poc_04_results.json")

        completed = [lbl for lbl, obj in [("PoC 1", p1), ("PoC 2", p2), ("PoC 3", p3), ("PoC 4", p4)] if obj is not None]
        lines = [
            "# Compiled Proof-of-Concept Evidence Report (`e_GMM-scaling`)",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
            f"**Completed Tiers**: {', '.join(completed) if completed else 'None yet'}",
            "",
            "---",
            "",
        ]

        if p1:
            ident = p1["identities"]
            wall = p1["full_cov_wall_k65k_d1024"]
            lines += [
                "## PoC 1: Exact Mathematical Identities, PSD Floor & Memory/HBM Audit",
                "",
                f"- **Flash-GMM 2d-GEMM Max $|\\Delta \\ln p|$**: `{ident['flash_gemm_max_log_diff']:.3e}`",
                f"- **Woodbury + Determinant Lemma Max $|\\Delta \\ln p|$**: `{ident['woodbury_max_log_diff']:.3e}`",
                f"- **Rank-1 Collapsed Batch Min Eigenvalue**: `{ident['rank1_batch_min_eig']:.6e}` (floor `{ident['rank1_batch_floor']:.1e}`)",
                f"- **Spectral Split Mean Conservation Error**: `{ident['spectral_split_mean_diff']:.3e}`",
                f"- **Full-Covariance Memory Wall ($k=65536, d=1024$)**: `{wall['GiB']} GiB` (`{wall['GB_decimal']} GB`)",
                "",
                "| Preset | Cov Mode | $k$ | $d$ | $B$ | State (GiB) | Naive $B\\times k$ HBM (GiB) | Tiled HBM (GiB) |",
                "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
            for s in p1["scale_profiles"]:
                lines.append(
                    f"| `{s['preset']}` | `{s['cov_type']}` | {s['k']} | {s['d']} | {s['B']} | "
                    f"{s['state_fp32_GiB']} | {s['naive_bk_HBM_GiB']} | {s['tiled_HBM_GiB']} |"
                )
            lines.append("")

        if p2:
            lines += [
                "---",
                "",
                "## PoC 2: Collapse, Spectral Splitting & Overlapping Soft-EM Advantage",
                "",
                "### Regime A: Zipf-1.2 Rotated Anisotropic GMM (Extreme Class Imbalance)",
                "| Method | Splits | Dead | Max/Mean | Norm Entropy | Sing/Floor Ev | Train NLL | Test NLL | Test BIC |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | ---: |",
            ]
            for r in p2["runs"]:
                lines.append(
                    f"| `{r['method']}` | {r['splits']} | {r['dead_components']} | {r['max_to_mean']}x | "
                    f"{r['norm_entropy']} | {r['singular_or_floor_events']} | {r['train_nll']:.4f} | {r['test_nll']:.4f} | {r['test_bic']:.1f} |"
                )
            if p2.get("overlapping_runs"):
                lines += [
                    "",
                    "### Regime B: Overlapping Rotated Anisotropic GMM ($\\sigma_\\mu^2 = 0.2, \\sigma_w^2 = 0.8$)",
                    "| Method | Splits | Dead | Max/Mean | Norm Entropy | Test NLL | Test BIC |",
                    "| :--- | :---: | :---: | :---: | :---: | :---: | ---: |",
                ]
                for r in p2["overlapping_runs"]:
                    lines.append(
                        f"| `{r['method']}` | {r['splits']} | {r['dead_components']} | {r['max_to_mean']}x | "
                        f"{r['norm_entropy']} | {r['test_nll']:.4f} | {r['test_bic']:.1f} |"
                    )
            lines.append("")

        if p3:
            fw = p3["fit_with_d"]
            fnd = p3["fit_no_d"]
            fb = p3.get("fit_batch_knee")
            ood = p3["ood_unseen_d_transfer"]
            lines += [
                "---",
                "",
                "## PoC 3: Empirical Dimension-Aware Scaling Law & Unseen-$d$ Transfer",
                "",
                f"- **Sweep Size**: {p3['num_observations']} $(n, d, k, B)$ configurations on `StretchedGaussian`",
                f"- **Fitted $\\alpha_0$ Law ($d$-Aware)**: $\\alpha_0 = {fw['c_alpha']} \\cdot k^{{{fw['a_k']:+.2f}}} B^{{{fw['a_b']:+.2f}}} n^{{{fw['a_n']:+.2f}}} d^{{{fw['a_d']:+.2f}}}$ ($R^2 = {fw['r2_log']}$ vs. `{fnd['r2_log']}` without $d$)",
                f"- **LODO-$d$ Rel Error**: `{fw['lodo_mean_rel_err_pct']}%` (with $d^{{a_d}}$) vs. `{fnd['lodo_mean_rel_err_pct']}%` (without $d$)",
                f"- **LOSO-$n$ Rel Error**: `{fw['loso_mean_rel_err_pct']}%`",
            ]
            if fb:
                lines.append(
                    f"- **Fitted $B_{{\\text{{knee}}}}$ Law ($d$-Aware)**: $B_{{\\text{{knee}}}} = {fb['c_b']} \\cdot k^{{{fb['b_k']:+.2f}}} n^{{{fb['b_n']:+.2f}}} d^{{{fb['b_d']:+.2f}}}$ ($R^2 = {fb['r2_log']}$)"
                )
            lines += [
                f"- **Zero-Shot Unseen $d=24$ Transfer**: Oracle NLL `{ood['oracle_nll']:.4f}` | With-$d$ Law `{ood['with_d_nll']:.4f}` (`{ood['with_d_gap_nats']:+.4f}` nats, $\\alpha_0={ood['with_d_pred_alpha0']}$) | No-$d$ Law `{ood['no_d_nll']:.4f}` (`{ood['no_d_gap_nats']:+.4f}` nats, $\\alpha_0={ood['no_d_pred_alpha0']}$)",
                "",
            ]

        if p4:
            lines += [
                "---",
                "",
                "## PoC 4: Top-$m$ Sparsity, Low-Rank Woodbury & Finalization Ablations",
                "",
                "### Top-$m$ Sparse Responsibility Truncation (`k=32`)",
                "| Top-$m$ | M-Step Work Ratio | Test NLL | Fit Time (ms) |",
                "| :---: | :---: | :---: | ---: |",
            ]
            for r in p4["top_m_ablation"]:
                lines.append(f"| `{r['top_m']}` | {r['m_step_work_ratio']}x | {r['test_nll']:.4f} | {r['fit_ms']} |")

            lines += [
                "",
                "### Rotated Correlated Embeddings (`diag` vs `low_rank_diag` vs `full`)",
                "| Covariance Mode | Effective Params | Test NLL | Test BIC |",
                "| :--- | ---: | :---: | ---: |",
            ]
            for r in p4["covariance_mode_ablation"]:
                lines.append(f"| `{r['cov_type']}` | {r['n_params']} | {r['test_nll']:.4f} | {r['test_bic']:.1f} |")

            lines += [
                "",
                "### Finalization Strategy (`main_ema` vs `last_epoch` vs `slow_ema`)",
                "| Finalizer | Train NLL | Test NLL |",
                "| :--- | :---: | :---: |",
            ]
            for r in p4["finalization_ablation"]:
                lines.append(f"| `{r['finalizer']}` | {r['train_nll']:.4f} | {r['test_nll']:.4f} |")
            lines.append("")

        out_md = RESULTS_DIR / "COMPILED-POC-REPORT.md"
        out_md.write_text("\n".join(lines), encoding="utf-8")
        print(f"Compiled PoC report -> {out_md}")


if __name__ == "__main__":
    main()
