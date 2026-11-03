"""Tables and figures from the diagnostic and baseline JSON files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from osr_static.jobs import atomic_json
from osr_static.paths import REPO_ROOT, RESULTS_STATIC


def _curve(row: dict, *keys: str) -> tuple[np.ndarray, ...] | None:
    """Load named arrays from the ``.npz`` saved next to a result JSON.

    The baseline JSON files store the scalar OSCR. The curve arrays were not
    kept, so a missing ``.npz`` means that series is left off the figure.
    """
    path = Path(row["_path"]).with_suffix(".npz")
    if not path.is_file():
        return None
    with np.load(path) as saved:
        return tuple(np.asarray(saved[key]) for key in keys)


def _load(pattern: str) -> list[dict]:
    rows = []
    root = RESULTS_STATIC
    for path in sorted(root.glob(pattern)):
        if path.name.endswith(".job.json"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("ok", True):
            payload["_path"] = str(path)
            rows.append(payload)
    return rows


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in values if v is not None and v == v], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1))


def _fmt(mean: float, std: float) -> str:
    if mean != mean:
        return ""
    if std != std:
        return f"{mean:.2f}"
    return f"{mean:.2f} $\\pm$ {std:.2f}"


def _fmt_md(mean: float, std: float) -> str:
    return _fmt(mean, std).replace("$\\pm$", "±")


def _group(rows: list[dict], key_fn) -> dict:
    grouped: dict = {}
    for row in rows:
        grouped.setdefault(key_fn(row), []).append(row)
    return grouped


def _tex_table(caption: str, header: list[str], body: list[list[str]]) -> str:
    align = "l" + "c" * (len(header) - 1)
    lines = [
        "\\begin{table}",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]
    for row in body:
        lines.append(" & ".join(row) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def _write_pair(path: Path, header: list[str], body: list[list[str]], caption: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    csv_lines = [",".join(header)]
    for row in body:
        csv_lines.append(",".join(cell.replace(",", ";") for cell in row))
    path.with_suffix(".csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    path.with_suffix(".tex").write_text(_tex_table(caption, header, body), encoding="utf-8")


def _paper() -> dict[tuple[str, float], tuple[float, float]]:
    import csv

    table = {}
    paper_csv = REPO_ROOT / "reference" / "table1_paper.csv"
    if not paper_csv.is_file():
        return {}
    with open(paper_csv, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            table[(row["dataset"], round(float(row["openness_percent"]), 2))] = (
                float(row["micro_f1_mean"]),
                float(row["micro_f1_std"]),
            )
    return table


def _strict_curve(fpr: np.ndarray, ccr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fpr = np.asarray(fpr, dtype=float)
    ccr = np.asarray(ccr, dtype=float)
    _rev, rev_index = np.unique(fpr[::-1], return_index=True)
    index = np.sort(len(fpr) - 1 - rev_index)
    return fpr[index], ccr[index]


def _interp_band(curves: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = np.linspace(0.0, 1.0, 201)
    stacked = []
    for fpr, ccr in curves:
        fpr, ccr = _strict_curve(fpr, ccr)
        stacked.append(np.interp(grid, fpr, ccr))
    arr = np.vstack(stacked)
    std = np.zeros(grid.shape) if arr.shape[0] == 1 else arr.std(axis=0, ddof=1)
    return grid, arr.mean(axis=0), std


def _plot_oscr(diag: list[dict], base: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = ["-", "--", "-.", ":"]
    out = RESULTS_STATIC / "figures"
    out.mkdir(parents=True, exist_ok=True)
    keys = sorted({(r["dataset"], round(float(r["nominal_openness"]), 2)) for r in diag if float(r["actual_openness_percent"]) > 0})
    for dataset, nominal in keys:
        plt.figure(figsize=(3.5, 2.8))
        series = []
        for row in diag:
            if row["dataset"] == dataset and abs(float(row["nominal_openness"]) - nominal) < 1e-6:
                curve = _curve(row, "oscr_1nn_fpr", "oscr_1nn_ccr")
                if curve is not None:
                    series.append(curve)
        if series:
            grid, mean, std = _interp_band(series)
            plt.plot(grid, mean, styles[0], color="black", label="1_NN")
            plt.fill_between(grid, mean - std, mean + std, color="0.75", linewidth=0)
        slot = 1
        methods = sorted({r["method"] for r in base})
        for method in methods:
            curves = []
            for row in base:
                if row["dataset"] == dataset and abs(float(row["nominal_openness"]) - nominal) < 1e-6 and row["method"] == method:
                    curve = _curve(row, "oscr_fpr", "oscr_ccr")
                    if curve is not None:
                        curves.append(curve)
            if not curves:
                continue
            grid, mean, std = _interp_band(curves)
            plt.plot(grid, mean, styles[slot % len(styles)], color="black", label=method)
            slot += 1
        plt.xlabel("FPR")
        plt.ylabel("CCR")
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        plt.legend(fontsize=7, frameon=False)
        plt.tight_layout()
        plt.savefig(out / f"e1_oscr_{dataset}_O{nominal:.2f}.pdf")
        plt.close()


def _plot_e2(diag: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = RESULTS_STATIC / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for dataset in sorted({r["dataset"] for r in diag}):
        plt.figure(figsize=(3.5, 2.8))
        rows = [r for r in diag if r["dataset"] == dataset and float(r["actual_openness_percent"]) > 0]
        openness_values = sorted({round(float(r["nominal_openness"]), 2) for r in rows})
        styles = ["-", "--", "-.", ":"]
        for index, nominal in enumerate(openness_values):
            subset = [r for r in rows if abs(float(r["nominal_openness"]) - nominal) < 1e-6]
            if not subset:
                continue
            qs = sorted(float(q) for q in subset[0]["val_f1_by_q"])
            means = []
            stds = []
            for q in qs:
                mean, std = _mean_std([r["val_f1_by_q"][f"{q:.3f}"] for r in subset])
                means.append(mean)
                stds.append(0.0 if std != std else std)
            means_arr = np.asarray(means)
            std_arr = np.asarray(stds)
            plt.plot(qs, means_arr, styles[index % len(styles)], color="black", label=f"O {nominal:.2f}")
            plt.fill_between(qs, means_arr - std_arr, means_arr + std_arr, color="0.8", linewidth=0)
        plt.xlabel("q")
        plt.ylabel("validation micro-F1")
        plt.legend(fontsize=7, frameon=False)
        plt.tight_layout()
        plt.savefig(out / f"e2_val_curve_{dataset}.pdf")
        plt.close()


def _plot_e3(diag: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = RESULTS_STATIC / "figures"
    out.mkdir(parents=True, exist_ok=True)
    keys = sorted({(r["dataset"], round(float(r["nominal_openness"]), 2)) for r in diag if r.get("oracle_tau_1nn") is not None})
    for dataset, nominal in keys:
        subset = [r for r in diag if r["dataset"] == dataset and abs(float(r["nominal_openness"]) - nominal) < 1e-6]
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
        for axis, key, title in (
            (axes[0], "1nn", "1_NN score, ORACLE"),
            (axes[1], "radius85", "radius s_nn q=0.85, ORACLE"),
        ):
            curves = []
            for row in subset:
                loaded = _curve(row, f"tau_curve_{key}", f"f1_curve_{key}")
                if loaded is None:
                    continue
                tau, f1 = loaded
                finite = np.isfinite(tau)
                if finite.sum() < 2:
                    continue
                curves.append((tau[finite], f1[finite]))
            if not curves:
                continue
            lo = min(float(c[0].min()) for c in curves)
            hi = max(float(np.quantile(c[0], 0.99)) for c in curves)
            grid = np.linspace(lo, max(hi, lo + 1e-6), 200)
            stacked = [np.interp(grid, tau, f1) for tau, f1 in curves]
            arr = np.vstack(stacked)
            axis.plot(grid, arr.mean(axis=0), "-", color="black")
            axis.axvline(1.0, color="black", linestyle="--", linewidth=0.8)
            axis.set_title(title, fontsize=8)
            axis.set_xlabel("tau")
        axes[0].set_ylabel("micro-F1")
        fig.tight_layout()
        fig.savefig(out / f"e3_f1_vs_threshold_ORACLE_{dataset}_O{nominal:.2f}.pdf")
        plt.close(fig)


def run_aggregate() -> dict:
    diag = _load("diagnostics/raw/*/*/*/1_NN.json")
    base = _load("baselines/raw/*/*/*/*.json")
    paper = _paper()
    tables = RESULTS_STATIC / "tables"
    e1_header = ["dataset", "openness", "method", "variant", "AUROC", "AUPR", "OSCR"]
    e1_body = []
    variants = (
        ("1_NN", "primary", "auroc_1nn", "aupr_1nn", "oscr_1nn"),
        ("1_NN", "s_all q=0.85", "auroc_s_all_q85", "aupr_s_all_q85", "oscr_s_all_q85"),
        ("1_NN", "s_nn q=0.85", "auroc_s_nn_q85", "aupr_s_nn_q85", "oscr_s_nn_q85"),
        ("1_NN", "s_raw", "auroc_s_raw", "aupr_s_raw", "oscr_s_raw"),
    )
    cells = sorted({(r["dataset"], round(float(r["nominal_openness"]), 2)) for r in diag})
    for dataset, nominal in cells:
        subset = [r for r in diag if r["dataset"] == dataset and abs(float(r["nominal_openness"]) - nominal) < 1e-6]
        if float(subset[0]["actual_openness_percent"]) <= 0:
            continue
        for method, variant, a_key, p_key, o_key in variants:
            e1_body.append([
                dataset,
                f"{nominal:.2f}",
                method,
                variant,
                _fmt_md(*_mean_std([r[a_key] for r in subset])),
                _fmt_md(*_mean_std([r[p_key] for r in subset])),
                _fmt_md(*_mean_std([r[o_key] for r in subset])),
            ])
        for method in sorted({r["method"] for r in base}):
            brows = [r for r in base if r["dataset"] == dataset and abs(float(r["nominal_openness"]) - nominal) < 1e-6 and r["method"] == method]
            if not brows:
                continue
            e1_body.append([
                dataset, f"{nominal:.2f}", method, "reimplemented",
                _fmt_md(*_mean_std([r["auroc"] for r in brows])),
                _fmt_md(*_mean_std([r["aupr"] for r in brows])),
                _fmt_md(*_mean_std([r["oscr"] for r in brows])),
            ])
    _write_pair(tables / "e1_auroc_oscr", e1_header, e1_body, "E1 threshold-free metrics. Reimplemented baselines are a separate block from reported survey numbers.")

    dec_header = ["dataset", "openness", "method", "FRR", "FAR", "closed acc"]
    dec_body = []
    for dataset, nominal in cells:
        subset = [r for r in diag if r["dataset"] == dataset and abs(float(r["nominal_openness"]) - nominal) < 1e-6]
        dec_body.append([
            dataset, f"{nominal:.2f}", "1_NN",
            _fmt_md(*_mean_std([r["decomposition_1nn"]["FRR"] for r in subset])),
            _fmt_md(*_mean_std([r["decomposition_1nn"]["FAR"] for r in subset])),
            _fmt_md(*_mean_std([r["decomposition_1nn"]["closed_acc"] for r in subset])),
        ])
    _write_pair(tables / "e1_error_decomposition", dec_header, dec_body, "Error decomposition at the 1_NN operating point.")

    e2_header = ["dataset", "openness", "paper", "1_NN", "radius q=0.85", "E2a sequential", "E2a joint", "best reimplemented baseline"]
    e2_body = []
    q_header = ["dataset", "partition", "openness", "selected_q", "applied"]
    q_body = []
    for dataset, nominal in cells:
        subset = [r for r in diag if r["dataset"] == dataset and abs(float(r["nominal_openness"]) - nominal) < 1e-6]
        paper_mean, paper_std = paper.get((dataset, nominal), (float("nan"), float("nan")))
        baseline_means = []
        for method in sorted({r["method"] for r in base}):
            brows = [r for r in base if r["dataset"] == dataset and abs(float(r["nominal_openness"]) - nominal) < 1e-6 and r["method"] == method]
            if brows:
                baseline_means.append((_mean_std([r["paper_f1_percent"] for r in brows])[0], method))
        best = max(baseline_means)[0] if baseline_means else float("nan")
        e2_body.append([
            dataset, f"{nominal:.2f}", _fmt_md(paper_mean, paper_std),
            _fmt_md(*_mean_std([r["actual_f1_1nn"] for r in subset])),
            _fmt_md(*_mean_std([r["radius_q85_f1"] for r in subset])),
            _fmt_md(*_mean_std([r["e2_sequential_f1"] for r in subset])),
            _fmt_md(*_mean_std([r["e2_joint_f1"] for r in subset])),
            "" if best != best else f"{best:.2f}",
        ])
        for row in subset:
            q_body.append([
                dataset, str(row["partition_id"]), f"{nominal:.2f}",
                "" if row["selected_q"] is None else f"{float(row['selected_q']):.3f}",
                "no" if row["rejection_off"] else "yes",
            ])
    _write_pair(tables / "e2_calibrated_f1", e2_header, e2_body, "E2 calibrated radius. 1_NN is the repository operating point. Radius columns use class-centroid scores. Baselines are reimplemented locally.")
    _write_pair(tables / "e2_selected_q", q_header, q_body, "Selected radius percentile q. Sequential and joint coincide because the GNG triple is a single candidate.")

    e3_header = ["dataset", "openness", "method", "actual", "oracle tau", "regret"]
    e3_body = []
    for dataset, nominal in cells:
        subset = [r for r in diag if r["dataset"] == dataset and abs(float(r["nominal_openness"]) - nominal) < 1e-6 and r.get("oracle_tau_1nn") is not None]
        if not subset:
            continue
        e3_body.append([
            dataset, f"{nominal:.2f}", "1_NN ORACLE",
            _fmt_md(*_mean_std([r["actual_f1_1nn"] for r in subset])),
            _fmt_md(*_mean_std([r["oracle_tau_1nn"] for r in subset])),
            _fmt_md(*_mean_std([r["regret_1nn"] for r in subset])),
        ])
        for method in sorted({r["method"] for r in base}):
            brows = [r for r in base if r["dataset"] == dataset and abs(float(r["nominal_openness"]) - nominal) < 1e-6 and r["method"] == method and r.get("regret") is not None]
            if not brows:
                continue
            e3_body.append([
                dataset, f"{nominal:.2f}", f"{method} ORACLE",
                _fmt_md(*_mean_std([r["paper_f1_percent"] for r in brows])),
                _fmt_md(*_mean_std([r["oracle_tau_f1"] for r in brows])),
                _fmt_md(*_mean_std([r["regret"] for r in brows])),
            ])
    _write_pair(tables / "e3_oracle_regret_ORACLE", e3_header, e3_body, "ORACLE threshold regret. The oracle maximises test micro-F1 using unknown labels.")

    cmp_header = ["dataset", "nominal_openness", "actual_openness", "method", "micro_f1", "AUROC", "FRR", "FAR", "n"]
    cmp_body = []
    for dataset, nominal in cells:
        subset = [r for r in diag if r["dataset"] == dataset and abs(float(r["nominal_openness"]) - nominal) < 1e-6]
        actual = f"{float(subset[0]['actual_openness_percent']):.2f}"
        cmp_body.append([
            dataset, f"{nominal:.2f}", actual, "1_NN",
            _fmt_md(*_mean_std([r["actual_f1_1nn"] for r in subset])),
            _fmt_md(*_mean_std([r["auroc_1nn"] for r in subset])),
            _fmt_md(*_mean_std([r["decomposition_1nn"]["FRR"] for r in subset])),
            _fmt_md(*_mean_std([r["decomposition_1nn"]["FAR"] for r in subset])),
            str(len(subset)),
        ])
        for method in sorted({r["method"] for r in base}):
            brows = [r for r in base if r["dataset"] == dataset and abs(float(r["nominal_openness"]) - nominal) < 1e-6 and r["method"] == method]
            if not brows:
                continue
            cmp_body.append([
                dataset, f"{nominal:.2f}", actual, method,
                _fmt_md(*_mean_std([r["paper_f1_percent"] for r in brows])),
                _fmt_md(*_mean_std([r["auroc"] for r in brows])),
                _fmt_md(*_mean_std([r["decomposition"]["FRR"] for r in brows])),
                _fmt_md(*_mean_std([r["decomposition"]["FAR"] for r in brows])),
                str(len(brows)),
            ])
    _write_pair(
        tables / "baseline_comparison",
        cmp_header,
        cmp_body,
        "Local comparison on the log partitions. 1_NN is the saved GNG model. Other rows are baselines trained on the same splits.",
    )
    _write_markdown_table(RESULTS_STATIC / "BASELINE_TABLE.md", cmp_header, cmp_body)

    _plot_oscr(diag, base)
    _plot_e2(diag)
    _plot_e3(diag)
    summary = _summary_md(diag, base, e1_body, e3_body)
    (RESULTS_STATIC / "SUMMARY.md").write_text(summary, encoding="utf-8")
    atomic_json(RESULTS_STATIC / "aggregate_summary.json", {"diagnostics": len(diag), "baselines": len(base)})
    return {"diagnostics": len(diag), "baselines": len(base)}


def _write_markdown_table(path: Path, header: list[str], body: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Baseline comparison",
        "",
        "Micro-F1 uses the paper definition. AUROC treats the unknown class as the positive class and is empty at openness 0.",
        "Nominal openness is the log key. Actual openness is computed from the class counts in that log.",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _summary_md(diag, base, e1_body, e3_body) -> str:
    regrets = [r["regret_1nn"] for r in diag if r.get("regret_1nn") is not None]
    frr = [r["decomposition_1nn"]["FRR"] for r in diag if float(r["actual_openness_percent"]) > 0]
    mean_regret, _ = _mean_std(regrets)
    mean_frr, _ = _mean_std(frr)
    lines = [
        "# Summary",
        "",
        "Phase 1 trained GNG + `1_NN` and was accepted with the published Table 1 gap left as a protocol difference.",
        "E1–E3 rescore those saved models. Radius scores use one centroid per class in the scaled training space.",
        "Sequential and joint q selection match, because the legacy GNG triple is a single candidate.",
        "Reimplemented baselines, when present, are separate from the survey numbers in `reference/table1_paper.csv`.",
        "",
        f"Mean threshold regret of `1_NN` (oracle tau minus the validation operating point): {mean_regret:.2f} points.",
        f"Mean false-reject rate of `1_NN` on open partitions: {mean_frr:.3f}.",
        "",
        "E1 and E3 tables are in `results/tables/`. Figures are in `results/figures/`.",
        "",
        f"Diagnostic jobs: {len(diag)}. Baseline jobs: {len(base)}.",
        "",
    ]
    if e3_body:
        worst = max(e3_body, key=lambda row: _regret_cell(row[-1]))
        lines.append(f"Largest listed oracle regret is {worst[-1]} for {' '.join(worst[:3])}.")
        lines.append("")
    if e1_body:
        lines.append("AUROC and micro-F1 need not rank methods the same way. Compare `e1_auroc_oscr` with `e2_calibrated_f1` before treating a high AUROC as a high F1.")
        lines.append("")
    return "\n".join(lines)


def _regret_cell(text: str) -> float:
    try:
        return float(text.split("±")[0].strip())
    except ValueError:
        return -1.0
