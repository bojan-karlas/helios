"""Interactive HTML report of A-MIL training traces (see :mod:`helios.components.mil_trace`).

:func:`render_trace_report` finds every fold trace under the given paths (a run's logs
directory, one fold's trace directory, or any directory above them), overlays all runs
and folds in the same charts, and writes one self-contained HTML file with plotly.js
embedded, so it opens offline in any browser. Per-step series are summarized in fixed
bins (median and maximum); spike steps are kept as individual points.
"""
from __future__ import annotations

import html
import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from helios.components.mil_trace import BLOCKS

STEP_BIN = 50
#: Per-step lines keep at most this many bins; longer runs get wider bins.
MAX_POINTS = 300
#: House color scheme. Fold lines cycle through the main, then dark, then light shades.
COLORS = {
    "red": "#DF362A",
    "blue": "#2BBFD9",
    "yellow": "#FAC802",
    "green": "#8AB365",
    "pink": "#C670D2",
    "grey": "#788591",
    "dark red": "#7C2D27",
    "dark blue": "#006D8F",
    "dark yellow": "#AE7E1E",
    "dark green": "#008F6B",
    "dark pink": "#815FA5",
    "dark grey": "#434A51",
    "light red": "#E79892",
    "light blue": "#96D3DE",
    "light yellow": "#F5E294",
    "light green": "#C0E2A1",
    "light pink": "#DDAAE4",
    "light grey": "#8D96A0",
}
#: Validation class colors: negative (0) and positive (1) labels.
CLASS_COLORS = {0: COLORS["blue"], 1: COLORS["red"]}
REPORT_NAME = "trace-report.html"
_FOLD = re.compile(r"^fold=(.+)$")


@dataclass
class FoldTrace:
    """One fold's trace tables plus where they came from."""

    path: Path
    run_id: str
    fold: str
    run_dir: Path | None
    steps: pd.DataFrame
    epochs: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame | None

    @property
    def label(self) -> str:
        return f"{self.run_id} fold={self.fold}"

    @property
    def selected_epoch(self) -> int | None:
        """The kept epoch: the last that improved on the best so far, else the last one (no validation)."""
        if self.epochs.empty:
            return None
        improved = self.epochs.loc[self.epochs["improved"].astype(bool), "epoch"]
        return int(improved.max() if len(improved) else self.epochs["epoch"].max())

    @property
    def averaged_epochs(self) -> list[int]:
        """Epochs folded into the saved weight average (empty when the run selected an epoch)."""
        if "averaged" not in self.epochs:
            return []
        return [int(x) for x in self.epochs.loc[self.epochs["averaged"].astype(bool), "epoch"]]


def find_traces(paths: list[Path]) -> list[FoldTrace]:
    """Every fold trace (a directory holding ``epochs.parquet``) at or below ``paths``, in a stable order."""
    dirs: set[Path] = set()
    for path in paths:
        path = Path(path).resolve()
        if (path / "epochs.parquet").is_file():
            dirs.add(path)
        dirs.update(p.parent for p in path.rglob("epochs.parquet"))
    traces = [t for t in (_load(d) for d in sorted(dirs)) if t is not None]
    return sorted(traces, key=lambda t: (t.run_id, _fold_key(t.fold)))


def render_trace_report(
    paths: list[Path], output: Path, *, refresh_seconds: int | None = None, title: str = "MIL training traces",
) -> Path:
    """Write the HTML report for every trace under ``paths`` to ``output`` (atomically); returns ``output``.

    ``refresh_seconds`` makes an open browser tab reload itself, for watching a run live.
    """
    traces = find_traces(paths)
    colors = _palette(len(traces))
    sections = [
        ("Runs", _runs_table(traces)),
        ("Learning curves", _learning_curves(traces, colors)),
        ("Training stability", _stability(traces, colors)),
        ("Model internals", _internals(traces, colors)),
        ("Validation predictions", _predictions(traces, colors)),
        ("Events", _events(traces)),
    ]
    from plotly.offline import get_plotlyjs

    refresh = f'<meta http-equiv="refresh" content="{int(refresh_seconds)}">' if refresh_seconds else ""
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    nav = " | ".join(f'<a href="#s{i}">{html.escape(name)}</a>' for i, (name, _) in enumerate(sections))
    body = "\n".join(
        f'<h2 id="s{i}">{html.escape(name)}</h2>\n{content}' for i, (name, content) in enumerate(sections)
    )
    if not traces:
        body = "<p>No traces found yet.</p>"
    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">{refresh}<title>{html.escape(title)}</title>
<script>{get_plotlyjs()}</script>
<style>{_CSS}</style></head>
<body><h1>{html.escape(title)}</h1>
<p class="meta">Generated {generated}; {len(traces)} fold trace(s) from {html.escape(", ".join(map(str, paths)))}.
{"Reloads every " + str(int(refresh_seconds)) + " s." if refresh_seconds else ""}</p>
<p class="nav">{nav}</p>
{body}
<script>{_SORT_JS}</script>
</body></html>
"""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    try:
        tmp.write_text(page, encoding="utf-8")
        os.replace(tmp, output)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return output


# -- loading ------------------------------------------------------------------
def _load(path: Path) -> FoldTrace | None:
    try:
        epochs = pd.read_parquet(path / "epochs.parquet")
        steps = pd.read_parquet(path / "steps.parquet")
        val = pd.read_parquet(path / "val_predictions.parquet")
    except (FileNotFoundError, OSError, ValueError):
        return None
    test_path = path / "test_predictions.parquet"
    test = pd.read_parquet(test_path) if test_path.is_file() else None
    match = _FOLD.match(path.name)
    fold = match.group(1) if match else "none"
    traces_root = path.parent if match else path
    run_dir = traces_root.parent.parent if traces_root.parent.name == "traces" else None
    run_id = run_dir.name if run_dir is not None else str(path)
    return FoldTrace(path, run_id, fold, run_dir, steps, epochs, val, test)


def _fold_key(fold: str) -> tuple[int, str]:
    return (int(fold), "") if fold.isdigit() else (1 << 30, fold)


def _code_version(run_dir: Path | None) -> str:
    try:
        code = json.loads((run_dir / "status.json").read_text()).get("code") if run_dir else None
    except (OSError, ValueError):
        code = None
    if not code:
        return "unknown"
    return f"{str(code.get('commit', ''))[:10]}{' (with changes)' if code.get('dirty') else ''}"


def _settings_lines(traces: list[FoldTrace]) -> list[str]:
    lines: list[str] = []
    for run_dir in sorted({t.run_dir for t in traces if t.run_dir is not None}):
        log = run_dir / "run.log"
        if log.is_file():
            for line in log.read_text(errors="replace").splitlines():
                if "MIL device=" in line:
                    lines.append(f"{run_dir.name}: {line.split('MIL device=', 1)[-1]}")
    return lines


# -- sections -----------------------------------------------------------------
def _runs_table(traces: list[FoldTrace]) -> str:
    rows = []
    for t in traces:
        e = t.epochs
        sel = t.selected_epoch
        at = e[e["epoch"] == sel].iloc[0] if sel is not None else None
        averaged = t.averaged_epochs
        prefix = "test_average" if averaged else "test"
        rows.append({
            "run": t.run_id, "fold": t.fold, "code": _code_version(t.run_dir), "epochs": len(e),
            "saved model": f"average of epochs {averaged[0]}-{averaged[-1]}" if averaged else f"epoch {sel}",
            "val AUPRC (saved)": "-" if averaged else _num(at, "val_auprc"),
            "test AUPRC (saved)": _num(at, f"{prefix}_auprc"), "test AUROC (saved)": _num(at, f"{prefix}_auroc"),
            "loss spikes": int(e["loss_spikes"].sum()), "grad spikes": int(e["grad_spikes"].sum()),
            "collapse epochs": ", ".join(str(x) for x in e.loc[e["training_collapse"], "epoch"]) or "-",
            "updated": datetime.fromtimestamp((t.path / "epochs.parquet").stat().st_mtime).strftime("%m-%d %H:%M"),
        })
    settings = _settings_lines(traces)
    details = (
        "<details><summary>Training settings (from run.log)</summary><pre>"
        + html.escape("\n".join(settings)) + "</pre></details>"
    ) if settings else ""
    return _table(pd.DataFrame(rows)) + details


def _learning_curves(traces: list[FoldTrace], colors: list[str]) -> str:
    panels = [
        ("Training loss (mean over bags)", "train_loss"),
        ("Validation AUPRC, all bags (selects the epoch)", "val_auprc"),
        ("Validation AUPRC, original bags", "val_original_auprc"),
        ("Validation AUROC, all bags", "val_auroc"),
        ("Validation loss (class-weighted)", "val_loss"),
    ]
    if any(t.test is not None for t in traces):
        panels += [("Test-fold AUPRC (analysis only)", "test_auprc"), ("Test-fold AUROC (analysis only)", "test_auroc")]
        panels += [
            ("Test-fold AUPRC, running weight average (analysis only)", "test_average_auprc"),
            ("Test-fold AUROC, running weight average (analysis only)", "test_average_auroc"),
        ]
    note = (
        "<p class='note'>Stars mark each fold's selected epoch (the last epoch for weight-averaged runs); "
        "crosses mark training-collapse epochs; hollow circles are epochs that cannot be kept: before "
        "<code>min_epochs</code>, or before weight averaging starts.</p>"
    )
    return note + _epoch_figure(traces, colors, panels)


def _stability(traces: list[FoldTrace], colors: list[str]) -> str:
    note = (
        f"<p class='note'>Per-step values in bins of at least {STEP_BIN} steps "
        f"(at most {MAX_POINTS} bins per line): "
        "solid lines are bin medians, dotted lines bin maxima. Points are spike steps (hover for the slide). "
        "The x axis is in epochs.</p>"
    )
    first = _step_figure(
        traces, colors, [("Loss", "loss", False), ("Total gradient size", "grad_total", True)], spikes=True,
    )
    updates = _step_figure(
        traces, colors, [(f"Update size / weight size: {b}", f"update_ratio_{b}", True) for b in BLOCKS],
    )
    grads = _step_figure(traces, colors, [(f"Gradient size: {b}", f"grad_{b}", True) for b in BLOCKS])
    return note + first + updates + grads


def _internals(traces: list[FoldTrace], colors: list[str]) -> str:
    branches = sorted({int(c[3:].split("_")[0]) for t in traces for c in t.steps if re.match(r"att\d+_", c)})
    attention = []
    for k in branches:
        attention += [
            (f"Branch {k} attention entropy (1 = uniform)", f"att{k}_entropy", False),
            (f"Branch {k} max attention x tiles (at most 7.4)", f"att{k}_max_x_n", False),
        ]
    attention.append(("Similarity between branch attentions", "branch_similarity", False))
    gate = [
        ("Quality gate mean", "gate_mean", False), ("Quality gate minimum", "gate_min", False),
        ("Fraction of tiles gated below 0.05", "gate_frac_below_0.05", False),
        ("Fraction of tiles gated above 0.95", "gate_frac_above_0.95", False),
        ("Feature enhancer residual / input size", "enhancer_residual_ratio", True),
    ]
    dead = [(f"Inactive-unit fraction: {c.removeprefix('dead_frac_')}", c, False)
            for c in ("dead_frac_branch_fusion", "dead_frac_classifier_512", "dead_frac_classifier_256",
                      "dead_frac_classifier_128")]
    outputs = [("Pre-sigmoid score", "logit", False), ("Fraction of bags with clamped probability", "clamped", False)]
    weights = [(f"Weight size: {b}", f"weight_{b}") for b in BLOCKS]
    return (
        _step_figure(traces, colors, attention) + _step_figure(traces, colors, gate)
        + _step_figure(traces, colors, dead) + _step_figure(traces, colors, outputs)
        + _epoch_figure(traces, colors, weights)
    )


def _predictions(traces: list[FoldTrace], colors: list[str]) -> str:
    lines = _epoch_figure(traces, colors, [
        ("Mean validation probability, positive bags", "val_mean_prob_pos"),
        ("Mean validation probability, negative bags", "val_mean_prob_neg"),
        ("Spread (std) of validation probabilities", "val_prob_std"),
    ])
    return lines + _distribution_figure(traces)


def _events(traces: list[FoldTrace]) -> str:
    frames = []
    for t in traces:
        s = t.steps
        if s.empty:
            continue
        flagged = s[s["loss_spike"] | s["grad_spike"] | s["nonfinite"]]
        if len(flagged):
            frames.append(flagged.assign(run=t.run_id, fold=t.fold))
    if not frames:
        return "<p>No loss spikes, gradient spikes, or non-finite steps.</p>"
    events = pd.concat(frames, ignore_index=True)
    events["kind"] = np.select(
        [events["nonfinite"], events["loss_spike"] & events["grad_spike"], events["loss_spike"]],
        ["non-finite", "loss + gradient", "loss"], "gradient",
    )
    cols = ["run", "fold", "epoch", "step", "kind", "image_id", "bag", "label", "n_tiles", "loss", "grad_total",
            "probability"]
    top = (
        events.groupby(["image_id", "label"]).agg(spikes=("step", "size"), runs=("run", "nunique"))
        .reset_index().sort_values("spikes", ascending=False).head(30)
    )
    return (
        f"<p>{len(events)} flagged step(s). Click a column header to sort.</p>"
        + _table(events[cols]) + "<h3>Slides with the most spikes</h3>" + _table(top)
    )


# -- figures ------------------------------------------------------------------
def _epoch_figure(traces: list[FoldTrace], colors: list[str], panels: list[tuple[str, str]]) -> str:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    panels = [(name, col) for name, col in panels if any(col in t.epochs for t in traces)]
    if not panels:
        return ""
    fig = make_subplots(rows=len(panels), cols=1, shared_xaxes=True, subplot_titles=[p[0] for p in panels],
                        vertical_spacing=0.25 / len(panels))
    for i, t in enumerate(traces):
        e = t.epochs
        sel = t.selected_epoch
        for row, (_, col) in enumerate(panels, 1):
            if col not in e:
                continue
            fig.add_trace(go.Scatter(
                x=e["epoch"], y=e[col], mode="lines+markers", name=t.label, legendgroup=t.label,
                showlegend=row == 1, line=dict(color=colors[i], width=1.5),
                marker=dict(size=5, symbol=["circle" if ok else "circle-open" for ok in e["eligible"]]),
                hovertemplate=f"{html.escape(t.label)}<br>epoch %{{x}}<br>%{{y:.4g}}<extra></extra>",
            ), row=row, col=1)
            marks = e[(e["epoch"] == sel) | e["training_collapse"]]
            if len(marks):
                fig.add_trace(go.Scatter(
                    x=marks["epoch"], y=marks[col], mode="markers", legendgroup=t.label, showlegend=False,
                    marker=dict(color=colors[i], size=12, line=dict(width=1, color=COLORS["dark grey"]),
                                symbol=["star" if ep == sel else "x" for ep in marks["epoch"]]),
                    hoverinfo="skip",
                ), row=row, col=1)
    fig.update_xaxes(title_text="epoch", row=len(panels), col=1)
    return _html(fig, 230 * len(panels) + 80)


def _step_figure(
    traces: list[FoldTrace], colors: list[str], panels: list[tuple[str, str, bool]], *, spikes: bool = False,
) -> str:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    panels = [p for p in panels if any(p[1] in t.steps for t in traces)]
    if not panels:
        return ""
    fig = make_subplots(rows=len(panels), cols=1, shared_xaxes=True, subplot_titles=[p[0] for p in panels],
                        vertical_spacing=0.25 / len(panels))
    for i, t in enumerate(traces):
        s = _with_epoch_position(t.steps)
        if s.empty:
            continue
        width = max(STEP_BIN, -(-len(s) // MAX_POINTS))
        key = s["_seq"] // width
        x = s.groupby(key)["_x"].mean().astype("float32")
        for row, (_, col, log_y) in enumerate(panels, 1):
            if col not in s:
                continue
            values = s[col].astype(float)
            if log_y:
                values = values.where(values > 0)
            grouped = values.groupby(key)
            fig.add_trace(go.Scatter(
                x=x, y=grouped.median().astype("float32"), mode="lines", name=t.label, legendgroup=t.label,
                showlegend=row == 1,
                line=dict(color=colors[i], width=1.5),
                hovertemplate=f"{html.escape(t.label)}<br>epoch %{{x:.2f}}<br>median %{{y:.4g}}<extra></extra>",
            ), row=row, col=1)
            fig.add_trace(go.Scatter(
                x=x, y=grouped.max().astype("float32"), mode="lines", legendgroup=t.label, showlegend=False,
                line=dict(color=colors[i], width=1, dash="dot"),
                hovertemplate=f"{html.escape(t.label)}<br>epoch %{{x:.2f}}<br>max %{{y:.4g}}<extra></extra>",
            ), row=row, col=1)
            if spikes:
                flagged = s[s["loss_spike"] | s["grad_spike"]]
                if len(flagged):
                    fig.add_trace(go.Scatter(
                        x=flagged["_x"], y=flagged[col], mode="markers", legendgroup=t.label, showlegend=False,
                        marker=dict(
                            color=colors[i], size=7, symbol="diamond", line=dict(width=1, color=COLORS["dark grey"]),
                        ),
                        customdata=np.stack([flagged["image_id"], flagged["bag"], flagged["step"]], axis=1),
                        hovertemplate=(f"{html.escape(t.label)}<br>%{{customdata[0]}} (%{{customdata[1]}})<br>"
                                       "step %{customdata[2]}<br>%{y:.4g}<extra>spike</extra>"),
                    ), row=row, col=1)
            if log_y:
                fig.update_yaxes(type="log", tickformat=".2g", dtick="D2", row=row, col=1)
    fig.update_xaxes(title_text="epoch", row=len(panels), col=1)
    return _html(fig, 210 * len(panels) + 80)


def _distribution_figure(traces: list[FoldTrace]) -> str:
    """Box plots of validation probabilities by class per epoch, one fold at a time (dropdown)."""
    import plotly.graph_objects as go

    fig = go.Figure()
    owners: list[int] = []
    for i, t in enumerate(traces):
        if "probability" not in t.val:
            continue
        v = t.val.dropna(subset=["probability"])
        for label, color in CLASS_COLORS.items():
            q = v[v["label"] == label].groupby("epoch")["probability"].quantile([0.0, 0.25, 0.5, 0.75, 1.0])
            if q.empty:
                continue
            q = q.unstack()
            fig.add_trace(go.Box(
                x=q.index, lowerfence=q[0.0], q1=q[0.25], median=q[0.5], q3=q[0.75], upperfence=q[1.0],
                name=f"{'positive' if label else 'negative'} ({label})", marker_color=color, visible=i == 0,
                offsetgroup=str(label),
            ))
            owners.append(i)
    if not owners:
        return ""
    buttons = [
        dict(label=t.label, method="update", args=[{"visible": [o == i for o in owners]}])
        for i, t in enumerate(traces)
    ]
    fig.update_layout(
        title="Validation probability by class (choose a fold)", boxmode="group",
        updatemenus=[dict(buttons=buttons, x=1.0, xanchor="right", y=1.15, yanchor="top")],
        xaxis_title="epoch", yaxis_title="probability",
    )
    return _html(fig, 420)


# -- helpers ------------------------------------------------------------------
def _with_epoch_position(steps: pd.DataFrame) -> pd.DataFrame:
    """Add ``_x`` (epoch number plus fraction of the epoch done) and a per-epoch step sequence for binning."""
    if steps.empty:
        return steps
    s = steps.sort_values("step").reset_index(drop=True)
    within = s.groupby("epoch").cumcount()
    size = s.groupby("epoch")["step"].transform("size")
    s["_x"] = s["epoch"] - 1 + (within + 0.5) / size
    s["_seq"] = s["epoch"] * 10**7 + within
    s["clamped"] = s["clamped"].astype(float)
    return s


def _palette(n: int) -> list[str]:
    base = list(COLORS.values())
    return [base[i % len(base)] for i in range(n)]


def _num(row: pd.Series | None, col: str) -> str:
    if row is None or col not in row or pd.isna(row[col]):
        return "-"
    return f"{float(row[col]):.3f}"


def _html(fig, height: int) -> str:
    fig.update_layout(height=height, margin=dict(l=60, r=20, t=60, b=40), hovermode="closest",
                      legend=dict(font=dict(size=10)), template="plotly_white")
    return fig.to_html(full_html=False, include_plotlyjs=False, default_height=f"{height}px")


def _table(frame: pd.DataFrame) -> str:
    return frame.to_html(index=False, classes="sortable", border=0, float_format=lambda x: f"{x:.4g}")


_CSS = """
body { font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; margin: 24px; color: #434A51; }
h2 { border-bottom: 2px solid #2BBFD9; padding-bottom: 4px; margin-top: 40px; }
.meta, .note { color: #788591; font-size: 0.9em; }
.nav a { margin: 0 4px; color: #006D8F; }
table.sortable { border-collapse: collapse; font-size: 0.85em; margin: 8px 0; }
table.sortable th, table.sortable td { padding: 3px 8px; border-bottom: 1px solid #eee; text-align: right; }
table.sortable th { cursor: pointer; background: #f6f6f6; position: sticky; top: 0; }
pre { font-size: 0.8em; white-space: pre-wrap; }
"""

_SORT_JS = """
document.querySelectorAll("table.sortable th").forEach((th, col) => th.addEventListener("click", () => {
  const body = th.closest("table").tBodies[0];
  const asc = th.dataset.dir !== "asc"; th.dataset.dir = asc ? "asc" : "desc";
  const key = r => { const t = r.cells[col].textContent.trim(); const n = parseFloat(t);
    return isNaN(n) || !/^-?[0-9.]/.test(t) ? t : n; };
  [...body.rows].sort((a, b) => { const x = key(a), y = key(b);
    return (x > y ? 1 : x < y ? -1 : 0) * (asc ? 1 : -1); }).forEach(r => body.appendChild(r));
}));
"""
