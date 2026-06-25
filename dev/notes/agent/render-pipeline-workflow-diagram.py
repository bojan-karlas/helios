#!/usr/bin/env python3
"""Render the HELIOS pipeline DAG from the pipeline specification.

The dependency graph is DERIVED, never hand-maintained: an edge A -> B exists
whenever (A.produces ∩ B.consumes) != {}. This script walks the
producers -> consumers relationship and emits either a Mermaid or Graphviz
diagram.

Two layouts:
  - bipartite (default): two distinguishable node types -- operators (functional
    components) and data artifacts. Trainable operators (model-fitting steps) are
    filled #C670D2; other operators #2BBFD9; data artifacts #FAC802.
    All nodes are square with 2pt black borders. Edges are plain
    (artifact -> operator for consumes, operator -> artifact for produces).
  - flow: operators only; each edge is labeled with the artifact that connects
    two operators.

Examples
--------
    # Bipartite Mermaid (default), written next to the spec
    python render-pipeline-workflow-diagram.py

    # Operator-only flow view as Graphviz DOT
    python render-pipeline-workflow-diagram.py --layout flow --format dot

    # Explicit paths, print to stdout
    python render-pipeline-workflow-diagram.py \
        --spec pipeline-spec.yaml --output - --format dot

    # Write DOT and rasterize to PNG (or svg/pdf) via Graphviz
    python render-pipeline-workflow-diagram.py --render png
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("PyYAML is required: pip install pyyaml")


# Node fill colors (aesthetic spec).
DATA_COLOR = "#FAC802"        # data artifacts
TRAINABLE_COLOR = "#C670D2"   # trainable component nodes (model training steps)
COMPONENT_COLOR = "#2BBFD9"   # all other (non-trainable) component nodes
BORDER_COLOR = "#000000"
BORDER_WIDTH = "2"            # points


def load_spec(path: Path) -> dict:
    with path.open() as fh:
        spec = yaml.safe_load(fh)
    if not isinstance(spec, dict) or "components" not in spec:
        sys.exit(f"{path}: not a valid pipeline spec (missing 'components').")
    return spec


def is_trainable(component: dict) -> bool:
    """A component is trainable if it produces a model.

    Override via the optional `trainable:` boolean field in the spec;
    otherwise defaults to `kind == 'training'`.
    """
    if "trainable" in component:
        return bool(component["trainable"])
    return component.get("kind") == "training"


def component_color(component: dict) -> str:
    return TRAINABLE_COLOR if is_trainable(component) else COMPONENT_COLOR


def build_graph(spec: dict):
    """Return (components, artifacts, edges, missing).

    edges: set of (src_component_id, dst_component_id, artifact_id)
    missing: artifacts consumed but never produced (true inputs/pretrained).
    """
    components = {c["id"]: c for c in spec.get("components", [])}
    artifacts = {a["id"]: a for a in spec.get("artifacts", [])}

    producers: dict[str, list[str]] = {}
    for cid, c in components.items():
        for art in c.get("produces", []) or []:
            producers.setdefault(art, []).append(cid)

    edges = set()
    consumed = set()
    for cid, c in components.items():
        for art in c.get("consumes", []) or []:
            consumed.add(art)
            for src in producers.get(art, []):
                edges.add((src, cid, art))

    missing = sorted(consumed - set(producers))
    return components, artifacts, edges, missing


def render_mermaid(spec: dict, layout: str, show_artifacts: bool) -> str:
    components, artifacts, edges, missing = build_graph(spec)
    lines = ["%% Auto-generated from the HELIOS pipeline spec. Do not edit by hand.",
             "graph LR"]

    phases: dict[str, list[str]] = {}
    for cid, c in components.items():
        phases.setdefault(c.get("phase", "other"), []).append(cid)

    train_ids = [cid for cid, c in components.items() if is_trainable(c)]
    plain_ids = [cid for cid, c in components.items() if not is_trainable(c)]

    def class_defs() -> list[str]:
        return [
            f"  classDef dataNode fill:{DATA_COLOR},stroke:{BORDER_COLOR},"
            f"stroke-width:{BORDER_WIDTH}px,color:#000;",
            f"  classDef compNode fill:{COMPONENT_COLOR},stroke:{BORDER_COLOR},"
            f"stroke-width:{BORDER_WIDTH}px,color:#000;",
            f"  classDef trainNode fill:{TRAINABLE_COLOR},stroke:{BORDER_COLOR},"
            f"stroke-width:{BORDER_WIDTH}px,color:#000;",
        ]

    if layout == "bipartite":
        # Two node types: operators (components) and data artifacts.
        used_arts = set()
        for cid, c in components.items():
            used_arts.update(c.get("consumes", []) or [])
            used_arts.update(c.get("produces", []) or [])

        for phase, members in phases.items():
            lines.append(f"  subgraph {phase}")
            for cid in members:
                title = components[cid].get("title", cid)
                lines.append(f'    {cid}["{title}"]')
            lines.append("  end")

        for art in sorted(used_arts):
            title = artifacts.get(art, {}).get("title", art)
            lines.append(f'  {art}["{title}"]')

        # consumes: artifact -> operator ; produces: operator -> artifact
        for cid, c in components.items():
            for art in c.get("consumes", []) or []:
                lines.append(f"  {art} --> {cid}")
            for art in c.get("produces", []) or []:
                lines.append(f"  {cid} --> {art}")

        lines += class_defs()
        if used_arts:
            lines.append(f"  class {','.join(sorted(used_arts))} dataNode;")
        if plain_ids:
            lines.append(f"  class {','.join(plain_ids)} compNode;")
        if train_ids:
            lines.append(f"  class {','.join(train_ids)} trainNode;")
        return "\n".join(lines) + "\n"

    # --- layout == "flow": operators only, labeled edges ---
    for phase, members in phases.items():
        lines.append(f"  subgraph {phase}")
        for cid in members:
            title = components[cid].get("title", cid)
            lines.append(f'    {cid}["{title}"]')
        lines.append("  end")

    extra_data = []
    if show_artifacts:
        for art in missing:
            title = artifacts.get(art, {}).get("title", art)
            lines.append(f'  {art}["{title}"]')
            extra_data.append(art)

    for src, dst, art in sorted(edges):
        lines.append(f"  {src} -->|{art}| {dst}")

    if show_artifacts:
        for cid, c in components.items():
            for art in c.get("consumes", []) or []:
                if art in missing:
                    lines.append(f"  {art} --> {cid}")

    lines += class_defs()
    if extra_data:
        lines.append(f"  class {','.join(extra_data)} dataNode;")
    if plain_ids:
        lines.append(f"  class {','.join(plain_ids)} compNode;")
    if train_ids:
        lines.append(f"  class {','.join(train_ids)} trainNode;")

    return "\n".join(lines) + "\n"


def render_dot(spec: dict, layout: str, show_artifacts: bool) -> str:
    components, artifacts, edges, missing = build_graph(spec)
    lines = ["// Auto-generated from the HELIOS pipeline spec. Do not edit by hand.",
             "digraph helios {",
             "  rankdir=LR;",
             f'  node [shape=box, style=filled, fontname="Helvetica", '
             f'color="{BORDER_COLOR}", penwidth={BORDER_WIDTH}];',
             '  edge [fontsize=9, color="#555555"];']

    def op_node(cid: str) -> str:
        title = components[cid].get("title", cid).replace('"', "'")
        fill = component_color(components[cid])
        return f'  "{cid}" [label="{title}", fillcolor="{fill}"];'

    def data_node(art: str) -> str:
        title = artifacts.get(art, {}).get("title", art).replace('"', "'")
        return f'  "{art}" [label="{title}", fillcolor="{DATA_COLOR}"];'

    if layout == "bipartite":
        used_arts = set()
        for cid, c in components.items():
            used_arts.update(c.get("consumes", []) or [])
            used_arts.update(c.get("produces", []) or [])

        for cid in components:
            lines.append(op_node(cid))
        for art in sorted(used_arts):
            lines.append(data_node(art))
        for cid, c in components.items():
            for art in c.get("consumes", []) or []:
                lines.append(f'  "{art}" -> "{cid}";')
            for art in c.get("produces", []) or []:
                lines.append(f'  "{cid}" -> "{art}";')
        lines.append("}")
        return "\n".join(lines) + "\n"

    # --- layout == "flow": operators only, labeled edges ---
    for cid in components:
        lines.append(op_node(cid))

    if show_artifacts:
        for art in missing:
            lines.append(data_node(art))
        for cid, c in components.items():
            for art in c.get("consumes", []) or []:
                if art in missing:
                    lines.append(f'  "{art}" -> "{cid}";')

    for src, dst, art in sorted(edges):
        lines.append(f'  "{src}" -> "{dst}" [label="{art}"];')

    lines.append("}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render the HELIOS pipeline DAG from the spec.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--spec", type=Path, default=Path(__file__).resolve().parent / "pipeline-spec.yaml",
        help="Path to the pipeline spec YAML.",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output path. Defaults to pipeline-workflow.<ext> next to the spec. "
             "Use '-' for stdout.",
    )
    parser.add_argument(
        "--format", "-f", choices=["mermaid", "dot"], default="mermaid",
        help="Diagram format.",
    )
    parser.add_argument(
        "--layout", "-l", choices=["flow", "bipartite"], default="bipartite",
        help="flow: operators only, edges labeled with artifact. "
             "bipartite: artifacts and operators as distinct node types, plain edges.",
    )
    parser.add_argument(
        "--show-artifacts", action="store_true",
        help="(flow layout only) also render true input artifacts as nodes.",
    )
    parser.add_argument(
        "--render", "-r", choices=["svg", "png", "pdf"], default=None,
        help="Also rasterize/vectorize the diagram to an image next to the output "
             "(requires Graphviz 'dot'; forces --format dot).",
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="DPI for --render png.",
    )
    args = parser.parse_args(argv)

    if args.render:
        args.format = "dot"

    if args.output is None:
        ext = "mmd" if args.format == "mermaid" else "dot"
        args.output = str(args.spec.resolve().parent / f"pipeline-workflow.{ext}")

    spec = load_spec(args.spec)
    _, _, _, missing = build_graph(spec)

    render = render_mermaid if args.format == "mermaid" else render_dot
    text = render(spec, args.layout, args.show_artifacts)

    if args.output == "-":
        sys.stdout.write(text)
    else:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        ncomp = len(spec.get("components", []))
        print(
            f"Wrote {args.format} diagram for {ncomp} components to {out}",
            file=sys.stderr,
        )
        if missing:
            print(f"Root inputs (no producer): {', '.join(missing)}", file=sys.stderr)

        if args.render:
            image = _rasterize(out, args.render, args.dpi)
            print(f"Rendered {args.render.upper()} to {image}", file=sys.stderr)

    return 0


def _rasterize(dot_path: Path, fmt: str, dpi: int) -> Path:
    """Run Graphviz `dot` to convert a .dot file into svg/png/pdf."""
    if shutil.which("dot") is None:
        sys.exit("Graphviz 'dot' not found on PATH; install graphviz to use --render.")
    image = dot_path.with_suffix(f".{fmt}")
    cmd = ["dot", f"-T{fmt}"]
    if fmt == "png":
        cmd.append(f"-Gdpi={dpi}")
    cmd += [str(dot_path), "-o", str(image)]
    subprocess.run(cmd, check=True)
    return image


if __name__ == "__main__":
    raise SystemExit(main())
