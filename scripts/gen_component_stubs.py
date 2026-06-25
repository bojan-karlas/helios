#!/usr/bin/env python3
"""Generate one typed component stub per spec component.

Reads ``configs/components.yaml`` and writes a stub function (named after the
component id) into the module file declared by each component's ``module``
path. Each stub raises ``NotImplementedError`` and carries a docstring with the
spec contract (consumes/produces/fold/partitions) so students can fill in the
domain logic against an exact, machine-checked interface.

Stubs are 1:1 with spec components. Re-running regenerates ONLY between the
AUTOGEN markers, preserving any imports/helpers added around them.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
COMPONENTS = ROOT / "configs" / "components.yaml"

BEGIN = "# --- AUTOGEN:component-stubs (do not edit between markers) ---"
END = "# --- /AUTOGEN:component-stubs ---"

HEADER = '''"""HELIOS component stubs ({module}).

Auto-generated from configs/components.yaml. Fill in the domain logic inside
each function; keep the signature ``(ctx, keys)`` and honor the documented
consumes/produces contract. Regenerate the skeleton with
``python scripts/gen_component_stubs.py``.
"""
from __future__ import annotations

from typing import Any

from helios.pipeline.context import ComponentContext
'''


def module_path(module: str) -> Path:
    return SRC / Path(module.replace(".", "/") + ".py")


def _fmt_ids(ids: list) -> str:
    """Render an id list, wrapping onto indented lines to stay within line length."""
    if not ids:
        return "[]"
    text = ", ".join(ids)
    if len(text) <= 80:
        return text
    return "\n      ".join(ids)


def stub_source(cid: str, comp: dict) -> str:
    consumes = comp.get("consumes") or []
    produces = comp.get("produces") or []
    partitions = comp.get("partitions") or []
    fold = comp.get("fold", "none")
    title = comp.get("title", cid)
    mode = comp.get("mode") or []
    lines = [
        f"def {cid}(ctx: ComponentContext, keys: dict[str, Any]) -> None:",
        f'    """{title}.',
        "",
        f"    fold: {fold} | mode: {', '.join(mode) or '-'}",
        f"    partitions: {partitions or '[]'}",
        f"    consumes: {_fmt_ids(consumes)}",
        f"    produces: {_fmt_ids(produces)}",
        "",
        "    ``keys`` holds this invocation's partition coordinates.",
        "    Read inputs via ``ctx.store.read(<id>, **keys)`` and write outputs via",
        "    ``ctx.store.write(<id>, obj, **keys)``.",
        '    """',
        f'    raise NotImplementedError("{cid} is not implemented yet")',
        "",
    ]
    return "\n".join(lines)


def render_module(module: str, comps: list[tuple[str, dict]]) -> str:
    body = "\n\n".join(stub_source(cid, c) for cid, c in comps)
    return (
        HEADER.format(module=module)
        + "\n\n"
        + BEGIN
        + "\n\n"
        + body
        + "\n\n"
        + END
        + "\n"
    )


def replace_autogen(existing: str, module: str, comps: list[tuple[str, dict]]) -> str:
    body = "\n\n".join(stub_source(cid, c) for cid, c in comps)
    block = BEGIN + "\n\n" + body + "\n\n" + END
    pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.DOTALL)
    if pattern.search(existing):
        return pattern.sub(block, existing)
    return existing.rstrip() + "\n\n" + block + "\n"


def ensure_packages(path: Path) -> None:
    rel = path.relative_to(SRC)
    pkg = SRC
    for part in rel.parts[:-1]:
        pkg = pkg / part
        pkg.mkdir(parents=True, exist_ok=True)
        init = pkg / "__init__.py"
        if not init.exists():
            init.write_text('"""HELIOS package."""\n')


def main() -> None:
    data = yaml.safe_load(COMPONENTS.read_text())["components"]
    by_module: dict[str, list[tuple[str, dict]]] = {}
    for cid, comp in data.items():
        by_module.setdefault(comp["module"], []).append((cid, comp))

    for module, comps in sorted(by_module.items()):
        path = module_path(module)
        ensure_packages(path)
        if path.exists():
            content = replace_autogen(path.read_text(), module, comps)
        else:
            content = render_module(module, comps)
        path.write_text(content)
        print(f"  {module} -> {path.relative_to(ROOT)} ({len(comps)} stub(s))")

    print(f"Generated stubs for {len(data)} components across {len(by_module)} modules.")


if __name__ == "__main__":
    main()
