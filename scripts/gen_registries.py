#!/usr/bin/env python3
"""Generate the shipped runtime registries from the design spec.

The pipeline spec (dev/notes/agent/pipeline-spec.yaml) is the single source of
truth for the artifact + component model. The package, however, must not depend
on non-shipped dev/ notes at runtime, so we derive two trimmed registries into
configs/ that the package loads:

    configs/artifacts.yaml   id -> {title, key, format, path}
    configs/components.yaml  id -> {title, kind, phase, mode, fold, module,
                                    consumes, produces, partitions, params}

Run this whenever the spec changes:  python scripts/gen_registries.py
"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "dev" / "notes" / "agent" / "pipeline-spec.yaml"
ARTIFACTS_OUT = ROOT / "configs" / "artifacts.yaml"
COMPONENTS_OUT = ROOT / "configs" / "components.yaml"

HEADER = (
    "# GENERATED from dev/notes/agent/pipeline-spec.yaml — do not hand-edit.\n"
    "# Regenerate via scripts/gen_registries.py\n"
)


def main() -> None:
    spec = yaml.safe_load(SPEC.read_text())

    artifacts = {
        a["id"]: {
            "title": a.get("title"),
            "key": a.get("key", []) or [],
            "format": a.get("format", "bundle"),
            "path": a.get("path"),
        }
        for a in spec["artifacts"]
    }
    components = {
        c["id"]: {
            "title": c.get("title"),
            "kind": c.get("kind"),
            "phase": c.get("phase"),
            "mode": c.get("mode", []),
            "fold": c.get("fold", "none"),
            "module": c.get("module"),
            "consumes": c.get("consumes", []),
            "produces": c.get("produces", []),
            "partitions": c.get("partitions", []),
            "params": c.get("params", {}),
        }
        for c in spec["components"]
    }

    with ARTIFACTS_OUT.open("w") as f:
        f.write(HEADER)
        yaml.safe_dump({"artifacts": artifacts}, f, sort_keys=False, width=120)
    with COMPONENTS_OUT.open("w") as f:
        f.write(HEADER)
        yaml.safe_dump({"components": components}, f, sort_keys=False, width=120)

    print(f"wrote {ARTIFACTS_OUT.relative_to(ROOT)} ({len(artifacts)} artifacts)")
    print(f"wrote {COMPONENTS_OUT.relative_to(ROOT)} ({len(components)} components)")


if __name__ == "__main__":
    main()
