"""Pipeline runner (approach B: hand-wired stage order).

A *plan* is an explicit, ordered list of component ids (defined in
``train.py`` / ``infer.py``). The runner walks the plan and, for each component:

* expands the **fold** axis per the component's ``fold`` tag
  (``map`` -> one run per fold; ``reduce``/``none`` -> a single run);
* applies the **hybrid skip-or-produce** rule — if every declared output already
  exists for a key, the component is skipped;
* dispatches to the component's stub, recording progress events.

Per-component DOMAIN LOGIC is still stubbed (``NotImplementedError``). The runner
treats an unimplemented stub as a non-fatal ``stub`` outcome so the full wiring
can be exercised end-to-end before real code lands.
"""
from __future__ import annotations

from dataclasses import dataclass

from helios.data.catalog import ComponentSpec, get_component
from helios.pipeline.context import ComponentContext
from helios.pipeline.dispatch import get_component_fn
from helios.utils.partition import fold_values


@dataclass
class StageResult:
    component: str
    keys: dict
    status: str  # "done" | "skipped" | "stub" | "error"
    detail: str = ""


def _fold_coords(comp: ComponentSpec, n_folds: int) -> list[int | None]:
    if comp.fold == "map":
        return fold_values(n_folds)
    return [None]


def _outputs_exist(ctx: ComponentContext, comp: ComponentSpec, keys: dict) -> bool:
    produces: tuple[str, ...] = comp.produces or ()
    if not produces:
        return False
    try:
        return all(ctx.store.exists(pid, **keys) for pid in produces)
    except Exception:  # noqa: BLE001 - missing partition keys => can't prove existence
        return False


def run_component(ctx: ComponentContext, component_id: str, *, force: bool = False) -> list[StageResult]:
    comp = get_component(component_id)
    fn = get_component_fn(component_id)
    results: list[StageResult] = []

    for fold in _fold_coords(comp, ctx.config.cv.n_folds):
        keys = {} if fold is None else {"fold": fold}

        if not force and _outputs_exist(ctx, comp, keys):
            ctx.reporter.stage(component_id, "skipped", **keys)
            results.append(StageResult(component_id, keys, "skipped"))
            continue

        ctx.reporter.stage(component_id, "start", **keys)
        try:
            fn(ctx, keys)
        except NotImplementedError as exc:
            ctx.reporter.stage(component_id, "stub", **keys)
            results.append(StageResult(component_id, keys, "stub", str(exc)))
            continue
        except Exception as exc:  # noqa: BLE001 - surface but keep plan position
            ctx.reporter.stage(component_id, "error", error=str(exc), **keys)
            results.append(StageResult(component_id, keys, "error", str(exc)))
            raise
        ctx.reporter.stage(component_id, "done", **keys)
        results.append(StageResult(component_id, keys, "done"))

    return results


def run_plan(ctx: ComponentContext, plan: list[str], *, force: bool = False) -> list[StageResult]:
    ctx.reporter.update(plan=plan, n_stages=len(plan))
    all_results: list[StageResult] = []
    for component_id in plan:
        all_results.extend(run_component(ctx, component_id, force=force))
    ctx.reporter.update(status="complete")
    return all_results
