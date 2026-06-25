"""Pure, typed pipeline components — the implementer's workspace.

A *component* is a pure function of its inputs. It must NOT touch the artifact
store, the path resolver, or the run config: those are the responsibility of the
owning *stage* (``helios.stages``), which loads inputs, calls the component once
per work unit, and writes the outputs.

Each component therefore has an explicit, fully typed signature describing
exactly what goes in and what comes out, plus a ``GRAIN`` note stating the unit
of work it expects (one image, a batch, an aggregate, ...).
"""
