"""Benchmark fintfm on TabArena-Lite and place it against the cached leaderboard.

Coverage is the number to read first: fintfm's checkpoints accept at most 136 features, and
multiclass and regression each need a checkpoint whose head is wide enough (``max_classes >=
K`` classes, or ``>= K`` quantile bins), so part of any general tabular suite is out of scope
by construction. A score without its coverage fraction beside it would describe a different,
easier benchmark than the one everyone else ran.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.contexts import TabArenaContext


def eligible_datasets() -> list[str]:
    """TabArena datasets fintfm can actually run, within the checkpoint's caps.

    Computed from TabArena's own task metadata rather than hardcoded, so the coverage fraction
    reported alongside any score is derived from the suite as it currently stands.

    ``FINTFM_PROBLEM_TYPES`` narrows this to a comma-separated subset (e.g. ``binary``), which
    is how a score stays comparable with an earlier run made before another problem type was
    supported. It does **not** change the reported coverage denominator.
    """
    from tabarena.contexts import TabArenaContext

    tm = TabArenaContext().task_metadata
    # **Read the problem types from the model, never restate them here.** A duplicated list
    # drifts: on 2026-09-25 the adapter was narrowed to binary only (§121, multiclass and
    # regression measured as ranking last) and this function went on reporting 90% coverage
    # for a model that would refuse 19 of those datasets. Same failure shape as the mirrored
    # defaults `fintfm/CLAUDE.md` requires a drift guard for -- except here the single source
    # of truth already exists, so use it.
    from tabarena.models.fintfm.model import FinTFMModel

    kinds = tuple(FinTFMModel._supported_problem_types)
    # A table wider than the checkpoint runs when FINTFM_PARAMS names a column strategy the
    # adapter accepts for its problem type: "select" for any, "subsample" for classification.
    strategy = json.loads(os.environ.get("FINTFM_PARAMS") or "{}").get("max_features_strategy")
    wide_ok = (strategy == "select") | ((strategy == "subsample") & (tm["problem_type"] != "regression"))
    fits_width = (tm["n_features"] <= 136) | wide_ok
    ok = tm[
        tm["problem_type"].isin(kinds)
        & fits_width
        & ((tm["problem_type"] != "multiclass") | (tm["n_classes"].fillna(2) <= 10))
    ]
    excluded = {
        "problem type not declared by the model": (~tm["problem_type"].isin(kinds)).sum(),
        "over 136 features": (tm["problem_type"].isin(kinds) & ~fits_width).sum(),
        "over 10 classes": ((tm["problem_type"] == "multiclass") & (tm["n_classes"].fillna(2) > 10)).sum(),
    }
    print(f"coverage: {len(ok)} of {len(tm)} datasets ({len(ok) / len(tm):.0%})")
    for reason, n in excluded.items():
        if n:
            print(f"  excluded, {reason}: {n}")
    wanted_types = os.environ.get("FINTFM_PROBLEM_TYPES")
    if wanted_types:
        keep = [t.strip() for t in wanted_types.split(",") if t.strip()]
        ok = ok[ok["problem_type"].isin(keep)]
        print(f"  restricted to problem types {keep}: {len(ok)} datasets will run")
    names = sorted(ok["dataset_name"].tolist())
    # A subset is for smoke-testing a preprocessing change before paying for the full suite.
    # Any score printed from a subset is not comparable with the leaderboard mean, so the
    # runner says so rather than leaving the reader to notice.
    subset = os.environ.get("FINTFM_DATASETS")
    if subset:
        wanted = [d.strip() for d in subset.split(",") if d.strip()]
        missing = [d for d in wanted if d not in names]
        if missing:
            raise SystemExit(f"not eligible or not found: {missing}")
        print(f"SUBSET RUN: {len(wanted)} of {len(names)} -- not a leaderboard mean")
        return wanted
    return names


if __name__ == "__main__":
    # No checkpoint argument. The wrapper downloads the published binary weights pinned to a
    # commit, so a submitted run is produced from the same artifact a maintainer would get
    # rather than from a local file only this machine has. FINTFM_CHECKPOINT still overrides,
    # which is how the unpublished multiclass and regression checkpoints are measured.
    full = "--full" in sys.argv

    here = Path(__file__).parent
    # Results are cached per (config, dataset, fold), and a preprocessing change does not
    # alter the config name -- so re-running in place silently returns the previous numbers
    # instead of measuring the change. A distinct run name is what forces a real refit, and
    # it keeps the earlier run's raw results intact for comparison.
    run_name = os.environ.get("FINTFM_RUN_NAME") or ("fintfm_lite_full" if full else "fintfm_lite_smoke")
    results_dir = str(here / "experiments" / run_name)
    eval_dir = here / "eval" / run_name

    experiments = TabArenaV0pt1ExperimentBundle(
        # default config only; the search space is empty by design. FINTFM_MODEL selects a
        # sibling entrant (e.g. FinTFM-LinProbe) for a local measurement run.
        models=[(os.environ.get("FINTFM_MODEL", "FinTFM"), 0)],
    ).build_experiments()

    context = TabArenaContext()
    context.build_and_run_jobs(
        experiments,
        expname=results_dir,
        subset="lite",
        build_kwargs={"dataset_names": eligible_datasets()},
        new_result_prefix="[New] ",
        debug_mode=True,
    )
    # A per-dataset queue calls this once per dataset to give each fit a fresh process (MPS
    # memory grows across fits in one process); the leaderboard comparison then runs once, in
    # the queue's final all-dataset call, instead of once per dataset.
    if os.environ.get("FINTFM_SKIP_COMPARE") == "1":
        sys.exit(0)

    leaderboard = context.compare(output_dir=eval_dir)
    website = context.leaderboard_to_website_format(leaderboard=leaderboard)
    print("\n=== TabArena leaderboard ===")
    print(website.to_markdown(index=False))
