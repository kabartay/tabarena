r"""Score several fintfm TabArena-Lite runs as one entry over every dataset they cover together.

``run_fintfm_lite.py`` runs one problem type at a time, because each needs its own checkpoint
(binary, multiclass, regression), so a full-coverage entry ends up spread over several run
directories. This script reads their cached ``results.pkl`` files, registers them as a single
new method, and places it on the Lite leaderboard. It fits nothing.

Two runs normally carry the same method name (``FinTFM-LinProbe_c1_default_BAG_L1``) and
merge as they are. Runs of different entrants, for example the zero-shot ``FinTFM`` on
multiclass beside ``FinTFM-LinProbe`` on binary, merge only with ``--framework``, which
relabels every result to the name given. That builds an assembled entry, so the name should
say so, and the provenance file lists which run supplied which datasets.

Refuses a dataset scored by two runs, a split other than Lite's (repeat 0, fold 0), and a
dataset count different from ``--expect`` when that is set.

Usage, from the repository root::

    .venv/bin/python -P merge_fintfm_lite.py --name fintfm_lite_all51 \\
        fintfm_linprobe_squash_part_090 fintfm_linprobe_squash_part_mc_1110 \\
        fintfm_linprobe_ridge_reg_1110 --expect 51
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent


def load_run(run: str) -> list[dict]:
    """Every cached result of one run, read from ``experiments/<run>/data``.

    Args:
        run: The run name (``FINTFM_RUN_NAME`` of the Lite run that produced it).

    Returns:
        The raw per-split result dicts.

    Raises:
        SystemExit: The run directory holds no results.
    """
    files = sorted((HERE / "experiments" / run / "data").glob("*/*/*/results.pkl"))
    if not files:
        raise SystemExit(f"{run}: no results under experiments/{run}/data")
    out = []
    for f in files:
        with gzip.open(f) as fh:
            out.append(pickle.load(fh))
    return out


def relabel(result: dict, framework: str) -> dict:
    """Return ``result`` under the method name ``framework``, the original kept in the metadata.

    Args:
        result: One raw result dict.
        framework: The method name the merged entry carries.

    Returns:
        A shallow copy with ``framework`` replaced, the per-method prediction dicts in
        ``simulation_artifacts`` re-keyed to it, its model type and name prefix set to
        it, and the originals recorded under ``method_metadata["merged_from"]``,
        ``["merged_from_model_type"]`` and ``["merged_from_name_prefix"]``.
    """
    old = result["framework"]
    out = dict(result)
    meta = dict(out.get("method_metadata") or {})
    meta["merged_from"] = old
    # One method needs one model type; runs of different entrants carry different ones.
    meta["merged_from_model_type"] = meta.get("model_type")
    meta["model_type"] = framework
    meta["ag_key"] = framework
    meta["merged_from_name_prefix"] = meta.get("name_prefix")
    meta["name_prefix"] = framework
    out["method_metadata"] = meta
    sim = dict(out.get("simulation_artifacts") or {})
    for key in ("pred_proba_dict_val", "pred_proba_dict_test"):
        if key in sim:
            sim[key] = {(framework if k == old else k): v for k, v in sim[key].items()}
    out["simulation_artifacts"] = sim
    out["framework"] = framework
    return out


def main() -> None:
    """Merge the runs named on the command line, check coverage, and print the leaderboard."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("runs", nargs="+", help="run names under experiments/")
    p.add_argument("--name", required=True, help="eval/<name> receives the leaderboard")
    p.add_argument("--framework", help="relabel every result to this method name (assembled entry)")
    p.add_argument("--expect", type=int, help="refuse unless exactly this many datasets are covered")
    args = p.parse_args()

    results, source = [], {}
    for run in args.runs:
        for r in load_run(run):
            meta = r["task_metadata"]
            if (meta.get("repeat", 0), meta.get("fold", 0)) != (0, 0):
                raise SystemExit(
                    f"{run}: {meta['name']} has split repeat {meta.get('repeat')} fold {meta.get('fold')}, not Lite's 0/0"
                )
            if meta["name"] in source:
                raise SystemExit(f"{meta['name']} is scored by both {source[meta['name']]} and {run}")
            source[meta["name"]] = run
            results.append(r)
    frameworks = Counter(r["framework"] for r in results)
    if len(frameworks) > 1 and not args.framework:
        raise SystemExit(f"runs carry different methods {dict(frameworks)}; pass --framework to assemble them")
    if args.framework:
        results = [relabel(r, args.framework) for r in results]

    by_type = Counter(r["problem_type"] for r in results)
    print(f"merged {len(results)} datasets from {len(args.runs)} runs: {dict(by_type)}")
    if args.expect is not None and len(results) != args.expect:
        raise SystemExit(f"expected {args.expect} datasets, have {len(results)}")

    from tabarena.contexts import TabArenaContext

    context = TabArenaContext()
    context.register(results, new_result_prefix="[New] ")
    eval_dir = HERE / "eval" / args.name
    leaderboard = context.compare(output_dir=eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "merged_from.json").write_text(
        json.dumps(
            {
                "runs": args.runs,
                "framework": args.framework,
                "datasets": source,
                "by_problem_type": dict(by_type),
                "methods": dict(frameworks),
            },
            indent=2,
        )
        + "\n"
    )
    website = context.leaderboard_to_website_format(leaderboard=leaderboard)
    print("\n=== TabArena leaderboard ===")
    print(website.to_markdown(index=False))


if __name__ == "__main__":
    main()
