"""
Phase E — Goal predicate identification from terminal-state valuations.

Given a Phase D ASP result (a solved STRIPS model), Phase E:

1. Re-runs clingo on the same .lp to capture the FULL model (Phase D only
   saves a 60-atom sample, which is insufficient for valuation extraction).
2. Parses all `val((P,(O1,O2)), S)` atoms — fluent predicate P holds for
   object-pair (O1,O2) at state S.
3. Identifies which states are terminal (WIN / GAME_OVER) via the Phase B
   graph and the Phase D node_remap.
4. Finds the MINIMAL DISCRIMINATING CONJUNCTION of fluent atoms that are
   simultaneously true at the terminal state(s) but do NOT co-occur at any
   non-terminal state.

   Formally: for fluent atom set F* ⊆ {val(K, term) | K in model},
   the goal condition is the smallest F* such that:
       {s : ∀ K∈F*, val(K,s) = 1} = {terminal states}

5. Records the initial-state valuation (true fluents at the start node,
   the earliest-first_seen_at node in the subgraph).
6. Writes output to neurosym/lp/<game>.phase_e.json.

The output is the input to Phase F (PDDL serialisation + pyperplan dispatch).

Usage:
    cd D:/arc3
    python neurosym/phase_e_goal.py --game vc33
    python neurosym/phase_e_goal.py --game ft09  # no terminal → reports best guess
    python neurosym/phase_e_goal.py --all
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import clingo  # type: ignore

ARC3_ROOT = Path(__file__).resolve().parents[1]
GRAPHS_DIR = ARC3_ROOT / "neurosym" / "graphs"
LP_OUT_DIR = ARC3_ROOT / "neurosym" / "lp"
KR21_DIR = ARC3_ROOT / "external" / "learner-strips" / "asp" / "clingo" / "kr21"


# ---------------------------------------------------------------------------
# Full model extraction (re-runs clingo, captures all symbols)
# ---------------------------------------------------------------------------


def _run_clingo_full(
    lp_src: str,
    num_actions: int,
    num_objects: int = 2,
    max_predicates: int = 5,
    max_static: int = 2,
    max_precs: int = 6,
    time_limit_s: int = 60,
) -> list[clingo.Symbol] | None:
    """Re-run clingo and return the complete first-model symbol list, or None on
    TIMEOUT / UNSAT."""
    ctl = clingo.Control([
        "-c", f"num_objects={num_objects}",
        "-c", f"max_predicates={max_predicates}",
        "-c", f"max_static={max_static}",
        "-c", f"max_precs={max_precs}",
        "-c", "opt_synthesis=1",
        "1",
    ])
    for fname in ("base2.lp", "constraints_blai.lp"):
        ctl.load(str(KR21_DIR / fname))

    inject = f"num_actions({num_actions}).\n"
    tmp = LP_OUT_DIR / f"_phase_e_{int(time.time()*1000)}.lp"
    tmp.write_text(inject + lp_src, encoding="utf-8")
    try:
        ctl.load(str(tmp))
        ctl.ground([("base", [])])
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass

    found: list[list[clingo.Symbol]] = []

    def on_model(m: clingo.Model) -> None:
        found.append(list(m.symbols(shown=True) or m.symbols(atoms=True)))

    timed_out = False

    def _cancel() -> None:
        nonlocal timed_out
        timed_out = True
        handle.cancel()

    handle = ctl.solve(on_model=on_model, async_=True)
    timer = threading.Timer(time_limit_s, _cancel)
    timer.start()
    try:
        handle.wait()
    finally:
        timer.cancel()

    if timed_out or not found:
        return None
    return found[0]


# ---------------------------------------------------------------------------
# Val-atom parsing
# ---------------------------------------------------------------------------


def extract_valuations(
    symbols: list[clingo.Symbol],
) -> tuple[dict[str, set[int]], list[str]]:
    """Parse all val atoms from the first model.

    Returns:
        fluent_true_at: dict mapping fluent-key string -> set of state ints
                        where the fluent is true.
        static_true:    list of static fluent-key strings (globally true, no
                        state index).

    The kr21 encoding #shows:
        val(K, S)  for fluent predicates that are true at state S
        val(K)     for static predicates that are true (no state index)
    where K = (predicate_id, (obj1, obj2)).
    """
    fluent_true_at: dict[str, set[int]] = {}
    static_true: list[str] = []

    for sym in symbols:
        if sym.name != "val":
            continue
        args = sym.arguments
        if len(args) == 2:
            # Fluent: val(K, S)
            k = str(args[0])
            try:
                s = args[1].number
            except RuntimeError:
                continue
            fluent_true_at.setdefault(k, set()).add(s)
        elif len(args) == 1:
            # Static: val(K)
            static_true.append(str(args[0]))

    return fluent_true_at, static_true


# ---------------------------------------------------------------------------
# Goal predicate computation
# ---------------------------------------------------------------------------


def find_minimal_goal(
    fluent_true_at: dict[str, set[int]],
    terminal_states: set[int],
    all_states: set[int],
) -> dict[str, Any]:
    """Find the minimal conjunction of fluent atoms that jointly discriminate
    terminal states from all non-terminal states.

    Returns a dict with:
        goal_conjunction    : list of fluent-key strings (the goal condition)
        conjunction_size    : int (number of fluents in the conjunction)
        discriminating      : bool (True if the conjunction uniquely identifies
                                    only the terminal states)
        false_positive_states: list of non-terminal state ints also satisfying
                                the conjunction (empty if discriminating=True)
        coverage            : dict{state: list[fluent-key]} for states satisfying
                                the conjunction
    """
    non_terminal = all_states - terminal_states

    # Fluents that are true in AT LEAST ONE terminal state.
    terminal_candidates: list[str] = [
        k for k, states in fluent_true_at.items()
        if terminal_states & states
    ]

    # Sort candidates by how discriminating they are: fewest non-terminal
    # states where they're true come first (makes search faster).
    terminal_candidates.sort(
        key=lambda k: len(non_terminal & fluent_true_at[k])
    )

    # BFS over conjunctions (smallest first) to find the minimal discriminating set.
    # We try size 1, 2, 3, ... up to min(len(candidates), 6).
    max_size = min(len(terminal_candidates), 6)
    best_conjunction: list[str] = []
    best_fps: set[int] = non_terminal  # false positives (non-terminal states satisfying conjunction)

    for size in range(1, max_size + 1):
        found = False
        for combo in combinations(terminal_candidates, size):
            # States satisfying ALL fluents in combo.
            satisfying = all_states.copy()
            for k in combo:
                satisfying &= fluent_true_at[k]
            fps = satisfying - terminal_states
            if len(fps) < len(best_fps):
                best_fps = fps
                best_conjunction = list(combo)
            if not fps:
                found = True
                best_conjunction = list(combo)
                best_fps = set()
                break
        if found:
            break

    # Coverage: for each state satisfying the best conjunction, list the true fluents.
    if best_conjunction:
        satisfying_states = all_states.copy()
        for k in best_conjunction:
            satisfying_states &= fluent_true_at[k]
    else:
        satisfying_states = set()

    return {
        "goal_conjunction": best_conjunction,
        "conjunction_size": len(best_conjunction),
        "discriminating": len(best_fps) == 0,
        "false_positive_states": sorted(best_fps),
        "satisfying_states": sorted(satisfying_states),
        "terminal_states": sorted(terminal_states),
    }


# ---------------------------------------------------------------------------
# Initial-state extraction
# ---------------------------------------------------------------------------


def extract_initial_state(
    fluent_true_at: dict[str, set[int]],
    static_true: list[str],
    start_state_int: int,
) -> dict[str, Any]:
    """Return the initial-state description: true fluents at the start node."""
    true_fluents = [k for k, states in fluent_true_at.items() if start_state_int in states]
    return {
        "start_state_int": start_state_int,
        "true_fluents": sorted(true_fluents),
        "static_true": sorted(static_true),
    }


# ---------------------------------------------------------------------------
# Per-game Phase E runner
# ---------------------------------------------------------------------------


def run_for_game(game: str, time_limit_s: int = 60) -> int:
    asp_result_path = LP_OUT_DIR / f"{game}.asp_result.json"
    lp_path = LP_OUT_DIR / f"{game}.lp"
    graph_path = GRAPHS_DIR / f"{game}.graph.json"

    for p in (asp_result_path, lp_path, graph_path):
        if not p.exists():
            print(f"[{game}] Missing file: {p} — run Phase D first.")
            return 2

    with open(asp_result_path) as f:
        asp = json.load(f)
    meta = asp["meta"]
    result = asp["result"]

    if result["sat"] != "SAT":
        print(f"[{game}] Phase D result is {result['sat']} — Phase E requires SAT.")
        return 1

    with open(graph_path) as f:
        graph = json.load(f)

    # Identify terminal nodes from the graph.
    node_remap: dict[str, int] = meta["node_remap"]
    graph_nodes = graph["nodes"]
    terminal_shorts = {
        n["short"] for n in graph_nodes
        if n.get("state") in {"WIN", "GAME_OVER"}
    }
    terminal_ints = {
        node_remap[s] for s in terminal_shorts
        if s in node_remap
    }
    # Start node (earliest first_seen_at that's in the subgraph).
    subgraph_shorts = set(node_remap.keys())
    subgraph_nodes = [n for n in graph_nodes if n["short"] in subgraph_shorts]
    start_node = min(subgraph_nodes, key=lambda n: n.get("first_seen_at", 0))
    start_int = node_remap[start_node["short"]]
    all_state_ints = set(node_remap.values())

    print(f"[{game}] Subgraph: {len(all_state_ints)} states, "
          f"terminal={sorted(terminal_ints)}, start={start_int}")

    if not terminal_ints:
        print(f"[{game}] WARNING: no terminal states in subgraph. "
              "Goal inference is not possible — will report initial-state only.")

    # Re-run clingo to get the full model.
    print(f"[{game}] Re-running clingo for full model (num_actions={meta['num_actions_effective']}) ...")
    lp_src = lp_path.read_text(encoding="utf-8")
    t0 = time.time()
    symbols = _run_clingo_full(
        lp_src,
        num_actions=meta["num_actions_effective"],
        num_objects=result["num_objects"],
        max_predicates=result["max_predicates"],
        time_limit_s=time_limit_s,
    )
    elapsed = time.time() - t0

    if symbols is None:
        print(f"[{game}] Clingo returned no model (TIMEOUT or UNSAT).")
        return 1

    print(f"[{game}] Got {len(symbols)} symbols in {elapsed:.2f}s")

    # Parse val atoms.
    fluent_true_at, static_true = extract_valuations(symbols)
    print(f"[{game}] Fluent atoms (P,T) with at least one true state: {len(fluent_true_at)}")
    print(f"[{game}] Static atoms: {len(static_true)}")

    # Report per-state val counts.
    for s in sorted(all_state_ints):
        true_here = [k for k, states in fluent_true_at.items() if s in states]
        tag = " [TERMINAL]" if s in terminal_ints else (" [START]" if s == start_int else "")
        print(f"  state {s:2d}{tag}: {sorted(true_here)}")

    # Goal condition.
    if terminal_ints:
        goal_info = find_minimal_goal(fluent_true_at, terminal_ints, all_state_ints)
        print(f"\n[{game}] Goal condition (size={goal_info['conjunction_size']}, "
              f"discriminating={goal_info['discriminating']}):")
        for k in goal_info["goal_conjunction"]:
            print(f"  val{k} = 1")
        if not goal_info["discriminating"]:
            print(f"  WARNING: also satisfied by non-terminal states: "
                  f"{goal_info['false_positive_states']}")
    else:
        goal_info = {
            "goal_conjunction": [],
            "conjunction_size": 0,
            "discriminating": False,
            "false_positive_states": [],
            "satisfying_states": [],
            "terminal_states": [],
            "note": "no terminal states in subgraph",
        }

    # Initial state.
    init_info = extract_initial_state(fluent_true_at, static_true, start_int)
    print(f"\n[{game}] Initial state (node {start_int}):")
    print(f"  true fluents: {init_info['true_fluents']}")
    print(f"  static true:  {init_info['static_true']}")

    # Write output.
    out = {
        "game": game,
        "phase_e": {
            "num_states": len(all_state_ints),
            "terminal_states": sorted(terminal_ints),
            "start_state": start_int,
            "total_fluent_keys": len(fluent_true_at),
            "goal": goal_info,
            "initial_state": init_info,
            "label_remap": meta["label_remap"],
            "node_remap": meta["node_remap"],
        },
    }
    out_path = LP_OUT_DIR / f"{game}.phase_e.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[{game}] Phase E result written to {out_path}")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", type=str, help="game id (e.g. vc33, ft09)")
    parser.add_argument("--all", action="store_true",
                        help="run Phase E for all games with a Phase D ASP result")
    parser.add_argument("--time-limit", type=int, default=60)
    args = parser.parse_args(argv)

    if args.all:
        rc = 0
        for asp_path in sorted(LP_OUT_DIR.glob("*.asp_result.json")):
            g = asp_path.stem.replace(".asp_result", "")
            rc |= run_for_game(g, time_limit_s=args.time_limit)
        return rc

    if not args.game:
        parser.error("either --game or --all required")
    return run_for_game(args.game, time_limit_s=args.time_limit)


if __name__ == "__main__":
    sys.exit(main())
