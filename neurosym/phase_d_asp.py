"""
Phase D — Bonet/Geffner KR21 ASP encoding wrapper.

Reads a Phase B graph JSON (`neurosym/graphs/<game>.graph.json`),
converts it to a Bonet/Geffner-style .lp graph file, and invokes the
KR21 ASP encoding shipped at:

    D:/arc3/external/learner-strips/asp/clingo/kr21/

via the `clingo` Python module. Reports whether the solver finds a
non-trivial first-order STRIPS model and prints the discovered
predicates / action schemas.

This is the very first proof-of-life check for whether the symbolic
side of the architecture has any signal on the random-baseline data we
already have.

Usage:
    cd D:/arc3
    python neurosym/phase_d_asp.py --selftest
    python neurosym/phase_d_asp.py --game ft09 [--num-objects 2] [--num-actions 4]

Honest caveat
-------------
The graphs we have right now were produced by the official ARC-AGI-3
recorder, which DOES NOT log the chosen action. So every edge is labeled
`a0` and the solver will at best discover a single-action model. This
script exists so that the moment we re-record with `record_actions_agent.py`
and have real edge labels, the symbolic pipeline lights up immediately.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import clingo  # type: ignore

ARC3_ROOT = Path(__file__).resolve().parents[1]
GRAPHS_DIR = ARC3_ROOT / "neurosym" / "graphs"
LP_OUT_DIR = ARC3_ROOT / "neurosym" / "lp"
LP_OUT_DIR.mkdir(parents=True, exist_ok=True)
KR21_DIR = ARC3_ROOT / "external" / "learner-strips" / "asp" / "clingo" / "kr21"


# ---------------------------------------------------------------------------
# LP serialisation
# ---------------------------------------------------------------------------


def _terminal_neighborhood(
    nodes: list[dict],
    all_edges: list[dict],
    max_nodes: int,
) -> tuple[list[dict], list[dict]]:
    """Return the BFS neighborhood of terminal nodes (WIN/GAME_OVER) backward
    through the graph, limited to `max_nodes` nodes.

    This extracts the most structurally informative slice for Phase D:
    the transitions that actually lead to terminal states have the strongest
    signal for what actions accomplish the goal.

    Returns (subset_nodes, subset_edges) — both filtered to the neighborhood.
    """
    node_by_short: dict[str, dict] = {n["short"]: n for n in nodes}
    terminal_shorts = {
        n["short"] for n in nodes
        if n.get("state") in {"WIN", "GAME_OVER"}
    }
    # Identify the start node (earliest first_seen_at).
    start_node = min(nodes, key=lambda n: n.get("first_seen_at", 0))
    start_short = start_node["short"]

    if not terminal_shorts:
        # No terminals: flag as unverifiable and return earliest-explored N nodes.
        # Always include the start node.
        subset = nodes[:max_nodes]
        short_set = {n["short"] for n in subset}
        short_set.add(start_short)
        if start_short not in {n["short"] for n in subset}:
            subset = [start_node] + subset[: max_nodes - 1]
        edges_sub = [
            e for e in all_edges
            if e["src"] in short_set and e["dst"] in short_set
        ]
        return subset, edges_sub

    # Build a reverse adjacency for non-self-loop edges.
    in_edges: dict[str, list[dict]] = {n["short"]: [] for n in nodes}
    for e in all_edges:
        if e["src"] != e["dst"]:
            in_edges.setdefault(e["dst"], []).append(e)

    # BFS backward from terminals.  Reserve slots for start + bridge path.
    max_interior = max_nodes - 1
    visited: set[str] = set(terminal_shorts)
    frontier = list(terminal_shorts)
    while frontier and len(visited) < max_interior:
        nxt = []
        for t in frontier:
            for e in in_edges[t]:
                if e["src"] not in visited:
                    visited.add(e["src"])
                    nxt.append(e["src"])
                    if len(visited) >= max_interior:
                        break
            if len(visited) >= max_interior:
                break
        frontier = nxt

    # Track whether the backward BFS already reached the start node.
    start_was_reached = start_short in visited

    # Always include the start node so the induced model can be tested
    # from the actual reset state.
    visited.add(start_short)

    # If the backward BFS did not reach the start node, the start is isolated:
    # it has no edges to the terminal neighborhood in the subset.  Fix this by
    # running a forward BFS from start through the unvisited part of the graph
    # until we connect to a node already in `visited`, then adding all bridge
    # nodes.  This ensures the ASP model has a connected path start → terminals
    # rather than an orphaned initial state with no transitions.
    if not start_was_reached:
        # Build forward adjacency (non-self-loop edges only).
        out_edges: dict[str, list[str]] = {}
        for e in all_edges:
            if e["src"] != e["dst"]:
                out_edges.setdefault(e["src"], []).append(e["dst"])

        fwd_frontier = [start_short]
        fwd_seen: set[str] = {start_short}
        fwd_parent: dict[str, str | None] = {start_short: None}
        bridge_found = False

        while fwd_frontier and not bridge_found:
            nxt = []
            for node in fwd_frontier:
                for dst in out_edges.get(node, []):
                    if dst in visited and dst != start_short:
                        # Found a bridge: trace the path back to start and
                        # add all intermediate nodes to visited.
                        cur: str | None = node
                        while cur is not None and cur not in visited:
                            visited.add(cur)
                            cur = fwd_parent.get(cur)
                        bridge_found = True
                        break
                    if dst not in fwd_seen:
                        fwd_seen.add(dst)
                        fwd_parent[dst] = node
                        nxt.append(dst)
                if bridge_found:
                    break
            fwd_frontier = nxt
        # If no bridge path exists (graph is truly disconnected between start
        # and terminals), start remains isolated — logged by graph_to_lp caller
        # via start_in_subset check.  This is correct: we cannot plan if the
        # recording never recorded a path from start toward any terminal.

    subset_nodes = [n for n in nodes if n["short"] in visited]
    subset_edges = [
        e for e in all_edges
        if e["src"] in visited and e["dst"] in visited
    ]
    return subset_nodes, subset_edges


def graph_to_lp(
    graph_path: Path,
    max_nodes: int | None = None,
    legal_only: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Convert a Phase B graph JSON to Bonet/Geffner-style LP source.

    Self-loop edges (src == dst) are dropped because the BG `eff_used`
    constraint requires every action to have at least one *visible* effect.

    If `legal_only=True` (default) and the graph has legality tags, only
    edges where `legal=True` are included. Edges from old recordings (no
    legality tag) are included unconditionally.

    If `max_nodes` is set and the graph exceeds it, the graph is trimmed to
    the BFS backward neighborhood of terminal (WIN/GAME_OVER) nodes, always
    preserving the start node (earliest first_seen_at).

    Returns the LP string and a metadata dict (id remap, action remap).
    """
    g = json.loads(graph_path.read_text(encoding="utf-8"))
    nodes = g["nodes"]
    all_edges = g["edges"]

    # Capture full counts before any trimming (fixes metadata bug where
    # trimmed counts were stored as "total").
    full_node_count = len(nodes)
    full_edge_count = len(all_edges)
    has_legal_tags = any(e.get("legal") is not None for e in all_edges)

    # Filter to legal-only edges if the recording carries legality tags.
    if legal_only and has_legal_tags:
        all_edges = [e for e in all_edges if e.get("legal") is not False]
    illegal_dropped = full_edge_count - len(all_edges) if has_legal_tags else 0

    # Optionally restrict to the terminal neighborhood.
    trimmed = False
    start_in_subset = True
    if max_nodes is not None and len(nodes) > max_nodes:
        nodes_before = {n["short"] for n in nodes}
        nodes, all_edges = _terminal_neighborhood(nodes, all_edges, max_nodes)
        trimmed = True
        subset_shorts = {n["short"] for n in nodes}
        # Verify start node is present.
        start_short = min(g["nodes"], key=lambda n: n.get("first_seen_at", 0))["short"]
        start_in_subset = start_short in subset_shorts

    # Split into real transitions (src != dst) and self-loops.
    trans_edges = [e for e in all_edges if e["src"] != e["dst"]]
    self_edges  = [e for e in all_edges if e["src"] == e["dst"]]

    # Only include action labels that appear in at least one real transition.
    effective_labels: set[str] = {e["label"] for e in trans_edges}
    noop_labels: set[str] = {e["label"] for e in self_edges} - effective_labels

    # Remap node IDs to dense 0..N-1 ints.
    node_to_id: dict[str, int] = {n["short"]: i for i, n in enumerate(nodes)}

    # Remap effective action labels to dense 1..K ints.  Action 0 is
    # reserved in the BG encoding for "no label assigned".
    action_set: list[str] = sorted(effective_labels)
    label_to_id: dict[str, int] = {lbl: i + 1 for i, lbl in enumerate(action_set)}

    lines: list[str] = []
    lines.append(f"% Generated from {graph_path.name}"
                 + (" (terminal neighborhood)" if trimmed else ""))
    lines.append(
        f"% {len(nodes)} nodes; {len(trans_edges)} real transitions "
        f"({len(self_edges)} self-loops dropped); {len(action_set)} effective labels"
    )
    if noop_labels:
        lines.append(f"% no-op labels (self-loop only, pruned): {sorted(noop_labels)}")
    lines.append(f"numedges({len(trans_edges)}).")
    for node in nodes:
        lines.append(f"node({node_to_id[node['short']]}).")
    for label, idx in label_to_id.items():
        lines.append(f'labelname({idx},"{label}").')
    for e in trans_edges:
        s, t = node_to_id[e["src"]], node_to_id[e["dst"]]
        a = label_to_id[e["label"]]
        lines.append(f"edge(({s},{t})).")
        lines.append(f"tlabel(({s},{t}),{a}).")

    meta = {
        "graph_file": str(graph_path),
        # Full-graph counts (before any trimming or legality filtering)
        "num_nodes_total": full_node_count,
        "num_edges_total": full_edge_count,
        "has_legal_tags": has_legal_tags,
        "illegal_edges_dropped": illegal_dropped,
        # Subgraph counts passed to clingo
        "num_nodes": len(nodes),
        "trimmed_to_neighborhood": trimmed,
        "start_node_in_subset": start_in_subset if trimmed else True,
        "num_self_loops": len(self_edges),
        "num_transitions": len(trans_edges),
        "num_actions_effective": len(action_set),
        "noop_labels": sorted(noop_labels),
        "node_remap": {n["short"]: node_to_id[n["short"]] for n in nodes},
        "label_remap": label_to_id,
    }
    return "\n".join(lines) + "\n", meta


# ---------------------------------------------------------------------------
# Clingo wrapper
# ---------------------------------------------------------------------------


def _kr21_files() -> list[Path]:
    """Return the list of kr21 .lp files we want to load."""
    files = [
        KR21_DIR / "base2.lp",
        KR21_DIR / "constraints_blai.lp",
    ]
    for f in files:
        if not f.exists():
            raise FileNotFoundError(f"missing kr21 file: {f}")
    return files


def run_asp(
    graph_lp_source: str,
    num_objects: int,
    num_actions: int,
    max_predicates: int = 5,
    max_static: int = 2,
    max_precs: int = 6,
    time_limit_s: int = 60,
    models_to_find: int = 1,
) -> dict[str, Any]:
    """Run the KR21 encoding on a given graph LP source.

    Returns a dict describing the result: SAT/UNSAT/TIMEOUT, time, predicates,
    schemas (best effort), and the raw symbol list of the first model.

    The time limit is enforced via an async solve handle: after `time_limit_s`
    seconds we call handle.cancel() which interrupts the solver without
    killing the process.
    """
    import threading

    ctl = clingo.Control(
        [
            f"-c", f"num_objects={num_objects}",
            f"-c", f"max_predicates={max_predicates}",
            f"-c", f"max_static={max_static}",
            f"-c", f"max_precs={max_precs}",
            f"-c", f"opt_synthesis=1",
            f"{models_to_find}",
        ]
    )

    # Load kr21 files via the file loader so #const propagation is consistent.
    for kr21_file in _kr21_files():
        ctl.load(str(kr21_file))

    # Write the graph + injected num_actions to a temporary file under
    # neurosym/lp/ and load it. We do this to keep the same loading path as
    # the kr21 files (and so that errors point to a real path).
    inject = f"num_actions({num_actions}).\n"
    program = inject + graph_lp_source + "\n"
    tmp_lp = LP_OUT_DIR / f"_runner_{int(time.time()*1000)}.lp"
    tmp_lp.write_text(program, encoding="utf-8")
    try:
        ctl.load(str(tmp_lp))

        t0 = time.time()
        ctl.ground([("base", [])])
        ground_time = time.time() - t0
    finally:
        try:
            tmp_lp.unlink()
        except OSError:
            pass

    found_models: list[list[clingo.Symbol]] = []
    timed_out = False

    def on_model(m: clingo.Model) -> None:
        found_models.append(list(m.symbols(shown=True) or m.symbols(atoms=True)))

    t1 = time.time()
    # Use async solve so we can cancel after the time limit.
    handle = ctl.solve(on_model=on_model, async_=True)

    def _cancel() -> None:
        nonlocal timed_out
        timed_out = True
        handle.cancel()

    timer = threading.Timer(time_limit_s, _cancel)
    timer.start()
    try:
        handle.wait()  # blocks until done or cancelled
    finally:
        timer.cancel()

    solve_time = time.time() - t1
    sat_result = handle.get()
    if timed_out:
        sat_str = "TIMEOUT"
    elif sat_result.satisfiable:
        sat_str = "SAT"
    elif sat_result.unsatisfiable:
        sat_str = "UNSAT"
    else:
        sat_str = "UNKNOWN"

    result: dict[str, Any] = {
        "sat": sat_str,
        "ground_time_s": round(ground_time, 3),
        "solve_time_s": round(solve_time, 3),
        "models_found": len(found_models),
        "num_objects": num_objects,
        "num_actions": num_actions,
        "max_predicates": max_predicates,
        "max_static": max_static,
        "stats_summary": _digest_stats(ctl.statistics),
    }

    if found_models:
        first = found_models[0]
        result["first_model_size"] = len(first)
        result["first_model_sample"] = [str(s) for s in first[:60]]
        result["predicates"] = sorted(
            {str(s) for s in first if s.name in {"pred", "p_static"}}
        )
        result["schemas"] = sorted(
            {
                str(s)
                for s in first
                if s.name in {"prec", "eff", "appl", "nappl", "labelname"}
            }
        )[:80]
    return result


def _digest_stats(stats: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        out["choices"] = stats["solving"]["solvers"]["choices"]
        out["conflicts"] = stats["solving"]["solvers"]["conflicts"]
        out["restarts"] = stats["solving"]["solvers"]["restarts"]
    except (KeyError, TypeError):
        pass
    return out


# ---------------------------------------------------------------------------
# Self-test using a known-good Bonet input
# ---------------------------------------------------------------------------


def selftest() -> int:
    """Run the kr21 encoding on Bonet's blocks2ops_2.lp known-good input.

    This verifies that:
    - clingo is installed and the Python API works
    - the kr21 files are reachable and parse cleanly
    - our injection of num_actions and num_objects is consistent with how
      the encoding wants those constants set

    blocks2ops_2 (3 nodes, 4 edges, 2 distinct labels) should be SAT
    with num_objects=2, num_actions=2.
    """
    bench = (
        ARC3_ROOT
        / "external"
        / "learner-strips"
        / "asp"
        / "graphs"
        / "full"
        / "blocks2ops_2.lp"
    )
    if not bench.exists():
        print(f"selftest: missing benchmark file {bench}")
        return 2
    src = bench.read_text(encoding="utf-8")
    print(f"selftest: running kr21 encoding on {bench.name}")
    result = run_asp(
        src,
        num_objects=2,
        num_actions=2,
        max_predicates=4,
        max_static=2,
        time_limit_s=120,
    )
    print(json.dumps(result, indent=2)[:2000])
    return 0 if result["sat"] == "SAT" else 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_for_game(
    game: str,
    num_objects: int,
    num_actions: int | None,
    max_predicates: int,
    time_limit_s: int,
    max_nodes: int | None = None,
) -> int:
    graph_path = GRAPHS_DIR / f"{game}.graph.json"
    if not graph_path.exists():
        candidates = sorted(GRAPHS_DIR.glob(f"{game}*.graph.json"))
        if candidates:
            graph_path = candidates[0]
        else:
            print(f"No graph for {game}; expected {graph_path}")
            return 2

    lp_src, meta = graph_to_lp(graph_path, max_nodes=max_nodes)
    lp_path = LP_OUT_DIR / f"{game}.lp"
    lp_path.write_text(lp_src, encoding="utf-8")
    legal_note = ""
    if meta["has_legal_tags"] and meta["illegal_edges_dropped"]:
        legal_note = f", {meta['illegal_edges_dropped']} illegal edges dropped"
    elif not meta["has_legal_tags"]:
        legal_note = " (no legality tags — re-record with new agent for clean data)"
    start_note = "" if meta["start_node_in_subset"] else " [WARNING: start node not in subset]"
    print(f"wrote {lp_path}  ({meta['num_nodes']}/{meta['num_nodes_total']} nodes, "
          f"{meta['num_transitions']} transitions (+{meta['num_self_loops']} self-loops dropped)"
          f"{legal_note}, {meta['num_actions_effective']} effective labels"
          + (f", no-ops: {meta['noop_labels']}" if meta["noop_labels"] else "")
          + start_note + ")")

    nact = num_actions if num_actions is not None else max(meta["num_actions_effective"], 1)

    print(
        f"running clingo with num_objects={num_objects}, num_actions={nact}, "
        f"max_predicates={max_predicates}, time_limit={time_limit_s}s"
    )
    result = run_asp(
        lp_src,
        num_objects=num_objects,
        num_actions=nact,
        max_predicates=max_predicates,
        time_limit_s=time_limit_s,
    )
    out_path = LP_OUT_DIR / f"{game}.asp_result.json"
    out_path.write_text(json.dumps({"meta": meta, "result": result}, indent=2),
                        encoding="utf-8")
    print(f"\n--- result for {game} ---")
    print(f"  sat:           {result['sat']}")
    print(f"  ground time:   {result['ground_time_s']}s")
    print(f"  solve time:    {result['solve_time_s']}s")
    print(f"  models found:  {result['models_found']}")
    if result.get("first_model_size"):
        print(f"  first model:   {result['first_model_size']} symbols")
        print("  predicates:")
        for p in result.get("predicates", [])[:10]:
            print(f"    {p}")
        print("  schema atoms (first 20):")
        for s in result.get("schemas", [])[:20]:
            print(f"    {s}")
    print(f"\nfull result written to {out_path}")
    return 0 if result["sat"] == "SAT" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true",
                        help="Run kr21 against Bonet's blocks2ops_2 known-good")
    parser.add_argument("--game", type=str,
                        help="game id (e.g. ft09, ls20)")
    parser.add_argument("--num-objects", type=int, default=2)
    parser.add_argument("--num-actions", type=int, default=None,
                        help="default = number of distinct labels in the graph")
    parser.add_argument("--max-predicates", type=int, default=5)
    parser.add_argument("--max-nodes", type=int, default=None,
                        help="Trim graph to BFS neighborhood of terminal nodes "
                             "with at most N nodes. Use ~20 for large graphs to "
                             "keep clingo tractable.")
    parser.add_argument("--time-limit", type=int, default=60)
    args = parser.parse_args(argv)

    if args.selftest:
        return selftest()
    if not args.game:
        parser.error("either --selftest or --game required")
    return run_for_game(
        args.game,
        num_objects=args.num_objects,
        num_actions=args.num_actions,
        max_predicates=args.max_predicates,
        max_nodes=args.max_nodes,
        time_limit_s=args.time_limit,
    )


if __name__ == "__main__":
    sys.exit(main())
