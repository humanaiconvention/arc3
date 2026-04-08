"""
Phase F — PDDL serialisation + planner dispatch.

Given Phase D (STRIPS schema) and Phase E (initial state + goal condition),
Phase F:

1. Re-runs clingo to extract the COMPLETE prec/eff schema (Phase D only
   saves 80 schema atoms; with 11 actions the prec atoms are cut off).
2. Filters prec/eff atoms to those with valid role indices (≤ num_objects)
   — atoms with role > num_objects are vacuous (pnext never fires for
   out-of-range positions) and should be ignored.
3. Serialises the filtered schema + initial state + goal to PDDL:
     lp/<game>_domain.pddl
     lp/<game>_problem.pddl
4. Runs pyperplan (BFS) on the generated PDDL.
5. Falls back to graph-BFS on the observed state-space if pyperplan is
   unavailable or returns no plan.
6. Writes lp/<game>.phase_f.json.

PDDL translation conventions
-----------------------------
- Objects: obj1, obj2 (= BG integers 1, 2)
- Predicates: p1..pM (fluent), pM+1..pN (static, but listed in :predicates
  for compatibility and also in :init — pyperplan handles them correctly
  since no action has effects on them).
- Action role 1 → PDDL parameter ?o1, role 2 → ?o2.
- prec(A,(P,(R1,R2)),1) → positive precondition (pP ?params[R1] ?params[R2])
- prec(A,(P,(R1,R2)),0) → negative precondition (not (pP ...))
- eff(A,(P,(R1,R2)),1)  → positive effect
- eff(A,(P,(R1,R2)),0)  → negative effect

Known caveat
------------
The BG encoding was designed for small demonstration traces, NOT full random-
walk graphs. The start state (all fluents false) may or may not be reachable to
the goal in the induced STRIPS domain, depending on how well the random trace
sampled the relevant transitions. The graph-BFS fallback always works on the
observed state-space directly.

Usage:
    cd D:/arc3
    python neurosym/phase_f_plan.py --game vc33
    python neurosym/phase_f_plan.py --all
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import clingo  # type: ignore

ARC3_ROOT = Path(__file__).resolve().parents[1]
GRAPHS_DIR = ARC3_ROOT / "neurosym" / "graphs"
LP_OUT_DIR = ARC3_ROOT / "neurosym" / "lp"
KR21_DIR = ARC3_ROOT / "external" / "learner-strips" / "asp" / "clingo" / "kr21"


# ---------------------------------------------------------------------------
# Full schema extraction (re-runs clingo)
# ---------------------------------------------------------------------------


def _run_clingo_schema(
    lp_src: str,
    num_actions: int,
    num_objects: int = 2,
    max_predicates: int = 5,
    max_static: int = 2,
    max_precs: int = 6,
    time_limit_s: int = 60,
) -> list[clingo.Symbol] | None:
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
    tmp = LP_OUT_DIR / f"_phase_f_{int(time.time()*1000)}.lp"
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

    return found[0] if (not timed_out and found) else None


def extract_schema(
    symbols: list[clingo.Symbol],
    num_objects: int,
    label_remap: dict[str, int],
) -> dict[str, Any]:
    """Extract prec, eff, labelname atoms; filter to valid roles (≤ num_objects).

    Returns:
        actions: dict{action_int -> {'precs': [...], 'effs': [...], 'name': str}}
        predicates: list of predicate ints
        static_predicates: set of static predicate ints
    """
    rev_label: dict[int, str] = {v: k for k, v in label_remap.items()}

    actions: dict[int, dict] = {}
    predicates: set[int] = set()
    static_predicates: set[int] = set()

    for sym in symbols:
        name = sym.name
        args = sym.arguments

        if name == "labelname":
            # labelname(action_int, "label_string")
            a = args[0].number
            lbl = str(args[1]).strip('"')
            actions.setdefault(a, {"precs": [], "effs": [], "name": lbl})
            actions[a]["name"] = lbl

        elif name in {"pred", "p_static"}:
            p = args[0].number
            predicates.add(p)
            if name == "p_static":
                static_predicates.add(p)

        elif name == "prec":
            # prec(A, (P, (V1, V2)), Bool)
            a = args[0].number
            inner = args[1].arguments  # [P, (V1,V2)]
            p = inner[0].number
            roles = inner[1].arguments  # [V1, V2]
            v1, v2 = roles[0].number, roles[1].number
            val = args[2].number
            # Only include valid roles (≤ num_objects)
            if v1 <= num_objects and v2 <= num_objects:
                actions.setdefault(a, {"precs": [], "effs": [], "name": f"a{a}"})
                actions[a]["precs"].append((p, v1, v2, val))
                predicates.add(p)

        elif name == "eff":
            # eff(A, (P, (V1, V2)), Bool)
            a = args[0].number
            inner = args[1].arguments
            p = inner[0].number
            roles = inner[1].arguments
            v1, v2 = roles[0].number, roles[1].number
            val = args[2].number
            if v1 <= num_objects and v2 <= num_objects:
                actions.setdefault(a, {"precs": [], "effs": [], "name": f"a{a}"})
                actions[a]["effs"].append((p, v1, v2, val))
                predicates.add(p)

    return {
        "actions": actions,
        "predicates": sorted(predicates),
        "static_predicates": sorted(static_predicates),
    }


# ---------------------------------------------------------------------------
# PDDL serialisation
# ---------------------------------------------------------------------------

_OBJ_NAME = {1: "obj1", 2: "obj2"}


def _role_to_param(role: int) -> str:
    return f"?o{role}"


def _pred_name(p: int) -> str:
    return f"p{p}"


def _obj_name(o: int) -> str:
    return _OBJ_NAME.get(o, f"obj{o}")


def serialise_domain(
    game: str,
    schema: dict[str, Any],
    num_objects: int,
) -> str:
    """Serialize the induced STRIPS schema to a PDDL domain string."""
    actions = schema["actions"]
    predicates = schema["predicates"]
    static_preds = set(schema["static_predicates"])

    # Use untyped parameters — pyperplan's type system can mishandle typed objects
    params_str = " ".join(f"?o{i}" for i in range(1, num_objects + 1))

    lines = [
        f"(define (domain {game}-induced)",
        "  (:requirements :strips :negative-preconditions)",
        "  (:predicates",
    ]
    for p in predicates:
        pred_args = " ".join(f"?x{i}" for i in range(1, num_objects + 1))
        lines.append(f"    ({_pred_name(p)} {pred_args})")
    lines.append("  )")

    for a_int, adict in sorted(actions.items()):
        precs = adict["precs"]
        effs = adict["effs"]
        aname = adict["name"].replace(".", "_").replace(",", "_")

        lines.append(f"  (:action {aname}")
        lines.append(f"    :parameters ({params_str})")

        # Preconditions
        prec_parts: list[str] = []
        for p, v1, v2, val in precs:
            atom = f"({_pred_name(p)} {_role_to_param(v1)} {_role_to_param(v2)})"
            prec_parts.append(atom if val == 1 else f"(not {atom})")
        # pyperplan requires non-empty formula; use (and) for no preconditions
        lines.append("    :precondition (and")
        for part in prec_parts:
            lines.append(f"      {part}")
        lines.append("    )")

        # Effects (skip static predicates — they shouldn't change)
        eff_parts: list[str] = []
        for p, v1, v2, val in effs:
            if p in static_preds:
                continue
            atom = f"({_pred_name(p)} {_role_to_param(v1)} {_role_to_param(v2)})"
            eff_parts.append(atom if val == 1 else f"(not {atom})")
        lines.append("    :effect (and")
        for part in eff_parts:
            lines.append(f"      {part}")
        lines.append("    )")

        lines.append("  )")

    lines.append(")")
    return "\n".join(lines) + "\n"


def serialise_problem(
    game: str,
    phase_e: dict[str, Any],
    schema: dict[str, Any],
    num_objects: int,
) -> str:
    """Serialize the initial state and goal to a PDDL problem string."""
    init_info = phase_e["initial_state"]
    goal_info = phase_e["goal"]
    static_preds = set(schema["static_predicates"])

    obj_list = " ".join(_obj_name(i) for i in range(1, num_objects + 1))

    lines = [
        f"(define (problem {game}-plan)",
        f"  (:domain {game}-induced)",
        f"  (:objects {obj_list})",
        "  (:init",
    ]

    # Static true atoms (always true, type-agnostic)
    for k in init_info["static_true"]:
        # k looks like "(4,(1,2))"
        import ast
        try:
            tup = ast.literal_eval(k)
            p, (o1, o2) = tup
            if o1 <= num_objects and o2 <= num_objects:
                lines.append(
                    f"    ({_pred_name(p)} {_obj_name(o1)} {_obj_name(o2)})"
                )
        except Exception:
            pass

    # Fluent true atoms at initial state
    for k in init_info["true_fluents"]:
        import ast
        try:
            tup = ast.literal_eval(k)
            p, (o1, o2) = tup
            if o1 <= num_objects and o2 <= num_objects:
                lines.append(
                    f"    ({_pred_name(p)} {_obj_name(o1)} {_obj_name(o2)})"
                )
        except Exception:
            pass

    lines.append("  )")

    # Goal: conjunction of fluent atoms from Phase E
    goal_parts: list[str] = []
    for k in goal_info["goal_conjunction"]:
        import ast
        try:
            tup = ast.literal_eval(k)
            p, (o1, o2) = tup
            if o1 <= num_objects and o2 <= num_objects:
                goal_parts.append(
                    f"({_pred_name(p)} {_obj_name(o1)} {_obj_name(o2)})"
                )
        except Exception:
            pass

    if goal_parts:
        lines.append("  (:goal (and")
        for part in goal_parts:
            lines.append(f"    {part}")
        lines.append("  ))")
    else:
        lines.append("  (:goal ())")

    lines.append(")")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# pyperplan dispatch
# ---------------------------------------------------------------------------


def run_pyperplan(domain_path: Path, problem_path: Path) -> dict[str, Any]:
    """Run pyperplan BFS on the generated PDDL.  Returns a result dict."""
    try:
        from pyperplan.planner import _parse, _ground, _search, SEARCHES
    except ImportError:
        return {"available": False, "error": "pyperplan not installed"}

    try:
        task = _parse(str(domain_path), str(problem_path))
        grounded = _ground(task)
        bfs = SEARCHES["bfs"]
        plan = _search(grounded, bfs, None)
        if plan is None:
            return {"available": True, "plan": None, "length": None,
                    "note": "no plan found (search exhausted)"}
        return {
            "available": True,
            "plan": [op.name for op in plan],
            "length": len(plan),
        }
    except Exception as exc:
        return {"available": True, "error": str(exc), "plan": None}


# ---------------------------------------------------------------------------
# Graph-BFS fallback
# ---------------------------------------------------------------------------


def graph_bfs(
    graph: dict,
    phase_e: dict[str, Any],
    meta: dict[str, Any],
) -> dict[str, Any]:
    """BFS on the full observed state-space graph from start to terminal.

    This uses the original Phase B graph (all 51 nodes, not just the 20-node
    subgraph). It bypasses the STRIPS abstraction and finds the shortest path
    that was actually observed in the recording.
    """
    nodes = graph["nodes"]
    node_by_short = {n["short"]: n for n in nodes}

    start_node = min(nodes, key=lambda n: n.get("first_seen_at", 0))
    start_short = start_node["short"]
    terminal_shorts = {n["short"] for n in nodes if n.get("state") in {"WIN", "GAME_OVER"}}

    # Build forward adjacency (legal, non-self-loop edges)
    adj: dict[str, list[tuple[str, str]]] = {}
    for e in graph["edges"]:
        if e.get("legal") is False or e["src"] == e["dst"]:
            continue
        adj.setdefault(e["src"], []).append((e["dst"], e["label"]))

    if not terminal_shorts:
        return {"found": False, "note": "no terminal states in graph"}

    queue: deque = deque([(start_short, [])])
    visited: set[str] = {start_short}
    while queue:
        node, path = queue.popleft()
        if node in terminal_shorts:
            # Translate labels back to action information
            label_remap = meta["label_remap"]
            rev_label = {v: k for k, v in label_remap.items()}
            step_list = []
            for src, dst, lbl in path:
                src_node = node_by_short[src]
                dst_node = node_by_short[dst]
                action_int = label_remap.get(lbl)
                step_list.append({
                    "step": len(step_list) + 1,
                    "from_short": src[:8],
                    "to_short": dst[:8],
                    "label": lbl,
                    "action_int": action_int,
                    "in_subgraph": dst in meta["node_remap"],
                })
            return {
                "found": True,
                "length": len(path),
                "terminal_short": node[:8],
                "terminal_state": node_by_short[node].get("state"),
                "steps": step_list,
                "note": "shortest path in observed state-space (BFS on Phase B graph)",
            }
        for dst, lbl in adj.get(node, []):
            if dst not in visited:
                visited.add(dst)
                queue.append((dst, path + [(node, dst, lbl)]))

    return {"found": False, "note": "no path from start to terminal in observed graph"}


# ---------------------------------------------------------------------------
# Per-game Phase F runner
# ---------------------------------------------------------------------------


def run_for_game(game: str, time_limit_s: int = 60) -> int:
    # Load all prerequisites
    asp_path = LP_OUT_DIR / f"{game}.asp_result.json"
    lp_path = LP_OUT_DIR / f"{game}.lp"
    phase_e_path = LP_OUT_DIR / f"{game}.phase_e.json"
    graph_path = GRAPHS_DIR / f"{game}.graph.json"

    for p in (asp_path, lp_path, phase_e_path, graph_path):
        if not p.exists():
            print(f"[{game}] Missing: {p} — run Phase D and Phase E first.")
            return 2

    with open(asp_path) as f:
        asp = json.load(f)
    with open(phase_e_path) as f:
        pe = json.load(f)
    with open(graph_path) as f:
        graph = json.load(f)

    meta = asp["meta"]
    result = asp["result"]

    if result["sat"] != "SAT":
        print(f"[{game}] Phase D is {result['sat']} — Phase F requires SAT.")
        return 1

    phase_e = pe["phase_e"]

    if not phase_e["goal"]["goal_conjunction"]:
        print(f"[{game}] No goal condition from Phase E. "
              "Running graph-BFS fallback only.")
        graph_result = graph_bfs(graph, phase_e, meta)
        _print_graph_result(game, graph_result)
        out = {"game": game, "pddl": None, "pyperplan": None,
               "graph_bfs": graph_result}
        _write(game, out)
        return 0 if graph_result["found"] else 1

    # Re-run clingo for full schema.
    print(f"[{game}] Re-running clingo for full schema ...")
    lp_src = lp_path.read_text(encoding="utf-8")
    t0 = time.time()
    symbols = _run_clingo_schema(
        lp_src,
        num_actions=meta["num_actions_effective"],
        num_objects=result["num_objects"],
        max_predicates=result["max_predicates"],
        time_limit_s=time_limit_s,
    )
    elapsed = time.time() - t0
    print(f"[{game}] Schema extracted: {len(symbols) if symbols else 0} symbols in {elapsed:.2f}s")

    if not symbols:
        print(f"[{game}] Clingo returned no model. Cannot build PDDL domain.")
        return 1

    # Extract schema.
    schema = extract_schema(symbols, result["num_objects"], meta["label_remap"])
    n_actions = len(schema["actions"])
    n_valid_prec = sum(len(v["precs"]) for v in schema["actions"].values())
    n_valid_eff = sum(len(v["effs"]) for v in schema["actions"].values())
    print(f"[{game}] Schema: {n_actions} actions, "
          f"{n_valid_prec} valid prec atoms, {n_valid_eff} valid eff atoms")

    # Serialise PDDL.
    domain_str = serialise_domain(game, schema, result["num_objects"])
    problem_str = serialise_problem(game, phase_e, schema, result["num_objects"])

    domain_path = LP_OUT_DIR / f"{game}_domain.pddl"
    problem_path = LP_OUT_DIR / f"{game}_problem.pddl"
    domain_path.write_text(domain_str, encoding="utf-8")
    problem_path.write_text(problem_str, encoding="utf-8")
    print(f"[{game}] PDDL written:")
    print(f"  domain:  {domain_path}")
    print(f"  problem: {problem_path}")

    # pyperplan.
    print(f"[{game}] Running pyperplan BFS ...")
    pp_result = run_pyperplan(domain_path, problem_path)

    if not pp_result.get("available"):
        print(f"[{game}] pyperplan not available: {pp_result.get('error')}")
    elif pp_result.get("error"):
        print(f"[{game}] pyperplan error: {pp_result['error']}")
    elif pp_result.get("plan") is not None:
        print(f"[{game}] PLAN FOUND (length={pp_result['length']}):")
        for step in pp_result["plan"]:
            print(f"  {step}")
    else:
        print(f"[{game}] pyperplan: {pp_result.get('note', 'no plan')}")

    # Graph-BFS fallback.
    print(f"\n[{game}] Graph-BFS fallback (full observed state-space) ...")
    graph_result = graph_bfs(graph, phase_e, meta)
    _print_graph_result(game, graph_result)

    # Output.
    out = {
        "game": game,
        "pddl": {
            "domain_path": str(domain_path),
            "problem_path": str(problem_path),
            "num_actions": n_actions,
            "num_valid_prec": n_valid_prec,
            "num_valid_eff": n_valid_eff,
        },
        "pyperplan": pp_result,
        "graph_bfs": graph_result,
    }
    _write(game, out)
    return 0


def _print_graph_result(game: str, r: dict) -> None:
    if r.get("found"):
        print(f"[{game}] Graph-BFS: PLAN FOUND (length={r['length']})")
        for step in r.get("steps", [])[:10]:
            tag = " [subgraph]" if step.get("in_subgraph") else ""
            print(f"  step {step['step']:2d}: {step['from_short']} --[{step['label']}]--> {step['to_short']}{tag}")
        if r["length"] > 10:
            print(f"  ... ({r['length'] - 10} more steps)")
        print(f"  --> reach {r['terminal_short']} ({r['terminal_state']})")
    else:
        print(f"[{game}] Graph-BFS: no plan. {r.get('note', '')}")


def _write(game: str, data: dict) -> None:
    out_path = LP_OUT_DIR / f"{game}.phase_f.json"
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\n[{game}] Phase F result written to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", type=str, help="game id (e.g. vc33)")
    parser.add_argument("--all", action="store_true",
                        help="run Phase F for all games with Phase D+E results")
    parser.add_argument("--time-limit", type=int, default=60)
    args = parser.parse_args(argv)

    if args.all:
        rc = 0
        for p in sorted(LP_OUT_DIR.glob("*.phase_e.json")):
            g = p.stem.replace(".phase_e", "")
            rc |= run_for_game(g, time_limit_s=args.time_limit)
        return rc

    if not args.game:
        parser.error("either --game or --all required")
    return run_for_game(args.game, time_limit_s=args.time_limit)


if __name__ == "__main__":
    sys.exit(main())
