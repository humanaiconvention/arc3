"""
Phase B prototype — transition graph builder over ARC-AGI-3 random-baseline
recordings.

Reads .recording.jsonl files written by the official ARC-AGI-3-Agents
recorder, builds a transition graph (nodes = distinct frames, edges =
observed transitions), and reports the stats that matter for deciding
whether the symbolic side of the architecture has any signal on each game.

Critical caveat about the input format
--------------------------------------
The official recorder appends one entry per FrameData *after* each action,
but does NOT capture the action that produced the frame. action_input.id is
always 0 in the recordings we have. So for this first prototype the
edges are unlabeled — they record "this frame followed that frame", not
"this frame followed that frame *via action k*".

This is enough for a first signal:
- node count, edge count, branching factor, connected components
- bisimulation under unlabeled edges (lower bound on symbolic compressibility)
- whether GAME_OVER / WIN frames are reachable in the explored fragment

The next iteration needs a recording-aware exploration agent that emits
{"action": k, "x": ?, "y": ?} records interleaved with frames. That is
written separately as `record_actions_agent.py`.

Usage:
    cd D:/arc3
    python neurosym/phase_b_graph.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ARC3_ROOT = Path(__file__).resolve().parents[1]
RECORDINGS_DIR = ARC3_ROOT / "external" / "ARC-AGI-3-Agents" / "recordings"
OUTPUT_DIR = ARC3_ROOT / "neurosym" / "graphs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def frame_hash(frame: Any) -> str:
    """Stable hash of a 3D frame (channels x H x W)."""
    return hashlib.blake2b(
        json.dumps(frame, separators=(",", ":")).encode("utf-8"),
        digest_size=12,
    ).hexdigest()


def load_recording(path: Path) -> list[dict[str, Any]]:
    """Read a .recording.jsonl file. Returns BOTH frames and action_request
    entries in temporal order. Strips the trailing scorecard event.

    The official recorder writes only frames; the arc3 RandomRecordActions
    subclass interleaves entries of shape:
        {"data": {"event": "action_request", "action_id": int, ...}}
    immediately *before* the action is submitted, so the JSONL ordering is
        action_request_0, frame_1, action_request_1, frame_2, ...
    Note there is no initial frame — frame_0 (the env reset state) is not
    captured by the recorder.
    """
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Trailing scorecard entry has no "data" wrapper -> skip.
        if "data" not in obj:
            continue
        d = obj["data"]
        # Keep frames AND action_request entries.
        if "frame" in d:
            out.append({"kind": "frame", **obj})
        elif d.get("event") == "action_request":
            out.append({"kind": "action", **obj})
    return out


def build_graph(
    events: list[dict[str, Any]],
    collapse_complex: bool = False,
) -> dict[str, Any]:
    """Build a transition graph from a sequence of (action, frame, ...) events.

    The expected stream is interleaved:
        action_request_0, frame_1, action_request_1, frame_2, ...

    For each frame_t we record an edge:
        (last_seen_frame_or_None) --label-of-most-recent-action--> frame_t

    The very first frame has no predecessor (no frame_0 is captured by the
    recorder), so it becomes the start node.

    Returns a dict with:
        nodes:    {hash: {"first_seen_at": int, "state": str, "available_actions": list, "level": int}}
        edges:    [(src_hash, dst_hash, label)]
        node_count, edge_count, distinct_edge_count
        branching: max out-degree
        components: number of weakly-connected components
        terminal_nodes: list of hashes that are GAME_OVER or WIN
        action_label_distribution: counter over edge labels
        edges_with_action: count of edges that carried a real action label (not 'a?')
    """
    nodes: dict[str, dict[str, Any]] = {}
    # edges now carry a 4th element: legal (True/False/None)
    edges: list[tuple[str, str, str, bool | None]] = []
    out_neighbours: dict[str, set[tuple[str, str]]] = defaultdict(set)
    in_neighbours: dict[str, set[tuple[str, str]]] = defaultdict(set)

    prev_hash: str | None = None
    pending_action_label: str = "a?"  # used until we see an action_request
    pending_legal: bool | None = None  # None = old recording without legal tag
    real_action_count = 0
    illegal_action_count = 0
    label_counter: Counter[str] = Counter()

    for i, ev in enumerate(events):
        kind = ev.get("kind")
        d = ev["data"]
        if kind == "action":
            # Capture the action that's about to be submitted.
            aid = d.get("action_id", "?")
            is_complex = bool(d.get("is_complex"))
            adata = d.get("action_data", {}) or {}
            if collapse_complex or not is_complex or "x" not in adata or "y" not in adata:
                pending_action_label = f"a{aid}"
            else:
                # Bin the (x,y) into a coarse grid so the label set stays small.
                xb = int(adata["x"]) // 16
                yb = int(adata["y"]) // 16
                pending_action_label = f"a{aid}_x{xb}y{yb}"
            # Propagate the legality tag if present (new-format recordings only).
            pending_legal = d.get("legal", None)
            continue

        if kind != "frame":
            continue

        h = frame_hash(d["frame"])
        if h not in nodes:
            nodes[h] = {
                "first_seen_at": i,
                "state": d.get("state", "?"),
                "available_actions": d.get("available_actions", []),
                "levels_completed": d.get("levels_completed", 0),
            }
        else:
            # If we re-encounter the node and the state changed, update.
            if d.get("state") in {"GAME_OVER", "WIN"}:
                nodes[h]["state"] = d["state"]

        if prev_hash is not None:
            label = pending_action_label
            label_counter[label] += 1
            if label != "a?":
                real_action_count += 1
            if pending_legal is False:
                illegal_action_count += 1
            edges.append((prev_hash, h, label, pending_legal))
            out_neighbours[prev_hash].add((h, label))
            in_neighbours[h].add((prev_hash, label))
        # Reset pending after consuming
        pending_action_label = "a?"
        pending_legal = None
        prev_hash = h

    # Distinct edges: (src, dst, label) triples — legal flag is not part of
    # the deduplication key (same (s,t,l) observed legal and illegal → one edge).
    distinct_edges = {(s, t, l) for (s, t, l, _leg) in edges}
    # For output, attach the most informative legality value per (s,t,l) triple.
    # If any observation was legal, mark the triple legal; None if all unknown.
    edge_legal: dict[tuple[str, str, str], bool | None] = {}
    for s, t, l, leg in edges:
        prev = edge_legal.get((s, t, l))
        if leg is True or prev is True:
            edge_legal[(s, t, l)] = True
        elif leg is False and prev is None:
            edge_legal[(s, t, l)] = False
        elif prev is None and leg is None:
            edge_legal[(s, t, l)] = None

    branching = max((len(v) for v in out_neighbours.values()), default=0)

    # Weakly-connected components via union-find
    parent: dict[str, str] = {h: h for h in nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for s, t, l, _ in edges:
        union(s, t)
    components = len({find(h) for h in nodes})

    terminal_nodes = [h for h, n in nodes.items() if n["state"] in {"GAME_OVER", "WIN"}]

    # Count legal vs illegal edges (only for recordings with legality tags).
    has_legal_tags = any(leg is not None for _, _, _, leg in edges)
    legal_edge_count = sum(1 for _, _, _, leg in edges if leg is True) if has_legal_tags else None
    illegal_edge_count = sum(1 for _, _, _, leg in edges if leg is False) if has_legal_tags else None

    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "distinct_edge_count": len(distinct_edges),
        "branching": branching,
        "components": components,
        "terminal_nodes": terminal_nodes,
        "nodes": nodes,
        "edges": [
            {"src": s, "dst": t, "label": l, "legal": edge_legal.get((s, t, l))}
            for (s, t, l) in distinct_edges
        ],
        "out_neighbours": {k: sorted(list(v)) for k, v in out_neighbours.items()},
        "edges_with_action": real_action_count,
        "illegal_edges": illegal_edge_count,
        "has_legal_tags": has_legal_tags,
        "action_label_counts": dict(label_counter),
        "distinct_action_labels": sorted(label_counter.keys()),
    }


def bisimulation_partition(graph: dict[str, Any]) -> dict[str, int]:
    """Coarsest stable partition under unlabeled (label-only) successor signature.

    Implements paige-tarjan-flavoured fixpoint refinement: nodes start in one
    block; refine by mapping each node to (current_block, sorted multiset of
    (label, successor_block)) until stable.
    """
    nodes = graph["nodes"]
    out_neighbours = graph["out_neighbours"]
    block: dict[str, int] = {h: 0 for h in nodes}

    while True:
        signatures: dict[str, tuple] = {}
        for h in nodes:
            successors = out_neighbours.get(h, [])
            sig = (
                block[h],
                tuple(sorted((label, block[t]) for (t, label) in successors)),
            )
            signatures[h] = sig
        sig_to_id: dict[tuple, int] = {}
        new_block: dict[str, int] = {}
        for h, sig in signatures.items():
            if sig not in sig_to_id:
                sig_to_id[sig] = len(sig_to_id)
            new_block[h] = sig_to_id[sig]
        if new_block == block:
            return block
        block = new_block


def short(h: str) -> str:
    return h[:8]


def analyse_one(rec_path: Path, collapse_complex: bool = False) -> dict[str, Any]:
    events = load_recording(rec_path)
    frame_count = sum(1 for e in events if e.get("kind") == "frame")
    action_count = sum(1 for e in events if e.get("kind") == "action")
    graph = build_graph(events, collapse_complex=collapse_complex)
    partition = bisimulation_partition(graph)
    bi_blocks = len(set(partition.values()))
    summary = {
        "recording": rec_path.name,
        "events_total": len(events),
        "frames": frame_count,
        "actions_recorded": action_count,
        "node_count": graph["node_count"],
        "edge_count": graph["edge_count"],
        "distinct_edges": graph["distinct_edge_count"],
        "edges_with_action_label": graph["edges_with_action"],
        "distinct_action_labels": graph["distinct_action_labels"],
        "branching_max": graph["branching"],
        "weakly_connected_components": graph["components"],
        "bisimulation_blocks": bi_blocks,
        "compression_ratio": (
            graph["node_count"] / bi_blocks if bi_blocks else None
        ),
        "terminal_node_count": len(graph["terminal_nodes"]),
        "terminal_states_seen": sorted(
            {
                graph["nodes"][h]["state"]
                for h in graph["terminal_nodes"]
            }
        ),
        "available_actions_seen": sorted(
            {
                tuple(n["available_actions"])
                for n in graph["nodes"].values()
            }
        ),
        "illegal_edges": graph["illegal_edges"],
        "has_legal_tags": graph["has_legal_tags"],
    }
    out_json = {
        "summary": summary,
        "nodes": [
            {
                "id": h,
                "short": short(h),
                **n,
                "block": partition[h],
            }
            for h, n in graph["nodes"].items()
        ],
        # edges is now a list of dicts: {src, dst, label, legal}
        # short() the hashes for compactness.
        "edges": [
            {
                "src": short(e["src"]) if len(e["src"]) > 8 else e["src"],
                "dst": short(e["dst"]) if len(e["dst"]) > 8 else e["dst"],
                "label": e["label"],
                "legal": e.get("legal"),
            }
            for e in graph["edges"]
        ],
    }
    # Take the short game prefix (before the first '-') as the file stem so
    # that downstream tools can look up by short id (`ft09`, `ls20`, ...).
    short_game = rec_path.name.split("-")[0]
    out_path = OUTPUT_DIR / f"{short_game}.graph.json"
    out_path.write_text(json.dumps(out_json, indent=2), encoding="utf-8")
    return summary


def _pick_recordings(prefer_pattern: str = "*randomrecordactions*") -> list[Path]:
    """Pick the most recent recording per game, preferring action-recording runs."""
    by_game: dict[str, Path] = {}
    candidates = sorted(RECORDINGS_DIR.glob("*.recording.jsonl"))
    # First pass: prefer randomrecordactions files (newest per game).
    preferred = sorted(
        RECORDINGS_DIR.glob(prefer_pattern + ".recording.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for p in preferred:
        game = p.name.split("-")[0]
        if game not in by_game:
            by_game[game] = p
    # Backfill from any other recordings if a game isn't covered yet.
    for p in candidates:
        game = p.name.split("-")[0]
        if game not in by_game:
            by_game[game] = p
    return [by_game[g] for g in sorted(by_game.keys())]


def main() -> int:
    import argparse as _argparse
    p = _argparse.ArgumentParser(description="Phase B graph builder")
    p.add_argument(
        "--collapse-complex",
        action="store_true",
        help="Collapse ACTION6_x?y? variants into a single 'a6' label. "
        "Use this for the first ASP pass to keep num_actions small.",
    )
    args = p.parse_args()
    recordings = _pick_recordings()
    if not recordings:
        print(f"No recordings found in {RECORDINGS_DIR}")
        return 1

    print(f"Phase B graph builder — {len(recordings)} recording(s)")
    print(f"Reading from: {RECORDINGS_DIR}")
    print(f"Writing to:   {OUTPUT_DIR}")
    print(f"collapse_complex = {args.collapse_complex}")
    for r in recordings:
        print(f"  {r.name}")
    print()

    summaries = []
    for path in recordings:
        s = analyse_one(path, collapse_complex=args.collapse_complex)
        summaries.append(s)

    # Pretty-print summary table
    print("=" * 110)
    print(
        f"{'game':6}  {'frames':>6}  {'acts':>4}  {'nodes':>6}  {'edges':>6}  "
        f"{'dist':>5}  {'a-lbl':>5}  {'branch':>6}  {'comp':>5}  {'biblk':>6}  "
        f"{'cmpr':>5}  {'term':>4}"
    )
    print("-" * 110)
    for s in summaries:
        game = s["recording"].split("-")[0]
        cmpr = (
            f"{s['compression_ratio']:.2f}" if s["compression_ratio"] else "—"
        )
        print(
            f"{game:6}  {s['frames']:>6}  {s['actions_recorded']:>4}  "
            f"{s['node_count']:>6}  "
            f"{s['edge_count']:>6}  {s['distinct_edges']:>5}  "
            f"{len(s['distinct_action_labels']):>5}  "
            f"{s['branching_max']:>6}  {s['weakly_connected_components']:>5}  "
            f"{s['bisimulation_blocks']:>6}  {cmpr:>5}  "
            f"{s['terminal_node_count']:>4}"
        )
    print("=" * 110)
    print()
    print("Legend:")
    print("  frames    = frame events in JSONL")
    print("  acts      = action_request events in JSONL (0 if recorder didn't log them)")
    print("  nodes     = distinct frames observed")
    print("  edges     = total transitions")
    print("  dist      = unique (src, dst, label) triples")
    print("  a-lbl     = number of distinct action labels on edges")
    print("  branch    = max out-degree")
    print("  comp      = weakly connected components")
    print("  biblk     = blocks after labeled bisimulation refinement")
    print("  cmpr      = node_count / biblk (higher = more compressible)")
    print("  term      = number of terminal-state nodes (GAME_OVER or WIN)")

    summary_json = OUTPUT_DIR / "phase_b_summary.json"
    summary_json.write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    print(f"\nSummary written to {summary_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
