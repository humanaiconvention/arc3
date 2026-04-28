"""GameProfile — output of characterize.py.

A GameProfile describes what a game looks like before we try to solve it.
Strategies consume profiles to decide (a) whether to attempt the game and
(b) how to parameterize themselves.

Keep this dataclass purely JSON-serializable (no numpy arrays in fields;
convert to lists). The raw initial frame is stored separately as .npy.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ActionEffect:
    """Effect of pressing a single action once from the initial frame."""
    action_id: int
    cells_changed: int          # total cells that changed value
    rows_changed: list          # sorted list of row indices that changed
    cols_changed: list          # sorted list of col indices that changed
    n_level_gained: int = 0     # did this action complete a level?
    is_terminal: bool = False   # did the action end the game?
    exception: str = ""         # step() exception if any
    delta_life: int = 0         # change in life-row non-zero count


@dataclass
class GameProfile:
    """Complete characterization of a game's initial state + action effects."""
    game_id: str                               # full id with hash
    game_prefix: str                           # short id (first 4-5 chars)
    win_levels: int = 0
    available_actions: list = field(default_factory=list)
    frame_shape: list = field(default_factory=list)  # [H, W]

    # Per-action effect summary
    action_effects: dict = field(default_factory=dict)  # {str(act_id): ActionEffect dict}

    # Structural features
    life_row: int = 63
    life_cells_initial: int = 0
    n_static_rows: int = 0                     # rows untouched by any action
    n_dynamic_rows: int = 0                    # rows touched by at least one action
    high_entropy_regions: list = field(default_factory=list)  # detected "busy" rectangles

    # Heuristic signals for strategy ranking
    looks_like_combo_lock: float = 0.0         # 0-1 confidence score
    looks_like_grid_click: float = 0.0
    looks_like_cursor_game: float = 0.0
    looks_like_static: float = 0.0             # game doesn't respond to anything

    # Coordinate-click heat-map: result of probing a sparse grid with the click
    # action. Each entry: {"x", "y", "cells_changed", "act"}. Sorted desc by
    # cells_changed. Empty if game has no click action or probe was skipped.
    # Populated by characterize when action 5 or 6 is available.
    click_heatmap: list = field(default_factory=list)
    click_heatmap_meta: dict = field(default_factory=dict)  # {grid_step, click_act, n_probes, n_terminal, danger_aborted}

    # Action-pair non-commutativity probe: per ordered nav-pair (a, b),
    # records final-frame hash + cell deltas. derived_pair_signals summarizes
    # non-commutative pairs, idempotent actions, and conditional-effect
    # candidates (b is no-op alone but active after a).
    action_pair_effects: dict = field(default_factory=dict)
    derived_pair_signals: dict = field(default_factory=dict)

    # Higher-level derived signals computed from action_pair_effects + heatmap.
    # inverse_pairs: pairs (a, b) where (a, b) returns to base frame — a undoes b.
    #   E.g., re86 inferred (1,2) and (3,4) — pure direction-pair structure.
    # toggle_actions: actions where (a, a) returns to base — applying twice cancels.
    #   E.g., re86 act5 — likely a select/commit toggle.
    # looks_like_cursor_click: 0-1 confidence the click action is just cursor
    #   placement (uniform low cells_changed across heatmap), not a state effect.
    inverse_pairs: list = field(default_factory=list)       # [[a, b], ...]
    toggle_actions: list = field(default_factory=list)      # [a, ...]
    looks_like_cursor_click: float = 0.0

    # 3-action cycle probe: ordered triples (a, b, c) over the available pair-
    # probe actions. Only populated when characterize detects 3+ inverse pairs
    # (the triangular-subgroup signature on tu93, g50t). Used to distinguish
    # "true 3-cycle group structure" from "pairwise-cancellation only".
    action_triple_effects: dict = field(default_factory=dict)   # {"a_b_c": {...}}
    three_cycles: list = field(default_factory=list)            # [[a,b,c], ...] returning to base

    # State-conditional pair-response probe: for games with looks_like_static
    # high (initial state is unresponsive), record cells_changed per next
    # action AFTER applying each available nav as a 3-step prelude. Surfaces
    # the "unfreeze lever" pattern (e.g. g50t acts 2/4) and dominant nav
    # actions hidden behind a static initial state.
    post_prelude_responses: dict = field(default_factory=dict)
    # Schema: {"act<P>_x<N>": {"unfreeze": bool, "responses": {act_id: cells_changed, ...}}}

    # dominant_actions: top-K actions by cells_changed. Computed from initial
    # action_effects for responsive games OR from post_prelude_responses for
    # static games. Strategies should bias toward these.
    dominant_actions: list = field(default_factory=list)        # [a, ...] desc by effect

    # Free-form notes (e.g. "5 slots at cols 15,22,29,36,43")
    notes: list = field(default_factory=list)

    # Recorded elapsed time for this probe
    profile_elapsed: float = 0.0

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)

    @classmethod
    def from_json(cls, s: str) -> "GameProfile":
        d = json.loads(s)
        return cls(**d)
