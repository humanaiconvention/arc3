"""grid_click_fine — grid_click with 4-cell grid (256 positions).

For games that need clicks at finer coordinates than 8-stride grid catches,
like lp85 whose live positions are at (48,26), (48,37), (20,17) etc — not
aligned to an 8-stride grid with offset 4.

Tradeoff: 4x more probes, so phase 1 scan costs ~4x more time. Budget
accordingly or give this strategy lower confidence than coarse grid_click.
"""
from __future__ import annotations

import numpy as np

from meta.budget import Budget
from meta.common import GameSession
from meta.profile import GameProfile
from meta.result import StrategyResult
from meta.strategies.grid_click import GridClick


class GridClickFine(GridClick):
    name = "grid_click_fine"

    GRID_STEP = 4  # 16x16 = 256 probe points on a 64x64 frame

    def confidence(self, profile: GameProfile) -> float:
        # Slightly lower than the coarse grid_click by default, because it's
        # more expensive. But we still want it to run for click-family games
        # when coarse missed.
        coarse = super().confidence(profile)
        return max(0.0, coarse * 0.8)
