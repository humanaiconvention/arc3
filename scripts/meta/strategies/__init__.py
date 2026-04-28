"""Strategy registry — central list of available strategies.

Each strategy is a subclass of Strategy (base.py). To register a new one:
  1. Create meta/strategies/my_thing.py with class MyThing(Strategy).
  2. Import it here and add to STRATEGIES.
"""
from meta.strategies.base import Strategy
from meta.strategies.action_spam import ActionSpam
from meta.strategies.cluster_click_then_nav import ClusterClickThenNav
from meta.strategies.combo_lock import ComboLock
from meta.strategies.cursor_walk import CursorWalk
from meta.strategies.grid_click import GridClick
from meta.strategies.grid_click_fine import GridClickFine
from meta.strategies.inverse_aware_walk import InverseAwareWalk
from meta.strategies.mimic_target import MimicTarget
from meta.strategies.mover_toggle_walk import MoverToggleWalk
from meta.strategies.nav_and_click import NavAndClick
from meta.strategies.policy_guided_walk import PolicyGuidedWalk
from meta.strategies.policy_guided_walk_cnn import PolicyGuidedWalkCNN
from meta.strategies.random_walk import RandomWalk
from meta.strategies.replay_known import ReplayKnown
from meta.strategies.sequence_search import SequenceSearch
from meta.strategies.target_state_match import TargetStateMatch

# Order here doesn't matter; the orchestrator ranks by confidence.
STRATEGIES: list = [
    ActionSpam,
    ClusterClickThenNav,
    ComboLock,
    CursorWalk,
    GridClick,
    GridClickFine,
    InverseAwareWalk,
    MimicTarget,
    MoverToggleWalk,
    NavAndClick,
    PolicyGuidedWalk,
    PolicyGuidedWalkCNN,
    RandomWalk,
    ReplayKnown,
    SequenceSearch,
    TargetStateMatch,
]
