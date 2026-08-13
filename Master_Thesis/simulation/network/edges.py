"""
Edge lifecycle management for the network simulation.

``record_reward`` appends the expresser's (and optionally responder's) reward to
the edge's rolling history (window ``REWARD_WINDOW_M``). Called on every
measured-reward interaction in both graph modes, and skipped on a no-signal round
(abstention or unparseable reward), whose reward consequences all freeze. This
history feeds ``select_responder`` (both modes) and ``update_edge`` (edge-strength
dynamics, ``GRAPH_DYNAMIC = True`` only).

``update_edge`` derives a signal from the rolling mean of each agent's history
and steps that agent's edge valuation by STRENGTH_DELTA; the edge is severed once
either agent's valuation falls to or below ``STRENGTH_FLOOR``. Called only when
``GRAPH_DYNAMIC = True`` and, like ``record_reward``, skipped on a no-signal
round. In the default fixed-graph mode the initial structure is preserved for the
full run, but ``record_reward`` still runs so reward-based responder selection
works in both modes.

Asymmetric vs. symmetric mode: ``record_reward`` is called with only ``reward_a``
in asymmetric mode; the responder's history stays empty and contributes a neutral
0.0. Pass ``reward_b`` too to activate symmetric evaluation once the caller wires
it up.

Note: ``ensure_connectivity`` and ``reconnect_isolated`` live in
``network/matching.py`` (reconnection is topologically a pairing operation).
"""

from __future__ import annotations

from config import STRENGTH_CAP, STRENGTH_DELTA, STRENGTH_FLOOR
from network.state import NetworkState


def record_reward(
    state: NetworkState,
    agent_a: str,
    agent_b: str,
    reward_a: float | None = None,
    reward_b: float | None = None,
) -> None:
    """Append interaction reward(s) to the edge's rolling history.

    Called on every measured-reward interaction (independent of ``GRAPH_DYNAMIC``)
    so ``select_responder`` has up-to-date history. The caller does not invoke it
    on a no-signal round (abstention or unparseable reward). ``None`` still no-ops
    defensively (e.g. ``reward_b`` in asymmetric mode): that history is unchanged.

    Parameters
    ----------
    state:
        The current ``NetworkState``. The graph is mutated in place.
    agent_a:
        Name of the expresser (first agent in the pair).
    agent_b:
        Name of the responder (second agent in the pair).
    reward_a:
        Reward for ``agent_a`` this interaction ∈ [−1, 1], or ``None`` to skip.
    reward_b:
        Reward for ``agent_b`` ∈ [−1, 1], or ``None`` to skip (asymmetric default).
    """
    edge = state.graph[agent_a][agent_b]["data"]

    if reward_a is not None:
        edge.reward_history[agent_a].append(reward_a)
    if reward_b is not None:
        edge.reward_history[agent_b].append(reward_b)


def update_edge(
    state: NetworkState,
    agent_a: str,
    agent_b: str,
) -> bool:
    """Derive an edge-valuation signal from existing reward history.

    Reads the mean reward over each agent's rolling window (populated by
    ``record_reward``) and steps each agent's edge valuation. An empty history
    yields a neutral 0.0 (no change). The caller skips this on a no-signal round,
    so such a round never nudges strength or severs the edge.

    Each agent's valuation updates independently:
        strength += mean_reward × STRENGTH_DELTA
    Both clamped to [0, STRENGTH_CAP]. The edge is removed once either agent's
    valuation falls to or below ``STRENGTH_FLOOR`` (one agent's dissatisfaction
    suffices). ``EdgeData.rounds_active`` is incremented when the edge survives.

    Parameters
    ----------
    state:
        The current ``NetworkState``. The graph is mutated in place.
    agent_a:
        Name of the expresser (first agent in the pair).
    agent_b:
        Name of the responder (second agent in the pair).

    Returns
    -------
    bool
        ``True`` if the edge survived, ``False`` if removed.
    """
    edge = state.graph[agent_a][agent_b]["data"]

    def _signal(history) -> float:
        return sum(history) / len(history) if history else 0.0

    signal_a = _signal(edge.reward_history[agent_a])
    signal_b = _signal(edge.reward_history[agent_b])

    edge.strengths[agent_a] = max(0.0, min(STRENGTH_CAP,
        edge.strengths[agent_a] + signal_a * STRENGTH_DELTA))
    edge.strengths[agent_b] = max(0.0, min(STRENGTH_CAP,
        edge.strengths[agent_b] + signal_b * STRENGTH_DELTA))

    if edge.strengths[agent_a] <= STRENGTH_FLOOR or edge.strengths[agent_b] <= STRENGTH_FLOOR:
        state.graph.remove_edge(agent_a, agent_b)
        return False

    edge.rounds_active += 1
    return True
