"""
Agent partner selection and network reconnection for the network simulation.

``select_responder`` implements the asymmetric rule's responder draw: given an
expresser and its neighbours, it draws a responder weighted by the expresser's
own past reward from each neighbour (per ``EdgeData.reward_history``, populated
by ``network.edges.record_reward``). β_sel = 0 gives a uniform draw (Banisch &
Olbrich 2019); β_sel > 0 prefers neighbours with higher expected reward.

``compute_pairings`` and ``ensure_connectivity`` are retained for the
``GRAPH_DYNAMIC = True`` extension (endogenous tie rewiring).
"""

from __future__ import annotations

import math
import random
from collections import deque

import networkx as nx

from config import REWARD_WINDOW_M
from network.state import EdgeData, NetworkState


def select_responder(
    expresser: str,
    neighbours: list[str],
    graph: nx.Graph,
    beta: float,
) -> str:
    """Draw a responder from the expresser's neighbours by expected reward.

    Called once per interaction in the main loops. Each neighbour's expected
    reward is the mean of the expresser's own reward history on that edge
    (``EdgeData.reward_history[expresser]``, populated by ``record_reward`` after
    every measured-reward interaction; no-signal rounds record nothing).
    Neighbours with no history yet contribute a neutral 0.0 (cold start). Kept
    standalone so the virtual-worlds multi-platform extension can swap in a
    different draw without touching the interaction loop.

    Weight formula:  w_j = exp(β_sel · expected_reward(expresser, j))

    Parameters
    ----------
    expresser:
        Name of the agent expressing an opinion this interaction.
    neighbours:
        Adjacency list of the expresser (already filtered to non-empty).
    graph:
        The network graph; edges carry ``EdgeData`` under ``"data"``.
    beta:
        Responder-selection inverse temperature β_sel ≥ 0.
        β_sel = 0 → uniform draw (Banisch & Olbrich 2019).
        β_sel > 0 → neighbours with higher expected reward weighted higher.

    Returns
    -------
    str
        Name of the selected responder.
    """
    if beta == 0.0:
        return random.choice(neighbours)

    def expected_reward(neighbour: str) -> float:
        history = graph[expresser][neighbour]["data"].reward_history[expresser]
        return sum(history) / len(history) if history else 0.0

    weights = [math.exp(beta * expected_reward(n)) for n in neighbours]
    total = sum(weights)
    r = random.random() * total
    cumulative = 0.0
    for name, w in zip(neighbours, weights):
        cumulative += w
        if r <= cumulative:
            return name
    return neighbours[-1]


def compute_pairings(state: NetworkState) -> list[tuple[str, str]]:
    """Compute agent pairings for the current round.

    Not called in the default asymmetric mode. Reserved for the
    ``GRAPH_DYNAMIC = True`` extension if a symmetric global-matching round
    structure is needed alongside endogenous tie rewiring.

    Agents are matched over existing edges by max-weight matching (Edmonds'
    blossom); unmatched agents pause the round with edge strengths and memories
    unchanged. If the agent count is odd, one agent is rotated out each round and
    recorded in ``state.idle_agent``, cycling via ``state.round % len(agents)`` so
    no agent is systematically excluded.

    Parameters
    ----------
    state:
        The current ``NetworkState``. ``state.idle_agent`` may be mutated.

    Returns
    -------
    list[tuple[str, str]]
        ``(agent_a_name, agent_b_name)`` pairs holding a discussion this round.
        Unmatched agents are omitted.
    """
    # Use a sorted list for deterministic sit-out rotation
    all_agents = sorted(state.agents.keys())

    if len(all_agents) % 2 == 1:
        sit_out_idx = state.round % len(all_agents)
        state.idle_agent = all_agents[sit_out_idx]
        active = [a for a in all_agents if a != state.idle_agent]
    else:
        state.idle_agent = None
        active = all_agents

    # Build a helper graph with 'weight' = edge strength for the matching algo
    H = nx.Graph()
    H.add_nodes_from(active)
    subgraph = state.graph.subgraph(active)
    for u, v in subgraph.edges():
        H.add_edge(u, v, weight=sum(state.graph[u][v]["data"].strengths.values()))

    matched = nx.max_weight_matching(H, maxcardinality=True)
    return [tuple(pair) for pair in matched]


def reconnect_isolated(state: NetworkState, agent_name: str) -> None:
    """Connect a degree-zero agent to a uniformly random other agent.

    The new edge is created at ``strength=0.5`` (neutral introductory level)
    so it does not immediately dominate the matching weight of established
    relationships.

    Parameters
    ----------
    state:
        The current ``NetworkState``.  ``state.graph`` is mutated in place.
    agent_name:
        Name of the agent to reconnect.
    """
    candidates = [a for a in state.agents if a != agent_name]

    if not candidates:
        return

    partner = random.choice(candidates)
    state.graph.add_edge(agent_name, partner, data=EdgeData(
        strengths={agent_name: 0.5, partner: 0.5},
        reward_history={
            agent_name: deque(maxlen=REWARD_WINDOW_M),
            partner:     deque(maxlen=REWARD_WINDOW_M),
        },
    ))


def ensure_connectivity(state: NetworkState) -> None:
    """Reconnect any agent whose degree has dropped to zero.

    Called once per round after all edge updates.  Iterates over every
    agent and invokes ``reconnect_isolated`` for those with no remaining
    edges, ensuring no agent is permanently excluded from future rounds.

    Parameters
    ----------
    state:
        The current ``NetworkState``.  ``state.graph`` may be mutated.
    """
    for name in list(state.agents.keys()):
        if state.graph.degree(name) == 0:
            reconnect_isolated(state, name)
