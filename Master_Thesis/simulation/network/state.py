"""
Data classes for the network simulation state.

NetworkState is the single mutable object passed through every round of the
simulation: agent registry, NetworkX graph, Q-value opinion states, round
bookkeeping. EdgeData is stored as a per-edge attribute under key "data".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx


@dataclass
class EdgeData:
    """Metadata stored on every edge of the communication network.

    Attributes
    ----------
    strengths:
        Per-agent internal valuation of the relationship, keyed by agent name.
        Updated each interaction from the mean of that agent's reward history
        times ``STRENGTH_DELTA``. Edge severed once either agent's value falls
        to or below ``STRENGTH_FLOOR``. Matching weight is the sum of both
        values, so well-established (mutually valued) relationships are
        preferred.
    reward_history:
        Rolling reward window per agent, keyed by agent name. Each value is a
        ``collections.deque`` with ``maxlen=REWARD_WINDOW_M``. Initialised at
        graph construction; updated by ``network/edges.py`` after each
        interaction. In asymmetric mode only the expresser's deque is
        populated; the responder's stays empty, contributing a neutral 0.0.
    rounds_active:
        Counter incremented after every discussion the edge survives. Useful
        for post-hoc analysis of relationship duration.
    """

    strengths: dict = field(default_factory=dict)       # {agent_name: float}
    reward_history: dict = field(default_factory=dict)  # {agent_name: deque[float]}
    rounds_active: int = 0


@dataclass
class NetworkState:
    """Complete mutable state of one network simulation run.

    Attributes
    ----------
    agents:
        Mapping from agent name (str) to the corresponding ``Agent`` instance.
        Treat as read-only after initialisation.
    graph:
        Undirected NetworkX graph whose nodes are agent names and whose edges
        carry an ``EdgeData`` instance under attribute key ``"data"``.
    round:
        Current simulation round, incremented by ``main_network.py`` before
        each round's processing begins.
    max_rounds:
        Total rounds to run. Outer loop exits when ``round >= max_rounds``.
    idle_agent:
        Name of the agent sitting out the current round under odd-count
        symmetric pairing (``compute_pairings()``). Always ``None`` in the
        default asymmetric interaction mode.
    opinion_states:
        Mapping from agent name to ``AgentOpinionState`` (Q-values and argmax
        stance). Populated by ``network/opinion.py`` at simulation start and
        updated after every discussion.
    """

    agents: dict
    graph: nx.Graph
    round: int = 0
    max_rounds: int = 30
    idle_agent: str | None = None
    opinion_states: dict = field(default_factory=dict)
    # maps agent name -> AgentOpinionState (q_pos, q_neg, argmax_stance)
