"""
Agent opinion state and Q-value dynamics for Social Feedback Theory (SFT).

Core reinforcement-learning mechanism from Banisch & Olbrich (2019): each agent
holds Q-values over the two stances (+1 / −1), draws a stance by softmax (inverse
temperature β), and updates that Q-value via a TD rule after each interaction.

Public API
----------
AgentOpinionState             dataclass: Q-values + argmax_stance + q_gap
init_opinion_states()         initialise one state per agent
draw_softmax_stance()         stochastic stance draw (β-parameterised)
update_q_value()              apply TD update for a given stance
opinion_states_to_dict()      serialise for logging (argmax + optional softmax)
compute_polarization_metrics()  population-level SFT metrics

Two stance quantities exist per agent per interaction: the argmax stance
(deterministic argmax over Q-values; metrics, logging, console) and the softmax
stance (drawn by draw_softmax_stance() for a specific interaction, anchoring
respond()). What each draft actually expressed lives in the gate records; the
main loops index the TD update on final_expressed. See README "Terminology".
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class AgentOpinionState:
    """SFT internal state: Q-values over the two opinion stances.

    Attributes
    ----------
    q_pos:
        Q(+1): perceived social value of expressing the pro-stance.
        Updated whenever the agent expresses +1 and receives feedback.
    q_neg:
        Q(−1): perceived social value of expressing the contra-stance.
        Updated whenever the agent expresses −1 and receives feedback.

    Both values start at 0.0 so neither stance is initially preferred,
    and the social environment drives divergence over rounds.
    """

    q_pos: float = 0.0
    q_neg: float = 0.0

    @property
    def argmax_stance(self) -> int:
        """Deterministic argmax over Q-values; ties resolve to +1.

        The indicator used for metrics and logging: the stance the agent would
        prefer, not necessarily the one drawn (draw_softmax_stance, which can
        differ at low conviction) or expressed in a given interaction.
        """
        return 1 if self.q_pos >= self.q_neg else -1

    @property
    def q_gap(self) -> float:
        """Signed confidence: positive = leaning pro, negative = leaning contra."""
        return self.q_pos - self.q_neg


def init_opinion_states(agent_names: list[str]) -> dict[str, AgentOpinionState]:
    """Initialise one neutral AgentOpinionState for each agent."""
    return {name: AgentOpinionState() for name in agent_names}


def update_q_value(
    opinion: AgentOpinionState,
    stance: int,
    reward: float,
    alpha: float,
) -> None:
    """Temporal-difference update for the Q-value of ``stance``.

    SFT rule (Banisch & Olbrich 2019): Q(o_i) ← (1 − α) · Q(o_i) + α · r. Only
    ``stance``'s Q-value is updated; the other receives no feedback. ``stance``
    must be the stance that entered the interaction (the reward responds to what
    was said): callers pass the expresser's ``gate_final_expressed`` and call
    this only when it is non-zero (an ambiguous expression is a strategic
    abstention, skipped upstream). Under the gate that stance is the softmax
    draw; in the ungated arm it is the classified expressed stance.

    Parameters
    ----------
    opinion:
        The agent's current opinion state. Mutated in place.
    stance:
        Stance (+1 or −1) whose Q-value receives the update.
    reward:
        Social feedback scalar r ∈ [−1.0, 1.0]. Positive = agreement.
    alpha:
        Learning rate α (LEARNING_RATE in config.py).
    """
    if stance == 1:
        opinion.q_pos = (1 - alpha) * opinion.q_pos + alpha * reward
    else:
        opinion.q_neg = (1 - alpha) * opinion.q_neg + alpha * reward


def draw_softmax_stance(opinion: AgentOpinionState, beta: float) -> int:
    """Draw a softmax stance via softmax with inverse temperature β.

    Uses the logistic (sigmoid) form, which is equivalent to a two-class
    softmax and numerically stable:

        p(+1) = σ(β · (q_pos − q_neg)) = 1 / (1 + exp(−β · Δq))

    β = 0  → p(+1) = 0.5  (uniform random, fixes tied-Q init artifact)
    β > 0  → p(+1) > 0.5 when q_pos > q_neg
    β → ∞  → deterministic argmax

    Parameters
    ----------
    opinion:
        Current Q-value state.
    beta:
        Inverse temperature β ≥ 0.

    Returns
    -------
    int
        +1 or −1.
    """
    import random
    p_pos = 1.0 / (1.0 + math.exp(-beta * opinion.q_gap))
    return 1 if random.random() < p_pos else -1


def opinion_states_to_dict(
    opinion_states: dict[str, AgentOpinionState],
    softmax_stances: dict[str, int] | None = None,
) -> dict[str, dict]:
    """Serialise opinion states for JSON logging.

    Parameters
    ----------
    opinion_states:
        Mapping of agent name to AgentOpinionState.
    softmax_stances:
        Optional mapping of agent name to the softmax-drawn stance for the
        current round.  When provided, each agent entry gains a
        ``"softmax"`` key with the interaction-level draw alongside the
        deterministic ``"argmax"`` indicator.  Agents absent from this dict
        (e.g. never selected as expresser in the round) receive no
        ``"softmax"`` key. Per-draft expressed stances are not snapshotted here;
        they live in the gate records logged with each turn (network/discussion.py).
    """
    result = {}
    for name, s in opinion_states.items():
        entry: dict = {
            "q_pos":  round(s.q_pos, 4),
            "q_neg":  round(s.q_neg, 4),
            "argmax": s.argmax_stance,
        }
        if softmax_stances is not None and name in softmax_stances:
            entry["softmax"] = softmax_stances[name]
        entry["q_gap"] = round(s.q_gap, 4)
        result[name] = entry
    return result


def compute_polarization_metrics(
    opinion_states: dict[str, AgentOpinionState],
) -> dict:
    """Population-level polarization metrics aligned with Banisch & Olbrich (2019).

    Parameters
    ----------
    opinion_states:
        Mapping of agent name to AgentOpinionState.

    Returns
    -------
    dict with keys:
        n_pos : int
            Agents with a strict positive Q-gap (q_pos > q_neg).
        n_neg : int
            Agents with a strict negative Q-gap (q_neg > q_pos).
            Agents with equal Q-values (neutral/tied) are excluded from both
            counts; n_pos + n_neg ≤ total agents.
        dispersion : float
            Variance of argmax stances in {−1, +1}. 0 (full consensus) to 1
            (maximally split population).
        mean_q_gap : float
            Mean |Q(+1) − Q(−1)| across agents: average certainty of the argmax
            stance. High values indicate committed opinions.
    """
    if not opinion_states:
        return {}

    stances = [s.argmax_stance for s in opinion_states.values()]
    n = len(stances)
    n_pos = sum(1 for s in opinion_states.values() if s.q_pos > s.q_neg)
    n_neg = sum(1 for s in opinion_states.values() if s.q_neg > s.q_pos)
    mean_stance = sum(stances) / n
    dispersion = sum((o - mean_stance) ** 2 for o in stances) / n
    mean_q_gap = sum(abs(s.q_gap) for s in opinion_states.values()) / n

    return {
        "n_pos":      n_pos,
        "n_neg":      n_neg,
        "dispersion": round(dispersion, 4),
        "mean_q_gap": round(mean_q_gap, 4),
    }
