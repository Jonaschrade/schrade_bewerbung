"""
Pairwise discussion runner for the network simulation.

Runs one asymmetric discussion: alternating gated LLM turns (Agent.respond,
Step B), then the reward product. Every turn enters the conversation on-stance or
ambiguous, never the opposite stance, and carries a gate record (draft texts,
per-draft expressed stance and graded value, cascade stage). The reward for each
speaker is the product of two graded expressions already classified in those gate
records, r = graded(expresser) * graded(reaction) (agent.reward_from_expressions);
no extra LLM call is made here. Gating the responder matters as much as the
expresser: the reaction's graded stance should come from a stance-coherent partner.

reward_a drives the expresser's Q-update (indexed on gate_final_expressed_a)
and, when GRAPH_DYNAMIC, the edge valuation; reward_b is computed but unused
(the update is asymmetric by design). Both consequences fire only on a
measured-reward round and freeze together on a no-signal round (abstention or
None reward). See README "SFT expression gate" and "Two reward signals".

On the first meeting the expresser responds to the moderator's topic; on repeat
meetings the caller passes prior_b_message so the exchange continues the
dialogue. The topic is always forwarded via the stance hint. Reward stays
causally separate from generation: each graded factor is an independent,
context-free classification of a single turn (Chuang et al. 2024).
"""

from __future__ import annotations

from agents.agent import Agent, reward_from_expressions


def gate_summary(attempts: int, fallback: bool) -> str:
    """One-line console summary of a turn's expression-gate outcome.

    Shared by both entry points so the gate line reads identically in
    network and pairwise mode.  ``attempts`` is the total draft count for
    the turn (1 = first draft passed); ``fallback`` flags whether the
    cascade left the soft-prompt stage.
    """
    return f"gate: {attempts} attempt(s){', fallback' if fallback else ''}"


def run_discussion(
    agent_a: Agent,
    agent_b: Agent,
    topic: str,
    topic_label: str = "",
    turns_per_agent: int = 1,
    verbose: bool = True,
    *,
    softmax_a: int,
    softmax_b: int,
    prior_b_message: str | None = None,
) -> dict:
    """Run a pairwise discussion between two agents and collect reward signals.

    Begins with a moderator message containing ``topic``; agents then alternate
    for ``turns_per_agent × 2`` gated turns, each anchored to the agent's softmax
    stance (``softmax_a`` / ``softmax_b``). When ``prior_b_message`` is given it
    is appended after the moderator opening, so agent_a's first turn continues
    from agent_b's last message instead of reacting to the moderator. After the
    last turn each speaker's reward is formed as the product of its own first
    turn's graded stance and the partner's last turn's graded stance, both read
    off the gate records (no additional LLM call).

    Parameters
    ----------
    agent_a, agent_b:
        Participants. agent_a speaks first (turns 0, 2, …), agent_b second.
    topic:
        Opening moderator message seeding the conversation.
    topic_label:
        Short identifier for the topic.
    turns_per_agent:
        Turns each agent takes. Default 1 (one asymmetric exchange).
    verbose:
        When True, each reply is printed to stdout.
    softmax_a, softmax_b:
        Softmax-drawn stances (+1 or −1) from draw_softmax_stance(); anchor the
        prompts and arbitrate the gate. Required.
    prior_b_message:
        agent_b's last message in a previous exchange, or None on first meeting.

    Returns
    -------
    dict with keys:
        ``turns`` : list of {"speaker", "content", "gate"} dicts (excl.
            moderator); each ``gate`` is the turn's full gate record.
        ``reward_a`` / ``reward_b`` : float in [−1, 1] or None. reward_a is
            graded(agent_a's first) × graded(agent_b's last reaction); reward_b is
            None unless turns_per_agent >= 2 (with one turn each, agent_a speaks
            before agent_b, so its message cannot react to agent_b). None means the
            reaction was UNRELATED/unparseable: skip the TD update, do not treat it
            as a genuine zero (a genuine ambivalent reaction gives 0.0, not None).
        ``softmax_a`` / ``softmax_b`` : int, echoed for the logger.
        ``gate_attempts_a`` / ``gate_attempts_b`` : int, drafts the first turn needed.
        ``gate_fallback_a`` / ``gate_fallback_b`` : bool, first turn reached the
            fallback cascade (incl. the fixed-template stage 3).
        ``gate_final_expressed_a`` / ``gate_final_expressed_b`` : int, expressed
            stance sign of the accepted first turn; the caller indexes the Q-update
            on it and skips when 0 (abstention).
        ``gate_final_graded_a`` / ``gate_final_graded_b`` : float in [−1, 1], graded
            stance of the accepted first turn (the reward factors); sign equals the
            corresponding gate_final_expressed_*.
        ``topic_label`` : str.
    """
    transcript: list[dict] = [{"speaker": "Moderator", "content": f"{topic}."}]
    if prior_b_message is not None:
        transcript.append({"speaker": agent_b.name, "content": prior_b_message})

    total_turns = turns_per_agent * 2

    # Each agent's first turn provides the gate convenience fields below,
    # captured here so the result dict need not dig through the turn list.
    gate_a_first: dict | None = None
    gate_b_first: dict | None = None

    turns: list[dict] = []
    for i in range(total_turns):
        if i % 2 == 0:
            speaker = agent_a
            softmax_stance = softmax_a
        else:
            speaker = agent_b
            softmax_stance = softmax_b

        last = transcript[-1]
        reply, gate = speaker.respond(
            last["content"], last["speaker"],
            softmax_stance=softmax_stance,
        )
        if i == 0:
            gate_a_first = gate
        elif i == 1:
            gate_b_first = gate
        if verbose:
            print(f"\n{speaker.name}: {reply}")
        transcript.append({"speaker": speaker.name, "content": reply})
        turns.append({"speaker": speaker.name, "content": reply, "gate": gate})

    # Reward as the product of graded expressions r = graded(expresser) *
    # graded(reaction), the o_i * o_j form (see agent.reward_from_expressions):
    #   reward_a: agent_a's first expression vs agent_b's last reaction.
    #             Valid for all turns_per_agent >= 1 (agent_b always speaks after agent_a).
    #   reward_b: agent_b's first expression vs agent_a's last reaction.
    #             Requires turns_per_agent >= 2: agent_a must speak again *after* agent_b
    #             for its last turn to be a genuine reaction. With turns_per_agent=1
    #             agent_a's only message precedes agent_b's, so reward_b is causally
    #             inverted and is set to None instead of being computed and misread as
    #             valid feedback.
    # The graded stance of each turn is already classified in its gate record, so
    # no additional LLM call is made here.
    graded_a_first = gate_a_first["final_expressed_graded"]
    graded_b_last = next(
        t["gate"]["final_expressed_graded"]
        for t in reversed(turns) if t["speaker"] == agent_b.name
    )
    reward_a: float | None = reward_from_expressions(graded_a_first, graded_b_last)

    def _fmt_g(g: float | None) -> str:
        return f"{g:+.2f}" if g is not None else "None"

    def _fmt_reward(r: float | None, g_expr: float | None, g_react: float | None) -> str:
        """Reward with its composition: r (= expr × react)."""
        factors = f"expr {_fmt_g(g_expr)} × resp {_fmt_g(g_react)}"
        if r is None:
            return f"None ({factors}; no signal, TD update skipped)"
        return f"{r:+.2f} (= {factors})"

    if turns_per_agent >= 2:
        graded_b_first = gate_b_first["final_expressed_graded"]
        graded_a_last = next(
            t["gate"]["final_expressed_graded"]
            for t in reversed(turns) if t["speaker"] == agent_a.name
        )
        reward_b: float | None = reward_from_expressions(graded_b_first, graded_a_last)
        print(
            f"\n  🎯 Reward  {agent_a.name}: {_fmt_reward(reward_a, graded_a_first, graded_b_last)}  |  "
            f"{agent_b.name}: {_fmt_reward(reward_b, graded_b_first, graded_a_last)}"
        )
    else:
        reward_b = None
        print(f"\n  🎯 Reward  {agent_a.name}: {_fmt_reward(reward_a, graded_a_first, graded_b_last)}")

    return {
        "topic_label":            topic_label,
        "turns":                  turns,
        "reward_a":               reward_a,
        "reward_b":               reward_b,
        "softmax_a":              softmax_a,
        "softmax_b":              softmax_b,
        "gate_attempts_a":        gate_a_first["n_attempts"],
        "gate_fallback_a":        gate_a_first["fallback_used"],
        "gate_final_expressed_a": gate_a_first["final_expressed"],
        "gate_final_graded_a":    graded_a_first,
        "gate_attempts_b":        gate_b_first["n_attempts"],
        "gate_fallback_b":        gate_b_first["fallback_used"],
        "gate_final_expressed_b": gate_b_first["final_expressed"],
        "gate_final_graded_b":    gate_b_first["final_expressed_graded"],
    }
