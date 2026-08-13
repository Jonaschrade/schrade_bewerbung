"""
Network simulation entry point.

Asymmetric interaction model (Jacob & Banisch 2023) on a single Stochastic Block
Model (SBM) graph, with SFT opinion dynamics (Banisch & Olbrich 2019): each agent
holds Q-values over stances (+1 / −1), draws a softmax stance (inverse temperature
β), and TD-updates that Q-value each interaction. Two stance quantities are
tracked per interaction (argmax and softmax); what each draft expressed is
recorded per turn in the expression gate's record (Agent.respond, Step B). Full
mechanism in the README.

Interaction rule (asymmetric)
------------------------------
  1. Draw one expresser uniformly from agents with at least one neighbour.
  2. Draw one responder from its neighbours, weighted by the expresser's expected
     reward per neighbour (β_sel = 0 recovers uniform).
  3. Draw the expresser's softmax stance by softmax(β).
  4. One exchange: expresser speaks, responder reacts (both turns gated).
  5. Reward r = graded(expresser's stance) × graded(reaction) from the gate records.
  6-7. Reward consequences (expresser only): Q-update, edge reward record, and
     (GRAPH_DYNAMIC) edge-strength update all fire iff the round produced a usable
     signal (clear stance AND non-None reward), and freeze together on a no-signal
     round (abstention or None reward). See README "Two reward signals".

A "round" groups INTERACTIONS_PER_ROUND such events for snapshotting.

Experimental variables
-----------------------
SBM_P_INTER               between-community coupling; sweep for the phase transition
RESPONDER_SELECTION_BETA  partner-selection bias; 0 = uniform, >0 = reward-weighted
OPINION_BETA              softmax inverse temperature; β=0 is 50/50, β→∞ is argmax

All tunable parameters live in ``config.py``.
"""

import os
import random
import time
from collections import deque

import networkx as nx
from langchain_ollama import OllamaLLM

from agents.agent import Agent
from agents.personas import sample_personas
from config import (
    GRAPH_DYNAMIC,
    INTERACTIONS_PER_ROUND,
    LEARNING_RATE,
    LLM_MODEL,
    LLM_NUM_CTX,
    NETWORK_MAX_ROUNDS,
    NUM_AGENTS,
    OLLAMA_HOST,
    OPINION_BETA,
    REFLECT_EVERY,
    RESPONDER_SELECTION_BETA,
    REWARD_WINDOW_M,
    SBM_NUM_COMMUNITIES,
    SBM_P_INTER,
    SBM_P_INTRA,
    STRENGTH_CAP,
    STRENGTH_DELTA,
    STRENGTH_FLOOR,
    TOPIC_LABEL,
    TOPIC_TEXT,
)
from network.discussion import gate_summary, run_discussion
from network.edges import record_reward, update_edge
from network.logger import SimulationLogger
from network.matching import ensure_connectivity, select_responder
from network.opinion import (
    compute_polarization_metrics,
    draw_softmax_stance,
    init_opinion_states,
    opinion_states_to_dict,
    update_q_value,
)
from network.state import EdgeData, NetworkState


def _distribute_sizes(n: int, k: int) -> list[int]:
    """Divide n agents as evenly as possible into k communities."""
    base, remainder = divmod(n, k)
    return [base + (1 if i < remainder else 0) for i in range(k)]


def _build_initial_graph(agent_names: list[str]) -> nx.Graph:
    """Create a Stochastic Block Model graph over the agent name list.

    Nodes are relabelled from integers to agent names.  Community membership
    is stored as a node attribute ``"community"`` for post-hoc analysis.
    Every edge is initialised with a fresh ``EdgeData`` instance at strength 1.0.

    ``SBM_P_INTER`` is the primary experimental variable: sweep it to reproduce
    SFT's polarization-to-consensus phase transition.
    """
    n = len(agent_names)
    sizes = _distribute_sizes(n, SBM_NUM_COMMUNITIES)
    p_matrix = [
        [SBM_P_INTRA if i == j else SBM_P_INTER for j in range(SBM_NUM_COMMUNITIES)]
        for i in range(SBM_NUM_COMMUNITIES)
    ]

    G_int = nx.stochastic_block_model(sizes, p_matrix)
    G = nx.relabel_nodes(G_int, {i: agent_names[i] for i in range(n)})

    offset = 0
    for comm_idx, size in enumerate(sizes):
        for i in range(offset, offset + size):
            G.nodes[agent_names[i]]["community"] = comm_idx
        offset += size

    for u, v in G.edges():
        G[u][v]["data"] = EdgeData(
            strengths={u: 1.0, v: 1.0},
            reward_history={
                u: deque(maxlen= REWARD_WINDOW_M),
                v: deque(maxlen=REWARD_WINDOW_M),
            },
        )

    return G


def _count_edge_types(G: nx.Graph) -> tuple[int, int]:
    """Return (n_intra, n_inter) based on node 'community' attribute."""
    n_intra = sum(
        1 for u, v in G.edges()
        if G.nodes[u].get("community") == G.nodes[v].get("community")
    )
    return n_intra, G.number_of_edges() - n_intra


def main() -> None:
    """Run the full network simulation."""
    llm = OllamaLLM(model=LLM_MODEL, base_url=f"http://{OLLAMA_HOST}", reasoning=False, num_ctx=LLM_NUM_CTX)
    logger = SimulationLogger(run_id=os.getenv("SIM_RUN_ID"))

    # -- Agent initialisation ---------------------------------------------
    print(f"\nSampling {NUM_AGENTS} personas...")
    personas = sample_personas(NUM_AGENTS, llm)
    agents: dict[str, Agent] = {
        p["name"]: Agent(name=p["name"], persona=p["persona"], llm=llm)
        for p in personas
    }

    print(f"\n{'━' * 60}")
    print("Participants")
    for name, agent in agents.items():
        print(f"\n  {name}: {agent.persona}")
        print(f"      Pro-Stance-Bridge:    {agent._stance_bridges[1]}")
        print(f"      Contra-Stance-Bridge: {agent._stance_bridges[-1]}")
    print(f"\n{'━' * 60}\n")

    logger.log_personas(agents)

    # -- Network initialisation -------------------------------------------
    G = _build_initial_graph(list(agents.keys()))
    state = NetworkState(agents=agents, graph=G, max_rounds=NETWORK_MAX_ROUNDS)
    state.opinion_states = init_opinion_states(list(agents.keys()))

    logger.snapshot_network(state)   # round-0 baseline

    community_sizes = _distribute_sizes(NUM_AGENTS, SBM_NUM_COMMUNITIES)
    print(
        f"Initial graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges  "
        f"(SBM: {SBM_NUM_COMMUNITIES} communities {community_sizes}, "
        f"p_intra={SBM_P_INTRA}, p_inter={SBM_P_INTER})\n"
        f"α={LEARNING_RATE}  β={OPINION_BETA}  β_sel={RESPONDER_SELECTION_BETA}  "
        f"interactions/round={INTERACTIONS_PER_ROUND}  dynamic={GRAPH_DYNAMIC}\n"
    )

    # last message each agent sent to a specific partner, keyed by (speaker, listener)
    last_message_to: dict[tuple[str, str], str] = {}

    # -- Main simulation loop ---------------------------------------------
    for round_n in range(1, NETWORK_MAX_ROUNDS + 1):
        state.round = round_n
        round_start = time.time()

        n_pos = sum(1 for s in state.opinion_states.values() if s.q_pos > s.q_neg)
        n_neg = sum(1 for s in state.opinion_states.values() if s.q_neg > s.q_pos)
        n_intra, n_inter = _count_edge_types(state.graph)

        print(f"\n{'━' * 60}")
        print(
            f"Round {round_n} / {NETWORK_MAX_ROUNDS}  "
            f"│  topic: {TOPIC_LABEL}  "
            f"│  edges: {state.graph.number_of_edges()} (intra: {n_intra}, inter: {n_inter})  "
            f"│  opinions: +{n_pos} / −{n_neg}  "
            f"│  elapsed: {logger.elapsed():.0f}s"
        )
        print(f"{'━' * 60}")

        # -- INTERACTIONS_PER_ROUND asymmetric interactions ---------------
        softmax_stances: dict[str, int] = {}    # last softmax stance per agent this round
        for interaction_i in range(INTERACTIONS_PER_ROUND):

            # 1. Draw expresser uniformly from agents with at least one neighbour
            eligible = [n for n in agents if state.graph.degree(n) > 0]
            if not eligible:
                break
            expresser_name = random.choice(eligible)

            # 2. Draw responder weighted by expresser's expected reward
            neighbours = list(state.graph.neighbors(expresser_name))
            responder_name = select_responder(
                expresser_name, neighbours, state.graph, RESPONDER_SELECTION_BETA
            )

            # 3. Draw softmax stances via softmax(β)
            softmax_a = draw_softmax_stance(state.opinion_states[expresser_name], OPINION_BETA)
            softmax_b = draw_softmax_stance(state.opinion_states[responder_name], OPINION_BETA)

            print(f"\n  [{interaction_i + 1}/{INTERACTIONS_PER_ROUND}]  "
                  f"{expresser_name} → {responder_name}  "
                  f"(softmax stances: {softmax_a:+d} / {softmax_b:+d})")

            # 4. One exchange: expresser speaks, responder reacts
            result = run_discussion(
                agents[expresser_name],
                agents[responder_name],
                TOPIC_TEXT,
                topic_label=TOPIC_LABEL,
                turns_per_agent=1,
                softmax_a=softmax_a,
                softmax_b=softmax_b,
                prior_b_message=last_message_to.get((responder_name, expresser_name)),
            )

            # record each agent's last utterance so future exchanges can continue from it
            for turn in result["turns"]:
                other = responder_name if turn["speaker"] == expresser_name else expresser_name
                last_message_to[(turn["speaker"], other)] = turn["content"]
            result["argmax_a"] = state.opinion_states[expresser_name].argmax_stance
            result["argmax_b"] = state.opinion_states[responder_name].argmax_stance
            logger.log_discussion(round_n, expresser_name, responder_name, result)

            # 5-7. Reward consequences (expresser only): Q-update, edge reward
            #    record, and edge-strength dynamics all fire iff the round
            #    produced a usable signal (clear stance AND measured reward).
            #    Two no-signal cases freeze all three, logged distinctly but
            #    with identical effect:
            #      - abstention: ambiguous expression (gate_final_expressed_a == 0);
            #      - unparseable reward (reward_a is None): fabricating r = 0 would
            #        falsely decay Q and drift the edge. See README "Two reward signals".
            expressed_a = result["gate_final_expressed_a"]
            abstained = expressed_a == 0
            apply_reward = not abstained and result["reward_a"] is not None
            if apply_reward:
                update_q_value(
                    state.opinion_states[expresser_name],
                    expressed_a,
                    result["reward_a"],
                    LEARNING_RATE,
                )
            elif abstained:
                print(f"    ⚠️  {expresser_name} abstained (ambiguous expression): Q-update, reward record & edge dynamics frozen this round")
            else:
                print(f"    ⚠️  {expresser_name}: reward unparseable. Q-update, reward record & edge dynamics frozen this round")
            softmax_stances[expresser_name] = softmax_a

            q = state.opinion_states[expresser_name]
            idx_note = f"Q-update index→{expressed_a:+d}" if apply_reward else "reward consequences frozen"
            print(
                f"    {expresser_name}: argmax→{result['argmax_a']:+d}  "
                f"softmax→{softmax_a:+d}  "
                f"{gate_summary(result['gate_attempts_a'], result['gate_fallback_a'])}"
            )
            print(
                f"    {responder_name}: argmax→{result['argmax_b']:+d}  "
                f"softmax→{softmax_b:+d}  "
                f"{gate_summary(result['gate_attempts_b'], result['gate_fallback_b'])}"
            )
            print(
                f"    Updated Q-gap {expresser_name}: {q.q_gap:+.3f} ({idx_note})"
            )

            # Reward record + edge dynamics: only on a measured-reward round.
            if apply_reward:
                # Record reward on this edge; feeds future responder draws.
                record_reward(
                    state,
                    expresser_name,
                    responder_name,
                    reward_a=result["reward_a"],
                    # reward_b=result["reward_b"],  # enable for symmetric mode
                )

                # Edge dynamics (GRAPH_DYNAMIC only)
                if GRAPH_DYNAMIC and state.graph.has_edge(expresser_name, responder_name):
                    survived = update_edge(state, expresser_name, responder_name)
                    event_type = "edge_maintained" if survived else "edge_dropped"
                    logger.log_edge_event(round_n, event_type, expresser_name, responder_name)

        # -- Reconnect isolated agents (GRAPH_DYNAMIC only) ---------------
        if GRAPH_DYNAMIC:
            ensure_connectivity(state)

        # -- Reflection phase ---------------------------------------------
        if round_n % REFLECT_EVERY == 0:
            print(f"\n-- Reflection phase (round {round_n}) --")
            for agent in agents.values():
                agent.reflect(state.opinion_states[agent.name].argmax_stance)
                logger.log_reflection(round_n, agent.name)

        # -- Snapshot -----------------------------------------------------
        print(f"\n  Round {round_n} completed in {time.time() - round_start:.1f}s  "
              f"│  total elapsed: {logger.elapsed():.0f}s")
        pol_metrics = compute_polarization_metrics(state.opinion_states)
        logger.snapshot_network(
            state,
            extra_metrics=pol_metrics,
            opinion_states=opinion_states_to_dict(state.opinion_states, softmax_stances),
        )

    # -- Summary ----------------------------------------------------------
    print(f"\n\n{'━' * 60}")
    print("Simulation complete")
    print(f"  Rounds run    : {NETWORK_MAX_ROUNDS}")
    print(f"  Interactions  : {NETWORK_MAX_ROUNDS * INTERACTIONS_PER_ROUND}")
    print(f"  Final edges   : {state.graph.number_of_edges()}")
    print(f"  Components    : {nx.number_connected_components(state.graph)}")

    final_pol = compute_polarization_metrics(state.opinion_states)
    print(f"  Opinions +1   : {final_pol.get('n_pos', '?')}  /  "
          f"Opinions −1: {final_pol.get('n_neg', '?')}")
    print(f"  Dispersion    : {final_pol.get('dispersion', '?')}")
    print(f"  Mean |Q-gap|  : {final_pol.get('mean_q_gap', '?')}")
    print(f"  Logs written  : {logger.run_dir}")
    print(f"  Total time    : {logger.elapsed():.1f}s")
    print(f"{'━' * 60}\n")


if __name__ == "__main__":
    main()
