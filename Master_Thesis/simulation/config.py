"""
Global configuration: single source of truth for all tunable parameters.

All constants live here. No magic numbers in source files; import from here.

Terminology
-----------
turn         one respond() call; one agent speaks once
exchange     one full back-and-forth; both agents speak once (2 turns)
interaction  one asymmetric event: one expresser, one responder, one exchange,
             one Q-update for expresser only
round        INTERACTIONS_PER_ROUND interactions; the snapshotting unit

Environment-variable overrides
------------------------------
Primary sweep parameters (SBM_P_INTER, SBM_P_INTRA, RESPONDER_SELECTION_BETA,
OPINION_BETA, GRAPH_DYNAMIC) and the topic wording (TOPIC_LABEL, TOPIC_TEXT)
accept SIM_* overrides (e.g. SIM_SBM_P_INTER=0.3, SIM_GRAPH_DYNAMIC=1), letting
run_sweep.py launch each grid point as an independent subprocess without editing
this file. Defaults hold when unset.

On the cluster the overrides are the *primary* path, not a convenience:
run_stage.sh derives every SIM_* value from its ``STAGE`` argument, so several
stages can sit in the SGE queue at once. This file is read when a job *starts*,
which may be days after it was submitted — the defaults below are therefore
documentation of the last interactive run, not the record of what a queued job
will compute. That record is run_stage.sh's stage table plus the job log, which
echoes the effective values. See RUNBOOK "Batch-Betrieb".
"""

import os

# -- Models -------------------------------------------------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "127.0.0.1:11500")
LLM_MODEL   = "qwen3.5:35b" 
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", 4096))

# -- Memory -------------------------------------------------------------------
REFLECT_EVERY         = 5    # trigger reflection every N simulation rounds
MAX_MEMORIES_SEED     = 15   # recent memories fed into the reflection prompt
MAX_MEMORIES_RETRIEVE = 5    # relevant memories retrieved per agent

MEMORY_PERSIST = False         # persist ChromaDB to disk (True) or keep in-memory (False)
MEMORY_DIR     = "./memory.db" # path used when MEMORY_PERSIST is True

# -- Retrieval scoring weights -------------------------------------------------
#   Composite score =
#   recency·w + importance·w + relevance·w + stance_congruence·w
WEIGHT_RECENCY           = 0.25
WEIGHT_IMPORTANCE        = 0.25
WEIGHT_RELEVANCE         = 0.3
# Motivated-recall knob: biases retrieval toward memories tagged with the
# stance being expressed (see README "Memory system").
WEIGHT_STANCE_CONGRUENCE = 0.2

# -- Social Feedback Theory (SFT): Q-value learning ---------------------------
# TD update from Banisch & Olbrich (2019): Q(o_i) ← (1 − α) · Q(o_i) + α · r
LEARNING_RATE = 0.1  # α: step size for the Q-value TD update

# SFT expression gate. Rejection-sampling enforcement of the softmax-drawn
# stance at expression time (see README "SFT expression gate"). False = single
# ungated draft (Step A soft anchoring), the ablation arm.
SFT_GATE_ENABLED      = True
SFT_GATE_MAX_ATTEMPTS = 5   # attempt budget per cascade stage (soft prompt / context-free fallback)

# Softmax inverse temperature β for opinion expression.
# p(+1) = exp(β·q_pos) / (exp(β·q_pos) + exp(β·q_neg))
# β = 0  → 50/50 random
# β > 0  → preference for the higher Q-value
# β → ∞  → deterministic argmax
OPINION_BETA = float(os.getenv("SIM_OPINION_BETA", "5.0"))

# Responder-selection inverse temperature β_sel.
# Weights the responder draw by exp(β_sel · expected_reward(neighbour)).
# expected_reward is the mean of the expresser's own reward history on that
# edge (EdgeData.reward_history[expresser]); the expresser prefers partners
# who have rewarded it well.
# β_sel = 0  → uniform draw over neighbours (replicates Banisch & Olbrich 2019)
# β_sel > 0  → neighbours with higher past reward preferred
# No history yet contributes neutral expected_reward 0.0 (cold start).
RESPONDER_SELECTION_BETA = float(os.getenv("SIM_RESPONDER_SELECTION_BETA", "0.0"))

# -- Network simulation --------------------------------------------------------
NUM_AGENTS         = 20  # total agents in the network graph (pairwise mode always uses 2)
NETWORK_MAX_ROUNDS = 25  # total snapshot rounds

# Interactions per snapshot round; network mode only. Pairwise mode fixes its
# own local value of 2 in main_pairwise.py and ignores this constant.
# One interaction = one asymmetric event (expresser drawn uniformly, responder
# drawn with reward-based selection bias β_sel, expresser's Q updated).
# Default NUM_AGENTS gives each agent one expected interaction per round
# (random sequential update convention).
INTERACTIONS_PER_ROUND = NUM_AGENTS

# Graph topology: Stochastic Block Model (SBM). Inter-community coupling
# SBM_P_INTER is the primary experimental variable (low → stable polarization,
# high → consensus), reproducing the Banisch & Olbrich (2019) phase transition.
SBM_NUM_COMMUNITIES = 2    # number of opinion communities (blocks)
SBM_P_INTRA         = float(os.getenv("SIM_SBM_P_INTRA", "0.7"))  # within-community edge probability
SBM_P_INTER         = float(os.getenv("SIM_SBM_P_INTER", "0.15"))  # between-community edge probability (sweep this for phase transition)

# Graph dynamics: set False to hold the graph fixed (main SFT experiments).
# Set True to enable endogenous tie rewiring via the edge-valuation mechanism
# (extension chapter: homophilic tie formation).
GRAPH_DYNAMIC = os.getenv("SIM_GRAPH_DYNAMIC", "0") == "1"

# Edge dynamics: only active when GRAPH_DYNAMIC = True.
STRENGTH_CAP    = 3.0  # ceiling on each agent's internal edge valuation
STRENGTH_FLOOR  = 0.0  # edge removed when either agent's valuation falls to or below this
STRENGTH_DELTA  = 0.5  # valuation change per interaction = mean_reward × STRENGTH_DELTA
REWARD_WINDOW_M = 5    # rolling window length (interactions) for reward-history edge evaluation

# -- Discussion topic ---------------------------------------------------------
# One fixed topic for all rounds, required for Q-value coherence: the +1/−1
# stance dimension must refer to the same opinion object throughout (see README
# "Configuration" and PLAN_banisch_opinion.md).
TOPIC_LABEL = os.getenv("SIM_TOPIC_LABEL", "Energy policy")
# Wording rationale in topic_candidates.md. Whether the wording carries a
# directional prior is measured, not assumed: re-run the Stage 0b audit + β=0
# harvest (test_schedule.md) under this topic.
# The override exists for Stage 5 (stance-flip control), which reruns key points
# under the mirrored, permissive framing without touching this default.
TOPIC_TEXT  = os.getenv(
    "SIM_TOPIC_TEXT",
    "The United States should significantly expand nuclear power to meet its electricity demand",
)
