# Banisch & Olbrich Opinion State, Implementation Notes

This document records the design decisions made when implementing the SFT Q-value layer. It supersedes the original plan, which proposed a simpler wiring of a full-transcript self-scoring rating into a Q-update. The implementation differs in two significant ways: the reward source, and the opinion initialisation strategy.

---

## Theoretical background

Banisch & Olbrich (2019) model opinion dynamics as a Q-learning process. Each agent holds two private Q-values, `q_pos` and `q_neg`, representing how rewarding it has been to express a pro or contra stance in past interactions. The *argmax stance* is `argmax(q_pos, q_neg)`, the deterministic indicator of which stance the agent currently favours; the *softmax stance* in any given interaction is drawn stochastically by softmax with inverse temperature β and may differ from the argmax stance when conviction is low. The *expressed stance* is classified from the agent's generated text and may in turn differ from the softmax stance when persona/memory override the anchor (see decision 7 below). After each interaction the Q-value for the expressed stance is updated toward the social reward received, unless the expression is *ambiguous* (`0`), in which case the update is skipped (a vague utterance is a strategic abstention; see decision 7):

```
Q(o_i) ← (1 − α) · Q(o_i) + α · r
```

where r ∈ {−1, +1} in the original binary formulation.

---

## What was built and where

| Component | File | Notes |
|---|---|---|
| `AgentOpinionState` dataclass | `network/opinion.py` | `q_pos`, `q_neg`; `argmax_stance` property (argmax, stance favoured given current Q-values, not necessarily the stance drawn or expressed in a given interaction); `q_gap` property |
| `init_opinion_states()` | `network/opinion.py` | Initialises all agents at Q = (0, 0); no LLM call at startup |
| `update_q_value(stance, reward, α)` | `network/opinion.py` | TD update for the Q-value of the *expressed stance* (classified from the generated text). Only called for a non-zero stance, an ambiguous expression (`0`) is a strategic abstention and skips the update upstream |
| `draw_softmax_stance(β)` | `network/opinion.py` | Stochastic draw: β=0 → 50/50; β→∞ → argmax; uses logistic form |
| `compute_polarization_metrics()` | `network/opinion.py` | `n_pos`, `n_neg`, `dispersion`, `mean_q_gap` |
| `opinion_states_to_dict()` | `network/opinion.py` | Serialises Q-trajectories (argmax + optional softmax) for JSON logging; per-draft expressed stances live in the gate records, not the snapshot |
| `opinion_states` field | `network/state.py` | `dict[str, AgentOpinionState]` on `NetworkState` |
| Reward computation | `agents/agent.py` | `reward_from_expressions(graded_expr, graded_resp)` = `graded(expresser) · graded(reaction)`, the graded opinion product; no LLM call (both factors reused from the gate records). Superseded the relational `classify_reward()` on 2026-07-17 (see §1) |
| Expression classification | `agents/agent.py` | `classify_expression_graded(text)`, grades the stance a turn takes on the topic in [−1, 1]; its sign (`classify_expression()`) arbitrates the gate and tags memories, its magnitude feeds the reward product. The single classifier in the system |
| Opinion-conditioned response | `agents/agent.py` | `respond(message, speaker, softmax_stance)`, soft stance anchor plus a cached persona-stance bridge (`_derive_stance_bridges()`) so the LLM doesn't invent a reconciliation; the drawn stance is enforced output-side by the SFT expression gate (`_sft_gate()`), not by a hard prompt constraint (see §7) |
| Discussion wiring | `network/discussion.py` | Passes `softmax_a/b` to every `respond()` call; accepts `prior_b_message`, when set, agent_a continues from agent_b's last message rather than the moderator's opening; forms each reward as the product of two graded stances read off the gate records (`reward_from_expressions()`) |
| Main loop wiring (network) | `main_network.py` | asymmetric draw → `draw_softmax_stance(β)` → `run_discussion(turns=1, prior_b_message=…)` → `update_q_value(expressed_stance, …)` → snapshot; `last_message_to[(speaker, listener)]` tracks the last utterance each agent sent to each partner so repeat meetings continue the dialogue |
| Main loop wiring (pairwise) | `main_pairwise.py` | same SFT mechanisms; `INTERACTIONS_PER_ROUND` interactions/round (expresser drawn uniformly); no graph or community; same `last_message_to` continuation tracking |
| Responder selection | `network/matching.py` | `select_responder(β_sel)`, β_sel=0 uniform, β_sel>0 weighted by expresser's own mean reward history per neighbour |

---

## Key design decisions and divergences from the original plan

### 1. Reward source: an independent classifier, not transcript self-scoring

**Original plan:** have the speaking agent rate the whole transcript for opinion concordance and pass that score directly as the reward.

**Actual implementation (since 2026-07-17):** the reward is the graded **opinion product** `r = graded(expresser) · graded(reaction)` (`reward_from_expressions()`), where each factor is a context-free `classify_expression_graded()` reading of one turn against the topic — the same classifier that arbitrates the gate. No separate reward LLM call is made; both factors are reused from the gate records.

**Why:** a self-scoring rating is produced by the agent that just spoke, on the transcript it was part of generating. That couples expression and evaluation, the generating LLM scores its own interactional outcome in a shared context, contaminating the causal chain. Sárközi et al. (2022) specifically found that feedback processing is the critical link; keeping it causally independent from expression is necessary for the interpretability claim to hold. Grading each turn *in isolation* against the topic preserves that independence and, because both factors use one topic-anchored scale, makes the reward symmetric by construction (mutual favor and mutual opposition are the identical judgement).

The self-scoring approach was never carried into the codebase; edge valuation likewise derives entirely from the rolling mean of the reward products accumulated in `EdgeData.reward_history` (`network/edges.py`), so no separate LLM concordance judgment is made per interaction.

**Theoretical basis for the product form (design commitment, 2026-07-19):** multiplying both graded magnitudes commits to the position that social reward learning scales with the expressed determination of *both* parties — including the expresser's own, whose mean |graded| then ceilings its attainable conviction (hedger ceiling, diagnostic F3). Grounds: it is the direct continuous generalisation of Banisch & Olbrich's `r = o_i · o_j` (the own-side factor is already in the original, merely hidden by the binary encoding); product updates are the Hebbian/coactivation form of connectionist attitude models (van Overwalle & Siebler 2005; Monroe & Read 2008); the own factor tracks self-perception and public-commitment effects (Bem 1972; Kiesler 1971; Downing, Judd & Brauer 1992), the partner factor reinforcement magnitude in verbal operant conditioning (Verplanck 1955; Hildum & Brown 1956). The rival pure-feedback reading `r = sign(own) · graded(reaction)` is kept as a planned ablation. Full argument in README "`reward_from_expressions`".

> **History (2026-07-12 → 2026-07-17).** Through 2026-07-16 the reward was a *relational* `classify_reward(expression_text, reaction_text)` method that scored how the partner's reaction agreed with the position in the expresser's own statement (minimal prompt, no persona/transcript). It was replaced by the product form after the five 2026-07-12/-17 runs showed it under-scored *mutual rejection* as disagreement (4–22 % of mutual-oppose exchanges vs <4 % of mutual-agreement), a direction-biased artifact that tilted every run toward a +1 consensus; grading each turn absolutely removes the relational judgement where that asymmetry lived. See README "Where the extension lives".

### 2. Opinion initialisation: neutral, not LLM-bootstrapped

**Original plan:** call the LLM for each agent to get a `ja/nein` answer and initialise Q-values with a small random offset in the stated direction.

**Actual implementation:** all agents initialise at Q(+1) = Q(−1) = 0.0.

**Why:** LLM-bootstrapped opinions introduce model bias (the left-lean and truth-bias documented by Chuang et al. 2024) at the very first time step, before any social interaction. This conflates model bias with social dynamics and makes the Q-trajectories harder to interpret. Starting at zero means any divergence in Q-values is entirely caused by the social feedback received, which is exactly what the interpretability check measures. Agents may still start expressing different stances by round 2 due to randomness in the softmax draw and asymmetric early rewards.

### 3. Graph initialisation: SBM instead of Watts-Strogatz

The original plan did not specify a graph topology change. The implementation switches to a Stochastic Block Model (SBM) with `SBM_NUM_COMMUNITIES` blocks and a tunable `SBM_P_INTER` coupling parameter. This is necessary because the phase-transition baseline sanity check (Banisch & Olbrich 2019) requires sweeping community modularity as a controlled variable. Watts-Strogatz rewiring probability does not give a clean inter-group coupling knob.

### 4. Asymmetric interaction rule: one expresser, one Q-update

**Previous implementation:** symmetric pairs matched by max-weight matching; both agents' Q-values updated after every discussion.

**Actual implementation:** each interaction draws one expresser uniformly at random, one responder from the expresser's neighbourhood, runs one exchange (`turns_per_agent=1`), and updates only the expresser's Q-value.

**Why:** Jacob & Banisch (2023) establish that the one-directional update is what produces the asymmetric social-feedback dynamics: agents shift their opinion based on the reaction they receive when speaking, not when listening. The symmetric bilateral update obscures this directionality and changes the phase-transition behaviour. Using `turns_per_agent=1` in `run_discussion()` matches the single-exchange base unit of the model; `INTERACTIONS_PER_ROUND` (default = `NUM_AGENTS`) defines how many such events constitute a snapshot round.

On the first meeting between a pair, the expresser responds to the moderator's opening (`TOPIC_TEXT`). On all subsequent meetings, the expresser responds to the partner's last message from the previous exchange, making repeated interactions a continuous dialogue rather than independent topic-seeded conversations. The topic text is always injected into the stance hint so agents remain oriented to the discussion question regardless of what the preceding message was. Both main entry points maintain a `last_message_to[(speaker, listener)]` dict for this; the key is directed so that role swaps (B becomes expresser where A was before) are handled correctly.

### 5. Softmax inverse temperature β replaces temperature τ

**Previous implementation:** `softmax_opinion(temperature)`, τ=0 collapses to argmax, τ>0 adds noise.

**Actual implementation:** `draw_softmax_stance(beta)`, uses the logistic form `p(+1) = σ(β · Δq)`.

**Why:** Banisch uses inverse temperature throughout. More importantly, the parameterisation change fixes the initialisation artifact: at β=0 (or equivalently β·Δq=0 when Q-values are tied at 0), `σ(0) = 0.5`, so agents are split 50/50 at initialisation without any random jitter. The old τ=0 path returned argmax and sent every agent to +1. The softmax stance is passed explicitly to `respond()` to anchor the prompt; see decision 7 for how the *Q-update* index was subsequently corrected to use the expressed stance instead.

### 6. Responder-selection bias β_sel

**Previous implementation:** responder drawn uniformly from graph neighbours.

**Actual implementation:** `select_responder(expresser, neighbours, graph, beta)` weights each neighbour by `exp(β_sel · expected_reward(expresser, neighbour))`, where `expected_reward` is the mean of the expresser's own rolling reward history on that edge (`EdgeData.reward_history[expresser]`), populated by `record_reward()` after every measured-reward interaction (a no-signal round, an abstention or an unparseable reward, freezes the reward record along with the Q-update and edge dynamics). Neighbours with no history yet contribute a neutral expected reward of 0.0 (cold start).

This is *reward-based* homophily: the expresser preferentially returns to partners who have historically validated their position. At β_sel=0 the draw is exactly uniform, recovering the Banisch & Olbrich (2019) baseline. At β_sel>0, neighbours associated with higher past reward receive higher selection weight. This differs from Jacob & Banisch (2023)'s conviction-similarity homophily (`exp(−h·|Δq_i−Δq_j|)`) and is a deliberate extension: rather than matching on stance similarity, the expresser is drawn toward socially rewarding partners, which can generate echo chambers through reinforcement even when conviction gaps are small.

Keeping this as a standalone swappable function means the virtual-worlds extension (replacing the neighbour set with a cross-platform adjacency list) requires no changes to the main loop.

### 7. Q-update index: the stance that entered the interaction, enforced by the expression gate

**Original implementation (bug):** `update_q_value()` was indexed on the softmax-drawn stance, the stance drawn to *anchor* the prompt, not the stance the agent actually produced in `first_a_msg`.

**Why it matters:** the reward `reward_a` is the social response to what the agent *actually said*. Updating `Q(softmax_a)` when the generated text expressed a different stance updates a Q-value for a stance that never entered the conversation, the TD update is only semantically valid when indexed on the stance that was actually socially evaluated. Persona and memory are legitimate opinion-shaping inputs that can pull the generated text off the softmax anchor (e.g. persona pulls the LLM toward its natural political valence), so the drawn and expressed stances can genuinely diverge.

**Actual implementation, the SFT expression gate** (`Agent._sft_gate()`, see README "SFT expression gate"): rather than let a divergent expression enter the interaction and then index the update on it, the drawn stance is enforced *output-side*. The generation prompt stays soft (persona-stance bridge, no hard anti-drift constraint), but each draft is classified by `classify_expression()` and a draft that *flips* the softmax stance or is *unparseable* (classifier returned `None`) is rejected and regenerated (on-stance or genuine-ambiguous drafts pass; a three-stage cascade guarantees termination). So the text that enters each interaction never contradicts the draw, and the main loops index `update_q_value()` on `gate_final_expressed_a`, which under the gate is the softmax draw itself. When that expressed stance is *ambiguous* (`0`), the update is skipped: a vague utterance is a strategic abstention that puts no stance up for social evaluation. Setting `SFT_GATE_ENABLED = False` disables the gate (single ungated draft); the same indexing-plus-ambiguous-skip then operates on the classified expressed stance, the ablation arm.

The persona field (how often the *first* draft diverges from the anchor) is not suppressed, it is measured: every turn's gate record logs each draft's expressed stance, so first-attempt fidelity and retry counts are reconstructed post-hoc by `analyze_gate.py`.

---

## Output format

Every `round_NNNN.json` snapshot now contains:

```json
{
  "metrics": {
    "density": 0.33,
    "n_components": 1,
    "avg_degree": 2.0,
    "n_edges": 4,
    "n_pos": 3,
    "n_neg": 1,
    "dispersion": 0.75,
    "mean_q_gap": 0.12
  },
  "opinion_states": {
    "Anna": {"q_pos": 0.18, "q_neg": 0.04, "argmax": 1, "softmax": 1, "q_gap": 0.14},
    "Ben":  {"q_pos": -0.02, "q_neg": 0.11, "argmax": -1, "softmax": -1, "q_gap": -0.13}
  }
}
```

`opinion_states` records the full Q-trajectory for every agent, enabling the interpretability check: regressing Q-gap trajectories against observed opinion switches to test whether the SFT layer genuinely governs the LLM's expressed positions.  `"argmax"` is the deterministic argmax (`argmax_stance` property, always present); `"softmax"` is the softmax-drawn stance from the agent's last interaction as expresser in the round (absent if not selected as expresser).  `n_pos`/`n_neg` in `metrics` count agents with a strict positive/negative Q-gap (`q_pos > q_neg` and `q_neg > q_pos` respectively); tied/neutral agents (equal Q-values) are excluded from both, so `n_pos + n_neg ≤ total agents`.

Each `discussion` record in `events.jsonl` also carries the stance fields for the expresser (a) and responder (b), plus the expression-gate summary:

| Field | Meaning |
|---|---|
| `softmax_a` / `softmax_b` | Softmax-drawn stances for this interaction (the prompt anchor, enforced by the gate) |
| `gate_final_expressed_a` / `gate_final_expressed_b` | Accepted first-turn stance (+1/−1/0); the Q-update indexes on `gate_final_expressed_a` and is skipped when it is `0` (ambiguous = strategic abstention) |
| `gate_attempts_a` / `gate_attempts_b` | Drafts the agent's first turn needed (1 = first draft passed) |
| `gate_fallback_a` / `gate_fallback_b` | Whether that turn's cascade left the soft-prompt stage |
| `argmax_a` / `argmax_b` | Deterministic argmax (pre-Q-update) going into this interaction, note: distinct from n_pos/n_neg counts, which use strict Q-gap comparisons |

Each turn inside `turns` additionally carries its full `gate` record (every draft text, per-draft expressed stance, cascade stage), the persona-field telemetry mined by `analyze_gate.py`. This allows per-interaction analysis without cross-referencing the round snapshot.

---

## What remains for the extension chapter

Setting `GRAPH_DYNAMIC = True` in `config.py` activates the homophilic tie-formation mechanism:

- Each interaction appends `reward_a` to the expresser's rolling history deque on that edge (`EdgeData.reward_history`, window length `REWARD_WINDOW_M` from `config.py`)
- The rolling mean of that history drives the expresser's valuation update via `update_edge()` (asymmetric, responder's side unchanged; enable symmetric mode by passing `reward_b` at the call site)
- Edges are severed when either agent's valuation reaches `STRENGTH_FLOOR`
- `ensure_connectivity()` reconnects isolated agents

This corresponds to Jacob & Banisch's (2023) virtual-worlds setup where structural co-evolution is driven by the same reward signal as opinion learning: the reward product (`reward_from_expressions()`) serves dual duty: Q-value update and edge evaluation. Known design risks: (1) graph crystallising before Q-values diverge, keep `STRENGTH_DELTA` small relative to `LEARNING_RATE`; (2) premature edge drops before the history window fills, cold-start signal defaults to 0.0 (neutral).

---

## References

Banisch, S., & Olbrich, E. (2019). Opinion polarization by learning from social feedback. *The Journal of Mathematical Sociology, 43*(2), 76-103.

Jacob, D., & Banisch, S. (2023). Polarization in social media: A virtual worlds-based approach. *Journal of Artificial Societies and Social Simulation, 26*(3), 11.

Sárközi, R., Denz, T., & Lorenz-Spreen, P. (2022). Testing social feedback theory: An experiment on the effect of social feedback on opinion expression. *PLOS ONE, 17*(4).

Chuang, Y.-S., Goyal, A., Harlalka, N., Suresh, S., Hawkins, R., Yang, S., ... & Yang, D. (2024). Simulating opinion dynamics with networks of LLM-based agents. *NAACL 2024*.
