# Weight-Level Social-Feedback Learning: A Middle Ground

This document works out whether — and how — the social-feedback signal could learn *inside the LLM's parameters* rather than in the external Q-layer, and lands on a concrete middle-ground design (periodic, offline, per-agent adapter distillation) that keeps the interpretability claim intact. It first restates why the *pure* RLHF-from-social-feedback version is the wrong main mechanism for this thesis, so the middle ground is read against that baseline, not in a vacuum.

Nothing here is implemented. This is an extension-chapter design note, in the same register as `PLAN_banisch_opinion.md`.

---

## 0. Terminology: two things are called "SFT"

| Acronym in context | Meaning here | Meaning in the ML literature |
|---|---|---|
| **SFT** (this project, throughout the README) | **Social Feedback Theory** (Banisch & Olbrich 2019) | — |
| **SFT** (RL/LLM literature) | — | **Supervised Fine-Tuning** |

The most natural weight-level version of "learn from social feedback in the model" is reward-filtered supervised fine-tuning — i.e. *SFT (supervised fine-tuning) on SFT (social-feedback-theory) rewards*. This collision is not cosmetic: it will confuse every ML-literate reader of the thesis. Whenever this document means the training method, it writes **supervised fine-tuning** in full; **SFT** alone always means Social Feedback Theory, consistent with the README.

---

## 1. What the current architecture is, stated precisely

The distinction the whole document turns on:

- The **Q-layer is a two-armed contextual bandit per agent**. Actions are `{express +1, express −1}`; the policy is `softmax(β · Q)` (`draw_softmax_stance`); learning is exponential-moving-average Q-learning (`update_q_value`, `Q ← (1−α)·Q + α·r`). This *is* Banisch & Olbrich's process, not an analogue of it.
- The **LLM is a fixed conditional generator**. It never learns. It is *steered* by the bandit through the prompt anchor, and the [expression gate](README.md#sft-expression-gate) exists precisely to force the frozen generator to emit the action the bandit selected.

Policy and generator are two separate objects. Every opinion shift traces to an auditable scalar Q-trajectory — the property the research question exists to preserve ("Can Social Feedback Theory serve as underlying reinforcement-learning mechanism … such that the resulting hybrid model preserves SFT's analytic interpretability").

**"Directly influencing the model itself" means merging the policy and the generator**: the LLM's generation distribution *becomes* the policy, and social reward updates its parameters. That single move is the subject of this document.

---

## 2. The pure version: RLHF from social feedback (and why it is the wrong main mechanism)

### 2.1 How it would work

Replace the tabular Q with the LLM's own generation policy `π_θ`. Keep `reward_from_expressions()` unchanged as the reward function — it is already causally independent of the generator and symmetric by construction, so it ports over as-is. Then optimise weights to maximise expected social reward:

```
∇θ E[r] = E[ r · ∇θ log π_θ(utterance | context) ]        (policy gradient / PPO / GRPO)
```

Because N agents need N divergent opinion trajectories, "θ" cannot be one shared model — it must be **one small per-agent adapter (LoRA) over a frozen base**. A single shared model cannot hold twenty different evolving opinions at once.

### 2.2 Why it is the wrong *main* mechanism (drawbacks restated)

These are the reasons the pure version is rejected as the core learning rule. The middle ground in §3 is designed specifically to avoid each one.

| # | Drawback | Why it is disqualifying here |
|---|---|---|
| **D1** | **Destroys the interpretability claim** | Opinion state moves from a legible scalar `Q(o)` into billions of parameters. The core interpretability check — regressing Q-gap trajectories against observed opinion switches — has nothing left to regress. This alone is fatal to the thesis as framed. |
| **D2** | **Severs the correspondence to Banisch** | The 2019 model *is* a tabular Q-learning bandit; the mapping to `r = o_i·o_j`, the α/β parameters, and the modularity-driven phase transition are *exact* only because the learning rule is literally theirs. Policy gradient on an LLM is an analogue. The phase-transition sanity check against the analytic result becomes impossible; comparability to the canon evaporates. |
| **D3** | **Catastrophic sample inefficiency** | Tabular Q converges in a handful of interactions per agent (why 25 rounds suffices). LLM policy gradient needs 10³–10⁶ reward samples to move weights without wrecking fluency, each costing a generation + a classification. The reward is sparse and ≈±1. Likely outcome: no movement, or mode collapse. |
| **D4** | **Reward hacking → sycophancy collapse** | A reward for agreeing with your partner drives the policy toward sycophancy and homogenised text, annihilating the discourse-corpus value (README "Where the extension lives", point 5). It is the learned, baked-in version of the stage-3-template problem. |
| **D5** | **Re-confounds model bias with social dynamics** | The neutral `Q = 0` initialisation (`PLAN_banisch_opinion.md` §2) exists to keep pretraining bias out of the first time step. Every gradient step re-entangles pretraining priors with social feedback; clean causal attribution gets *harder*. |
| **D6** | **Unstable, unsweepable multi-agent co-training** | With all adapters updating at once, every agent's reward distribution is non-stationary (its partners are also changing). And `run_sweep.py`'s cheap, deterministic grid over β / `SBM_P_INTER` / β_sel becomes a stochastic, days-long training run. The controlled-experiment backbone is lost. |

Note that Park et al. (2023), cited in the README, keep agent weights frozen for exactly these reasons: the field's default for *interpretable* social simulation is the harnessed architecture already built here, not weight-level RL.

---

## 3. The middle ground: periodic offline adapter distillation, Q stays the controller

The pure version fails because it makes the weights the *seat of opinion*. The middle ground keeps the **Q-layer as the sole seat of opinion and the sole learning rule**, and lets the weights carry only *expression* — how fluently and self-consistently an agent voices the opinion the Q-layer already holds.

### 3.1 The one design principle everything follows from

> **The Q-trajectory remains the ground-truth, legible opinion state. A per-agent adapter is a pure rendering/entrenchment layer, trained only to make the agent's expression of its Q-selected stance more fluent and characteristic. The adapter never carries the opinion; it renders it.**

Consequences that fall straight out of this principle:

- The interpretability check (D1) is untouched: `Q(o)` is still the opinion, still regressed against opinion switches. The adapter changes *diction*, not *state*.
- The Banisch correspondence (D2) is untouched: the learning rule is still `Q ← (1−α)·Q + α·r`; the phase transition still runs on the Q-layer; `run_sweep.py` still sweeps the same scalars.
- The reward function (`reward_from_expressions()`) is reused verbatim — the adapter is trained from the *same* signal that updates Q, so no second reward semantics is introduced.

### 3.2 Mechanism

Adapter training is **offline, periodic, per-agent, and reward-filtered** — never online policy gradient. Two admissible variants, in increasing fidelity/cost:

1. **ReST / rejection-sampling fine-tuning (recommended default).** Every `ADAPTER_DISTILL_EVERY` rounds, for each agent, gather that agent's *accepted, measured-reward* turns from `events.jsonl`, keep those whose social reward exceeds a threshold and whose expressed stance matches the agent's *current argmax stance*, and supervised-fine-tune the agent's LoRA on `(context → utterance)` pairs. Iterate the next period. Stable, cheapest, closest in spirit to expert iteration.
2. **DPO on reward-derived preference pairs.** For matched contexts, treat the higher-social-reward utterance as preferred over the lower-reward one; optimise the adapter with DPO. No reward-model training, more stable than online PPO, but needs paired data and is costlier to assemble.

Online PPO/GRPO is **explicitly out of scope** — it reintroduces D3 (sample hunger) and D6 (non-stationary co-training) that the offline, periodic cadence is chosen to avoid.

### 3.3 What ports unchanged vs. what is new

| Component | Status under the middle ground |
|---|---|
| `update_q_value`, `draw_softmax_stance`, Q as opinion state | **Unchanged.** Still the only learning rule and the only opinion state. |
| `reward_from_expressions()` | **Unchanged.** Same reward feeds both Q-update and adapter filtering. |
| Expression gate (`_sft_gate()`) | **Kept, and its role shrinks over time.** The gate still guarantees the drawn stance enters each interaction. But as an agent's adapter internalises its rewarded stance, first-attempt fidelity should *rise*, so the gate should fire fewer retries — a directly measurable prediction (see §3.5). |
| `analyze_gate.py` | **Unchanged, and becomes the primary instrument** for detecting internalisation (retry-rate decline per agent over distillation periods). |
| `run_sweep.py`, phase-transition sweep | **Unchanged.** The sweep runs on the Q-layer, which is untouched. |
| Per-agent LoRA adapters + offline trainer | **New**, and lives entirely outside the interaction loop. |
| `config.py` knobs (see §3.4) | **New**, all gated behind a single default-off switch. |

### 3.4 Configuration (all new, default-off)

Mirrors the `GRAPH_DYNAMIC` pattern: a single boolean that leaves the canonical experiment untouched when false.

| Setting | Default | Description |
|---|---|---|
| `ADAPTER_DISTILL_ENABLED` | `False` | Master switch. `False` reproduces the current frozen-generator architecture exactly. |
| `ADAPTER_DISTILL_EVERY` | `10` | Rounds between offline distillation passes. Must be ≫ `REFLECT_EVERY` so periods accumulate enough measured-reward turns. |
| `ADAPTER_METHOD` | `"rest"` | `"rest"` (rejection-sampling FT) or `"dpo"`. |
| `ADAPTER_REWARD_FLOOR` | `0.3` | Only turns with `reward_a ≥` this and stance = current argmax enter the training set (keeps the adapter aligned to the Q-state, not to noise). |
| `ADAPTER_MAX_PAIRS` | `64` | Cap on training examples per agent per period, bounding cost and drift. |

### 3.5 The experiment this enables

The middle ground is not just a safer engineering choice — it poses a real question that is structurally parallel to the existing gated-vs-ungated A/B (`SFT_GATE_ENABLED`):

> **Does internalising social experience into the expression layer change the macro-dynamics, holding the opinion-learning rule fixed?**

Because the Q-layer is identical across arms, any divergence in polarization metrics between `ADAPTER_DISTILL_ENABLED ∈ {False, True}` is attributable purely to expression internalisation, cleanly separated from opinion learning. Concrete, pre-registered predictions:

1. **Falling gate retry rate.** Per-agent first-attempt fidelity (`analyze_gate.py`) rises monotonically across distillation periods as the adapter learns the agent's rewarded stance — the empirical signature that internalisation is happening at all. (This is the same signature the dropped conviction-weighted-deliberation design predicted; see README "Future research".)
2. **Sharper linguistic camps.** Lexical/stylistic divergence between the two communities grows under distillation even at fixed Q-gaps — polarization the scalar layer cannot represent, surfacing in the discourse corpus.
3. **Possible hysteresis in the phase transition.** If expression entrenchment feeds back into partner reactions, the `SBM_P_INTER` consensus threshold may shift relative to the frozen-generator baseline — a testable second-order effect.

### 3.6 Residual risks (and how the design bounds them)

| Risk | Bound |
|---|---|
| Adapter drifts into carrying opinion (reintroducing D1) | `ADAPTER_REWARD_FLOOR` + stance = current-argmax filter admit only turns already consistent with the Q-state; the adapter can only *reinforce* what Q already selected, never override it. Audit: if an agent's *first draft* stance starts contradicting its argmax more often, the adapter has overstepped — `analyze_gate.py` already logs exactly this. |
| Sycophancy collapse (D4) leaking in via distillation | Offline reward-filtering on *accepted, on-stance* turns only, capped at `ADAPTER_MAX_PAIRS`, cannot chase a live reward gradient toward a degenerate utterance the way online PPO can. Periodic cadence + KL-anchoring to the base model (standard LoRA/DPO practice) further caps drift. |
| Distillation entangling pretraining bias (D5) | The base model stays frozen and shared; only low-rank per-agent deltas move, seeded from post-`Q=0` social experience, so the neutral-initialisation guarantee on the *opinion state* is preserved even though expression style personalises. |
| Cost / reproducibility | Training is offline and periodic, so it never blocks the interaction loop, and every training set is reconstructable from `events.jsonl` + the seed — reproducibility survives. |

---

## 4. Infrastructure reality

The middle ground is cheaper than the pure version but still crosses a stack boundary the current project has not:

- **Ollama is inference-only; it does no backprop.** Any weight update — even a LoRA — needs a training stack: `transformers` + `peft` + `trl` (for the ReST/DPO loop), and GPU memory to fine-tune adapters over a 35B base. Inference during the simulation could still run through Ollama/vLLM; only the periodic distillation pass needs the training stack.
- **N adapters, not N models.** Twenty LoRA adapters over one frozen base is the storage/compute unit — one optimizer state per agent during each offline pass, swapped in as the active adapter when that agent speaks. This is tractable on a single capable GPU precisely because distillation is periodic and offline, not per-turn.
- **Serving swappable adapters.** vLLM supports per-request LoRA selection, which fits the per-agent design directly; Ollama's adapter support is more limited and should be validated before committing.

This is a real but bounded jump — far smaller than the pure version's continuous online RL over a 35B policy.

---

## 5. Recommendation

- **Do not** make weight-level RL the main learning mechanism. The pure RLHF-from-social-feedback version trades away exactly the auditability the thesis exists to demonstrate (D1–D2) and is unstable and unsweepable on top of it (D3–D6).
- **Do** consider the middle ground as an **extension-chapter arm**, gated behind `ADAPTER_DISTILL_ENABLED = False` so the canonical experiment is byte-for-byte unaffected. It answers a genuine question — *does expression internalisation alter the macro-dynamics when opinion learning is held fixed?* — while the Q-layer remains the sole, legible seat of opinion.
- Frame it in the thesis as the deliberate boundary it is: the harnessed architecture keeps learning symbolic and external so the mechanism stays auditable; weight-level distillation buys expressive richness at the price of that auditability, and the middle ground pays only the part of that price (expression, not opinion) that the interpretability claim can afford.

---

## References

Banisch, S., & Olbrich, E. (2019). Opinion polarization by learning from social feedback. *The Journal of Mathematical Sociology, 43*(2), 76-103.

Jacob, D., & Banisch, S. (2023). Polarization in social media: A virtual worlds-based approach. *JASSS, 26*(3), 11.

Christiano, P., Leike, J., Brown, T., Martic, M., Legg, S., & Amodei, D. (2017). Deep reinforcement learning from human preferences. *NeurIPS 2017*. *(RLHF, the reward-model-from-feedback template this idea instantiates with a simulated society.)*

Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback. *NeurIPS 2022*. *(InstructGPT; the canonical RLHF/PPO pipeline and its reward-hacking failure modes, D4.)*

Bai, Y., et al. (2022). Constitutional AI: Harmlessness from AI feedback. *arXiv:2212.08073*. *(RLAIF; feedback from a model rather than humans — the precedent for feedback from a simulated society.)*

Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). Direct preference optimization: Your language model is secretly a reward model. *NeurIPS 2023*. *(DPO; the offline preference-pair variant in §3.2.)*

Gulcehre, C., et al. (2023). Reinforced self-training (ReST) for language modeling. *arXiv:2308.08998*. *(The offline, periodic, reward-filtered fine-tuning loop adopted as the §3.2 default.)*

Dong, H., et al. (2023). RAFT: Reward rAnked FineTuning for generative foundation model alignment. *TMLR*. *(Rejection-sampling fine-tuning, the ReST-family method underlying the default variant.)*

Hu, E. J., et al. (2021). LoRA: Low-rank adaptation of large language models. *arXiv:2106.09685*. *(The per-agent adapter mechanism that keeps N divergent expression styles over one frozen base.)*

Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2023). Generative agents: Interactive simulacra of human behavior. *UIST 2023*. *(Interpretable social simulation with frozen agent weights — the field default this document upholds.)*

> References recalled from background knowledge; verify each (year, venue, exact title) against the source before citing in the thesis text — consistent with the standing caution in the README and `PLAN_banisch_opinion.md`.
