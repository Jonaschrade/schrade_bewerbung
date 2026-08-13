# Test Schedule

## Design principles

Four methodological principles drive this schedule (vs. a naive one-run-per-cell grid):

1. **The regime outcome is stochastic: the dependent variable is a regime *fraction*, not a regime.**
   In Banisch & Olbrich (2019), the low-coupling regime makes bi-polarization a stable attractor
   *alongside* consensus: with 2 communities the system has four absorbing states
   (+/+, +/−, −/+, −/−), all stable at low coupling. Under stance symmetry, a single low-coupling
   run reaches bi-polarization with probability ≈ 0.5 and consensus-by-chance with probability
   ≈ 0.5. A single run per parameter point therefore cannot distinguish regimes at all; the phase
   transition manifests only as the *fraction of replications* that polarize, per `SIM_SBM_P_INTER`
   value. Every experimental cell below carries a replication count `r`.
   With r = 5, observing 5/5 consensus at the polarized anchor has p ≈ 0.03 under the symmetric
   50/50 expectation, enough to flag a broken symmetry; r = 3 is the floor for cells where only
   an effect *direction* is needed.

2. **Regime classification is 3-way, and consensus carries a direction.**
   Per run, at convergence: **bi-polarization** (community mean stances have opposite signs),
   **consensus** (same sign, record the sign), **unsettled** (a community's internal dispersion
   stays high / mean q_gap near 0). Consensus-by-chance and coupling-driven consensus are
   indistinguishable within a single run, only the ensemble fraction vs. the 50% baseline
   separates them. The **direction** of each consensus run is itself a bias probe: under
   symmetry, direction is a coin flip, so k consensus runs all landing pro-restriction has
   p = 0.5^k. This costs nothing and is tallied across all stages.

3. **The reward channel must be audited for stance asymmetry *before* the grid runs.**
   The classifier prompt (`classify_reward()`) is facially direction-neutral, but the bias risk
   lives in the model's priors interacting with topic content, invisible to prompt inspection.
   Since the TD fixed point is Q* = r̄, a mean classifier bias of Δ shifts the Q equilibrium
   one-to-one; against typical q_gaps of 0.1-0.5, |Δ| ≥ 0.1 is dynamically meaningful and would
   act as an external field that can wash out the phase transition at every coupling level.
   Stage 0b measures Δ offline and gates the rest of the schedule.

   > **Resolution (2026-07-17).** Under the nuclear-power topic this risk materialised: a
   > five-run log audit (`analysis/`, 2026-07-12/-17 runs) found the *relational*
   > `classify_reward()` under-scored **mutual rejection** as disagreement (4–22 % of
   > mutual-oppose exchanges vs <4 % of mutual-agreement), a direction-biased Δ that pushed
   > every run to a +1 consensus regardless of framing — overturning, for this topic, the
   > 2026-07-03 "reward channel ≈ symmetric" verdict in point 4 (which was measured under the
   > immigration wording). Fix: the reward was switched from the relational classifier to the
   > graded **opinion product** `graded(expresser) · graded(reaction)` (`reward_from_expressions()`),
   > symmetric by construction, so this standing pre-grid audit item is now satisfied at the
   > mechanism level rather than by per-topic measurement. Re-confirm with a log-mining
   > reward-by-dyad check on the first post-change run before the grid. See README "Where the
   > extension lives" and PLAN_banisch_opinion.md §1.

4. **The persona field is a study object, not a defect (design decision, 2026-07-03).**
   Stage 0b's diagnosis decomposed the external field: the **reward channel** is approximately
   direction-symmetric (Gate A PASS), but the **generation channel** is not, under β=0
   coin-flip anchors, agents anchored anti-restriction held their stance in only 43% of
   utterances vs. 84% for pro-restriction (p < 10⁻⁸), and spot-checks confirmed genuine
   persona-grounded drift (not classifier error). The generation prompt is deliberately kept
   *soft* (no hard anti-drift constraint, "Step A" soft anchoring); the SFT expression gate
   ("Step B", `SFT_GATE_ENABLED`, see README "SFT expression gate") then enforces the drawn
   stance *output-side* by rejection sampling, leaving the prompt soft so the first draft of
   every turn still samples the unconstrained distribution the persona field is read from:
   the conflict between social-feedback learning and the
   persona field is exactly the tension this thesis studies. Consequence: every regime
   prediction below is read against **two baselines**, the symmetric-SFT null *and* the
   measured field, and field-dominant outcomes (e.g. uniform pro-restriction consensus) are
   reported as findings about LLM-persona priors, not as failed replications.



## Fixed parameters (constant across all stages)

| Parameter | Value | Env var override |
|---|---|---|
| `NUM_AGENTS` | 20 | `SIM_NUM_AGENTS` |
| `NETWORK_MAX_ROUNDS` | 25 | `SIM_NETWORK_MAX_ROUNDS` |
| `INTERACTIONS_PER_ROUND` | 20 (= `NUM_AGENTS`) | - |
| `REFLECT_EVERY` | 5 | - |
| `LEARNING_RATE` α | 0.1 | - |
| `LLM_MODEL` | qwen3.5:35b | - |
| `LLM_NUM_CTX` | 4096 | - |
| `SBM_NUM_COMMUNITIES` | 2 | - |
| `SBM_P_INTRA` | **0.7 (fixed)** | `SIM_SBM_P_INTRA` |
| `STRENGTH_CAP` | 3.0 | - |
| `STRENGTH_FLOOR` | 0.0 | - |
| `STRENGTH_DELTA` | 0.5 | - |
| `REWARD_WINDOW_M` | 5 | - |
| `MAX_MEMORIES_SEED` | 15 | - |
| `MAX_MEMORIES_RETRIEVE` | 5 | - |
| `TOPIC_LABEL` | Energy Policy (switched 2026-07-04; Stage 0b baselines were measured under Immigration Policy) | `SIM_TOPIC_LABEL` |

> **How these become runs.** `run_stage.sh` encodes this schedule as a `STAGE` →
> parameter table and is submitted as `qsub -v STAGE=1-04 -N stage_1-04 …`; every
> `SIM_*` value above is set from that table, so a queued job's parameters are
> fixed at submission time rather than read from `config.py` when it finally
> starts. Points the schedule leaves open — Stage 2 and 5-02 (the Gate B
> transition point) and Stage 5's permissive wording — must be passed explicitly
> and abort the job if missing. See RUNBOOK "Batch-Betrieb".

---

## Standing diagnostics (computed on every run, zero LLM cost)

Extracted from `events.jsonl` after each run; aggregated per stage. These separate
feedback-driven dynamics from bias entering upstream:

| Diagnostic | Symmetry prediction | Bias signature |
|---|---|---|
| Early-round expressed-stance split (rounds 1-3, where Q ≈ 0 ⇒ softmax draw ≈ 50/50), from gate `first_attempt_expressed` | ≈ 50/50 | Skew ⇒ bias enters at *text generation*, before any feedback. **Measured (β=0 harvest, immigration wording): 74/26 pro-restriction** |
| First-attempt fidelity by anchored stance: P(`first_attempt_expressed` == anchor) for softmax = +1 vs −1 (`analyze_gate.py`) | equal | Asymmetric drift ⇒ persona field pulls one stance off its anchor. **Measured baseline (β=0 harvest, immigration wording): 43% (−1) vs 84% (+1)** |
| Ambiguity by anchored stance: rate of `first_attempt_expressed = 0` conditional on softmax stance (`analyze_gate.py`) | equal | One stance systematically less legible. (Genuine ambiguity only; an *unparseable* draft is `None`, not `0`, the gate retries it, and `analyze_gate.py` reports its rate separately as a classifier-reliability signal) |
| Reward by dyad: mean `reward_a` per (`gate_final_expressed_a`, `gate_final_expressed_b`) cell | E[r\|++] = E[r\|−−]; E[r\|+−] = E[r\|−+] | Cell asymmetry ⇒ reward channel bias (cf. Stage 0b) |
| Consensus direction tally (across runs) | coin flip | k same-direction consensus runs: p = 0.5^k |

---

## Stage 0b, Classifier audit & field diagnosis (0 simulation runs, ~500 classifier calls)

**Purpose:** Decompose the external field into its **reward-channel** component (directional
bias Δ in `classify_reward()`) and its **generation-channel** component (the persona field
pulling text off the anchored stance) before committing the grid budget. Runs offline against
logged text; no simulation. This stage doubles as the *diagnosis analysis* documenting the
social-feedback-vs-persona-field tension (design principle 4).

> **Tooling update (gate implementation):** the standalone `audit_reward.py` harness described
> in steps 1-4 below has been **retired**. The generation-channel diagnosis (step 4) is now
> `analyze_gate.py`, which reads the SFT expression gate's per-turn records straight from
> `events.jsonl`, first-attempt fidelity is the same estimand as the old `fidelity` mode, with
> retry/cascade stats on top (see README "Persona-field analysis"). The reward-channel modes
> (steps 1-3: `mine`/`mirror`/`synthetic`) are not currently reproduced; their 2026-07-03
> Gate A verdict stands as historical record. Note (2026-07-17): the reward-channel modes below
> target the *relational* `classify_reward()`, which no longer exists — the reward is now the
> symmetric graded product (`reward_from_expressions()`, see point 3 Resolution), so the
> mirror/synthetic tests of a directional Δ are moot; a log-mining reward-by-dyad check
> (over `gate_final_expressed_a/b` cells) is still the way to spot-verify per-topic. Steps 1-4
> below document the original methodology.

1. **Log mining**, from all existing runs: reward mean + label distribution per
   expressed-stance dyad (the standing diagnostic, applied retroactively). Flags only;
   content confounds direction.
2. **Mirror test** (decisive), sample 30-50 logged (STATEMENT, REACTION) pairs stratified by
   reward level; LLM-rewrite each pair stance-mirrored (pro ↔ anti) preserving hedging, tone,
   and the agreement relation; verify flips via `classify_expression()`, discard failures.
   Rate originals and mirrors ~5× each. Paired analysis: Δ = r(original) − r(mirror);
   Wilcoxon signed-rank + mean Δ.
3. **Synthetic matrix**, content-free reaction texts (one per agreement label) crossed with a
   pro and an anti statement built from `TOPIC_TEXT`, to cover cells the (consensus-heavy)
   logs lack. Needs no logs at all; also reports scale calibration and the global valence
   offset.
4. **Fidelity transition matrix** (`fidelity` mode), anchored (softmax) → expressed stance
   matrix over logged discussions, with a two-proportion test on per-anchor fidelity. The
   generation-channel diagnosis; most informative on a β=0 harvest run (coin-flip anchors).
   No LLM needed.

**How to run** (current generation-channel diagnosis via `analyze_gate.py`; run logs live on
the HPC, so either run it there or copy `logs/run_*/` directories over and pass their paths):

```bash
python analyze_gate.py --runs <run_dir> [...]     # first-attempt fidelity + retry/cascade stats (no LLM)
python analyze_gate.py --runs <run_dir> --drift   # + semantic drift across drafts (loads embeddings)
```

Results (per-anchor first-attempt fidelity/flip/ambiguity, mean drafts, fallback/template
rates, and the two-proportion fidelity-asymmetry p) are written to
`logs/analysis/<timestamp>/gate.json`. The reward-channel steps (1-3) used the retired
`audit_reward.py`; their verdict below is kept as record.

> **Gate A:** |mean Δ| < 0.1 (one α-step; rationale: Q* = r̄) → proceed to Stage 1.
> Otherwise: A/B prompt variants on the *same* mirror benchmark (direction-blind instruction +
> symmetric few-shot examples → counterbalanced rating → topic masking, in that cost order)
> and/or reword the topic; re-audit until the gate passes, or, if the field is irreducible,
> document Δ and reinterpret Stage 1 as a *biased-field* replication (report, don't hide).
> Also record the global valence offset (mean r across all cells): a uniformly positive tilt
> accelerates lock-in for whichever stance appears early, even if direction-symmetric.

---

## Stage 1, Phase transition scan (5 points × r = 3-5 → 23-25 runs)

**Purpose:** Locate the polarization-to-consensus phase transition as a **regime-fraction
curve** over `SIM_SBM_P_INTER`. Core replication of Banisch & Olbrich (2019).

**Fixed:** `SBM_P_INTRA=0.7`, `OPINION_BETA=5.0`, `RESPONDER_SELECTION_BETA=0.0`,
`GRAPH_DYNAMIC=0`, topic=restrictive

Grid trimmed from 7 to 5 points to fund replication; the transition is located adaptively
(see rule below) instead of by dense single runs. Run 1-01 is the polarized anchor
(~2 cross-community edges). Run 1-05 is the **Erdős-Rényi null model**: at
P_INTER = P_INTRA = 0.7 there is no block structure, so any *systematic* opinion separation,
or any consistent consensus *direction* across its replicates, is attributable to LLM priors,
not topology. With replication the ER null becomes the cheapest bias probe in the schedule:
r same-direction consensus outcomes there has p = 0.5^r under symmetry.

| Run | `SIM_SBM_P_INTER` | Cross-community edges (expected) | Expected regime fraction (symmetric SFT) | r |
|---|---|---|---|---|
| 1-01 | 0.02 | ~2 | ~50% bi-polarization / ~50% consensus (direction ~coin flip) | 5 |
| 1-02 | 0.08 | ~8 | Transition zone, mixed | 5 |
| 1-03 | 0.15 | ~15 | Post-transition, consensus-dominant | 5 |
| 1-04 | 0.30 | ~30 | Consensus | 3 |
| 1-05 | 0.70 | ~70 | ER null: consensus, direction coin flip | 5 |

**Sequential rule:** run r = 5 at 1-01, 1-02, 1-03 and 1-05, r = 3 at 1-04 (see below).
Extend a point with a *mixed* regime outcome to r = 5; if the polarization fraction jumps
sharply between adjacent points, add one midpoint at r = 3-5. This spends replication where the
transition actually is instead of guessing its location.

> **1-02 and 1-03 committed to r = 5 upfront (2026-08-04 / -08-06).** The extend-on-mixed rule
> would almost certainly fire at both anyway: 1-02 *is* the predicted transition zone, so a
> mixed outcome is the expectation rather than the surprise, and 1-03 at 0.15 is the point that
> has to resolve whether the transition has completed — the two measurements that jointly locate
> Gate B. Each extension costs a full queue cycle on the SCC (one `qsub` per sub-stage, waits of
> days when the `gpu` queue is full), which buys nothing when the extension is near-certain.
> Equal r across 1-01/1-02/1-03/1-05 also keeps the regime-fraction curve on a uniform
> denominator over the whole transition region, with only the deep-consensus point 1-04 — where
> the outcome is least in doubt — left at r = 3 under the original rule.

> **Gate B:** the empirical transition point (P_INTER where the polarization fraction crosses
> ~50%) anchors Stages 2 and 3. **Falsification check:** if 1-01 yields 5/5 consensus
> (p ≈ 0.03 under symmetry), and especially if all in the same direction, the external field
> dominates the feedback mechanism; return to Gate A remedies before spending Stages 2-5.

---

## Stage 2, Opinion beta sensitivity (3 values × r = 3 → 9 runs)

**Purpose:** Test how expression determinism (β) affects convergence speed and final
polarization depth, *at the empirically located transition point* where the system is most
sensitive.

**Fixed:** `SBM_P_INTRA=0.7`, `SBM_P_INTER=` **empirical transition point from Gate B**,
`RESPONDER_SELECTION_BETA=0.0`, `GRAPH_DYNAMIC=0`, topic=restrictive

| Run | `SIM_OPINION_BETA` | r |
|---|---|---|
| 2-01 | 1.0 | 3 |
| 2-02 | 2.0 | 3 |
| 2-03 | 5.0 | 3 |

> β = 3.0 dropped to fund replication, with r = 3 per value the *trend* across 1.0/2.0/5.0 is
> testable, which a 4-point single-run design was not. Outcome measures: rounds-to-lock-in,
> final dispersion, regime fraction.

---

## Stage 3, Responder selection mechanism (2 conditions × r = 3 → 6 runs)

**Purpose:** Isolate the effect of reward-weighted partner selection (β_sel > 0) against the
uniform baseline.

**Fixed:** `SBM_P_INTRA=0.7`, `SBM_P_INTER=0.02`, `OPINION_BETA=5.0`, `GRAPH_DYNAMIC=0`,
topic=restrictive

| Run | `SIM_RESPONDER_SELECTION_BETA` | r |
|---|---|---|
| 3-01 | 0.0 | 3 |
| 3-02 | 2.0 | 3 |

> At the polarized anchor, β_sel > 0 should raise the *bi-polarization fraction* and deepen
> conviction (echo-chamber amplification). Because the baseline regime is stochastic
> (~50/50), the comparison is between regime fractions and mean |q_gap| at convergence,
> a single pair of runs could differ by luck alone, hence r = 3 per condition minimum; extend
> to 5 if the fractions are not clearly separated. The 3-01 replicates double as the
> fixed-graph baseline for Stage 4.

---

## Stage 4, Dynamic graph (1 new condition × r = 3 → 3 runs)

**Purpose:** Test endogenous tie rewiring against the fixed-graph baseline.

**Fixed:** `SBM_P_INTRA=0.7`, `SBM_P_INTER=0.02`, `OPINION_BETA=5.0`,
`RESPONDER_SELECTION_BETA=0.0`, topic=restrictive

| Run | `SIM_GRAPH_DYNAMIC` | r | Note |
|---|---|---|---|
| 4-01 | 0 | - | **Reuses Stage 3-01 replicates** (identical configuration), no new runs |
| 4-02 | 1 | 3 | |

> Expected mechanics unchanged: at P_INTRA=0.7 / P_INTER=0.02, ~65 edges, avg degree ≈ 6.5,
> ~3.8 directed activations per ordered pair over 500 interactions; the ~2 cross-community
> edges likely sever after ~2 negative-mean activations (STRENGTH_DELTA=0.5). Expected result:
> structural stratification raises the bi-polarization fraction above the static baseline.
> Structural DVs (modularity, cross-community edge survival) are compared as distributions
> over replicates.

---

## Stage 5, Topic framing / stance-flip control (3 points × 1 framing × r = 3 → 9 runs)

**Purpose:** Stance-flip control for the external-field hypothesis: rerun key Stage 1 points
under the **permissive** framing (+1 now = openness). If regime fractions and, critically,
consensus *directions* follow the topic wording rather than mirroring, the LLM prior is acting
as a directional field (Chuang et al. 2024); if results mirror, the Stage 1 dynamics are
attributable to the feedback mechanism.

**Fixed:** `SBM_P_INTRA=0.7`, `OPINION_BETA=5.0`, `RESPONDER_SELECTION_BETA=0.0`,
`GRAPH_DYNAMIC=0`, topic=permissive (requires `SIM_TOPIC_TEXT`)

The restrictive arm is **not** rerun, Stage 1's replicates at the same P_INTER values *are*
the comparison arm. Three points suffice for the control: the polarized anchor, the empirical
transition point, and the ER null.

| Run | `SIM_SBM_P_INTER` | r | Compared against |
|---|---|---|---|
| 5-01 | 0.02 | 3 | Stage 1-01 |
| 5-02 | transition point (Gate B) | 3 | Stage 1 at same point |
| 5-03 | 0.70 (ER null) | 3 | Stage 1-05 |

> The paired ER nulls (1-05 vs 5-03, r = 5 + 3) directly measure the raw directional prior
> independent of any network effect: under a purely mechanism-driven model, consensus direction
> is a coin flip in both arms; a field flips with the wording. Mirror-symmetry of the regime
> fractions at 0.02 and the transition point tests whether the *location* of the transition is
> framing-dependent.

---

## Run summary

| Stage | Description | Runs | Interactions/run | Total interactions |
|---|---|---|---|---|
| 0 | Pipeline validation | 1 | 30 *(exception)* | 30 |
| 0b | Reward-classifier audit (offline) | 0 | - | ~500 classifier calls |
| 1 | Phase transition scan (5 pts, r = 3-5, adaptive) | 23-25 | 500 | 11 500-12 500 |
| 2 | Opinion beta sensitivity (3 × r3) | 9 | 500 | 4 500 |
| 3 | Responder selection (2 × r3) | 6 | 500 | 3 000 |
| 4 | Dynamic graph (baseline reused, 1 × r3) | 3 | 500 | 1 500 |
| 5 | Stance-flip control (3 × r3, permissive only) | 9 | 500 | 4 500 |
| **Total** | | **51-53** | | **25 030-26 030** |

Versus the previous single-run design (29 runs / 14 530 interactions): ~1.7-1.8× the budget,
but the previous design could not support any regime-fraction claim, at low coupling a single
run lands in consensus half the time under the *null* hypothesis, so per-cell replication is
the minimum admissible design, not a luxury. Savings funding the replicates: Stage 1 grid
7→5 points (adaptive midpoints instead), Stage 2 4→3 β values, Stage 4 baseline reused from
Stage 3, Stage 5 restrictive arm reused from Stage 1 and grid 7→3 points.
