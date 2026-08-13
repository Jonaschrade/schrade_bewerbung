# Candidate discussion topics

Alternatives to the current immigration topic (`config.py` → `TOPIC_LABEL` / `TOPIC_TEXT`),
collected after the 2026-07-03 β=0 harvest located a strong directional prior in the
*generation* channel for the restrictive-immigration wording (contra-restriction fidelity 43%
vs. pro-restriction 84%, p < 10⁻⁸; see `test_schedule.md` § "Topic wording"). This failure
mode is documented in the literature: LLM debate agents systematically drift from their
assigned persona toward the model's inherent stance (Taubenfeld et al. 2024), and networked
LLM-agent simulations converge toward the model's prior rather than the seeded opinion
distribution (Chuang et al. 2024). If Stage 1 is to run on a (near-)symmetric topic rather
than under a measured persona field, one of the candidates below replaces the immigration
wording. Whichever topic is chosen, the prior is **measured, not assumed**; see the
selection procedure at the end.

## Selection criteria

A topic must satisfy all five; they are ordered by how expensive a violation is to repair.

1. **Crisp binary endorsement.** The proposition must map cleanly onto the ±1 stance
   dimension (`config.py` § "Discussion topic"): a single opinion object, endorse/reject,
   stable semantic interpretation across all rounds. Propositions with graded or
   multi-dimensional answers ("how much", "under which conditions") blur the Q-value
   semantics.
2. **Low latent valence (directional prior).** The LLM must not carry a strong
   RLHF-instilled lean toward one stance. A left-libertarian lean is consistently
   documented across model families and instruments (Santurkar et al. 2023; Hartmann et
   al. 2023; Feng et al. 2023; Motoki et al. 2024; Rozado 2024) and is strongest on
   culture-war identity topics (immigration, guns, abortion); it is weakest where **both
   stances have socially-desirable framings**, so the model has no single "safe" answer to
   converge on. Note also that forced-choice survey instruments overstate the stability of
   these leans, measured stances shift with framing and context (Röttger et al. 2024a),
   which is a further argument for measuring the prior *in the simulation's own generation
   setting* rather than importing a political-compass score. The pilot showed the prior
   lives mainly in the *generation* channel (persona fidelity asymmetry), not the reward
   channel, so argument quality and confidence must be comparable on both sides, or the
   asymmetry re-enters as a polarization confound no reward-classifier fix can remove.
3. **Guardrail-safe.** Neither stance may trigger safety hedging ("this is a complex issue
   affecting real people…"). Exaggerated safety behaviour, refusing or hedging on benign
   prompts that lexically resemble unsafe ones, is a documented, systematic LLM failure
   mode (Röttger et al. 2024b). Hedging is asymmetric, it attaches to one stance, and
   breaks symmetry at *generation*, upstream of every mechanism the thesis studies. The
   pilot's harsher asylum wording tripped exactly this. Rule of thumb: policy questions
   about *systems* (energy, work, transport, voting) are safe; questions about *groups of
   people* are not.
4. **Sufficient affective charge.** Agents need substantive, genuinely contested material
   for conviction dynamics. A perfectly neutral but bland topic (daylight saving time)
   buys valence symmetry at the cost of nothing to polarize over.
5. **Flippable wording (Stage 5 control).** A natural mirrored proposition must exist
   (cf. restrictive/permissive immigration pair), so the stance-flip control can test
   whether dynamics follow the wording or the mechanism.

## Tier 1, recommended

### Nuclear power expansion

- **Default wording:** *"The United States should significantly expand nuclear power to meet
  its electricity demand."*
- **Flipped wording (Stage 5):** *"The United States should phase out nuclear power in favor
  of other energy sources."*
- **Binary (1):** clean, expand vs. don't expand, one opinion object.
- **Valence (2):** the key advantage. Both stances have progressive *and* conservative
  framings, pro: decarbonization, climate urgency, energy independence, baseload
  reliability; contra: waste, catastrophic risk, cost overruns, renewables-first. No single
  socially-desirable answer, so argument quality and confidence come out comparable on both
  sides. Real-world anchor points exist for both poles (France's fleet vs. Germany's
  Atomausstieg). *Caveat:* recent models may carry a **mild pro-nuclear lean**, US public
  support for expanding nuclear power rose from 43% (2020) to ~59% (2025), with majorities
  in both parties (Pew Research Center 2025), and training data will reflect that discourse
  shift, much weaker and less consistent than the immigration prior, but not guaranteed
  zero; the audit decides. (Hartmann et al. 2023's "pro-environmental" ChatGPT finding cuts
  the *other* way for nuclear, which is itself evidence the framings compete.)
- **Guardrails (3):** zero friction, energy policy, not people.
- **Charge (4):** high, genuinely contested with real stakes; decades of live political
  conflict to draw arguments from.
- **Flip (5):** natural expand/phase-out pair.

## Tier 2, viable runner-ups

### Universal Basic Income

- **Default:** *"The United States should introduce a universal basic income for all
  adults."* / **Flipped:** *"…should reject proposals for a universal basic income."*
- **Binary (1):** weakest point, "introduce" invites hedged intermediate positions (pilot
  programs, partial schemes) that muddy the ±1 endorsement.
- **Valence (2):** well-balanced argument space (automation-proofing, poverty floor,
  bureaucracy reduction vs. cost, inflation, work incentives); cross-cutting support
  (libertarian *and* left variants) mutes a uniform prior. Possible mild left lean.
- **Guardrails (3):** none. **Charge (4):** high. **Flip (5):** workable but the negation
  is rhetorically weaker than a true mirror.

### Mandated four-day work week

- **Default:** *"The United States should mandate a four-day work week by law."* /
  **Flipped:** *"…should leave the length of the work week entirely to employers."*
- **Binary (1):** crisp, the "by law" clause forces the divisive question (mandate vs.
  market) rather than the anodyne one (is leisure nice).
- **Valence (2):** mild left/labor lean expected; the mandate framing pulls in genuine
  contra material (small-business burden, wage effects, sectoral impossibility).
- **Guardrails (3):** none. **Charge (4):** medium-high, personally relatable to any
  persona. **Flip (5):** good natural mirror.

### Compulsory voting

- **Default:** *"The United States should make voting in federal elections compulsory."* /
  **Flipped:** *"…should keep voting in federal elections strictly voluntary."*
- **Binary (1):** crisp. **Valence (2):** genuinely low, democratic-participation framing
  (pro) vs. individual-liberty framing (contra) are *both* socially desirable; real-world
  anchors on both sides (Australia/Belgium vs. US/Germany). Probably the lowest-prior
  candidate on the list.
- **Guardrails (3):** none. **Charge (4):** the risk, medium at best; less
  emotionally loaded than nuclear or work-week, conviction dynamics may run shallow.
  **Flip (5):** excellent natural mirror.

### Urban congestion pricing

- **Default:** *"Large American cities should charge drivers a fee to enter their city
  centers."* / **Flipped:** *"…should keep city-center access free for all drivers."*
- **Binary (1):** crisp. **Valence (2):** low, clean trade-off structure (traffic, air
  quality, transit funding vs. regressive cost, commuter burden); live and genuinely
  divisive (NYC 2024-25). **Guardrails (3):** none. **Charge (4):** the weakness,
  medium-low unless personas are urban; may be too technocratic to sustain 25 rounds of
  conviction accumulation. **Flip (5):** good.

## Rejected (with reasons)

| Topic | Failing criterion |
|---|---|
| Illegal immigration (current) | (2)+(3): measured generation-channel prior (43/84 fidelity split); harsher wordings trip guardrails. Retained in `config.py` only under the "measured persona field" reading of Stage 1. |
| Abortion, gun control, death penalty | (2)+(3): strongest documented RLHF priors *and* highest hedging risk, the worst quadrant. |
| GM food, vaccine mandates | (2): strong pro-science prior; contra stance generated with visibly lower confidence. |
| Zoo bans, animal testing | (2): animal-welfare prior; contra stance drifts toward concession. |
| Daylight saving time, homework abolition | (4): valence-symmetric but bland, nothing to polarize over. |
| Congressional term limits | (2)/(4): lopsided real-world agreement (~80% support), the contra stance has no stable social base to roleplay, so drift mimics the immigration failure mode. |
| AI regulation | (3): self-referential for an LLM; hedging and meta-commentary risk at generation. |

## Selection procedure (measure, don't assume)

The infrastructure to decide empirically already exists; run it over the shortlist instead
of trusting intuition (mine included):

1. **Wire the wording** via the `SIM_TOPIC_TEXT` / `SIM_TOPIC_LABEL` env-var override
   (`test_schedule.md` § "Required infrastructure changes").
2. **Reward channel:** the `audit_reward.py` reward-symmetry harness has been retired (see
   `test_schedule.md` Stage 0b); if the reward channel needs re-auditing under a candidate
   wording, re-add a small log-mining reward-by-dyad check.
3. **Generation channel (the decisive test):** run a β=0 harvest with the gate enabled, then
   `analyze_gate.py`, the per-stance *first-attempt* fidelity split is the direct measurement
   of the latent prior. Pick the candidate with the smallest |pro − contra| fidelity asymmetry.
4. **Cross-checks:** early-round expressed-stance split and ER-null consensus-direction
   tally (standing diagnostics in `test_schedule.md`).
5. **Confirm charge (criterion 4)** hasn't been traded away: utterances should still show
   substantive argument, not agreeable filler, spot-check the harvest transcripts.
   (Agreeable filler is not hypothetical: RLHF-trained assistants exhibit systematic
   sycophancy, preferring responses that match the interlocutor's view, Sharma et al.
   2024, which in a deliberation setting masquerades as convergence.)

This turns topic choice from a judgment call into a measured design decision, and yields a
defensible methods-chapter sentence either way: either a near-symmetric topic for the clean
replication, or a quantified persona field read against both baselines.

## References

- Chuang, Y.-S., Goyal, A., Harlalka, N., Suresh, S., Hawkins, R., Yang, S., Shah, D.,
  Hu, J., & Rogers, T. T. (2024). Simulating Opinion Dynamics with Networks of LLM-based
  Agents. *Findings of the Association for Computational Linguistics: NAACL 2024*,
  3326-3346. <https://aclanthology.org/2024.findings-naacl.211/>
  (arXiv:[2311.09618](https://arxiv.org/abs/2311.09618))
- Feng, S., Park, C. Y., Liu, Y., & Tsvetkov, Y. (2023). From Pretraining Data to Language
  Models to Downstream Tasks: Tracking the Trails of Political Biases Leading to Unfair
  NLP Models. *Proceedings of ACL 2023*.
  (arXiv:[2305.08283](https://arxiv.org/abs/2305.08283))
- Hartmann, J., Schwenzow, J., & Witte, M. (2023). The political ideology of
  conversational AI: Converging evidence on ChatGPT's pro-environmental, left-libertarian
  orientation. (arXiv:[2301.01768](https://arxiv.org/abs/2301.01768))
- Motoki, F., Pinho Neto, V., & Rodrigues, V. (2024). More human than human: measuring
  ChatGPT political bias. *Public Choice*, 198(1-2), 3-23.
  <https://doi.org/10.1007/s11127-023-01097-2>
- Pew Research Center (2025). Support for expanding nuclear power is up in both parties
  since 2020.
  <https://www.pewresearch.org/short-reads/2025/10/16/support-for-expanding-nuclear-power-is-up-in-both-parties-since-2020/>
- Röttger, P., Hofmann, V., Pyatkin, V., Hinck, M., Kirk, H., Schütze, H., & Hovy, D.
  (2024a). Political Compass or Spinning Arrow? Towards More Meaningful Evaluations for
  Values and Opinions in Large Language Models. *Proceedings of ACL 2024*, Best Paper
  Award. <https://aclanthology.org/2024.acl-long.816/>
  (arXiv:[2402.16786](https://arxiv.org/abs/2402.16786))
- Röttger, P., Kirk, H., Vidgen, B., Attanasio, G., Bianchi, F., & Hovy, D. (2024b).
  XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language
  Models. *Proceedings of NAACL 2024*.
  (arXiv:[2308.01263](https://arxiv.org/abs/2308.01263))
- Rozado, D. (2024). The political preferences of LLMs. *PLOS ONE*, 19(7), e0306621.
  <https://doi.org/10.1371/journal.pone.0306621>
- Santurkar, S., Durmus, E., Ladhak, F., Lee, C., Liang, P., & Hashimoto, T. (2023). Whose
  Opinions Do Language Models Reflect? *Proceedings of ICML 2023*.
  (arXiv:[2303.17548](https://arxiv.org/abs/2303.17548))
- Sharma, M., Tong, M., Korbak, T., Duvenaud, D., Askell, A., Bowman, S. R., et al.
  (2024). Towards Understanding Sycophancy in Language Models. *ICLR 2024*.
  (arXiv:[2310.13548](https://arxiv.org/abs/2310.13548))
- Taubenfeld, A., Dover, Y., Reichart, R., & Goldstein, A. (2024). Systematic Biases in
  LLM Simulations of Debates. *Proceedings of EMNLP 2024*.
  (arXiv:[2402.04049](https://arxiv.org/abs/2402.04049))
