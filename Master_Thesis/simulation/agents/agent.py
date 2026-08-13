"""
Agent: synthesises persona, memory, and reflection into conversational behaviour.

Mechanism and rationale (expression gate, persona field, stance tagging) live in
the README ("SFT architecture", "Agent primitives", "SFT expression gate").

Public API:
    respond(message, speaker, softmax_stance) -> (str, dict)
        Generate an opinion-bearing reply anchored to softmax_stance (+1/−1)
        through the SFT expression gate (Step B, _sft_gate). Returns the
        accepted reply and a gate record; the caller indexes the Q-update on
        final_expressed and skips it when that is 0 (abstention). With
        SFT_GATE_ENABLED = False the gate degrades to one ungated draft (Step A),
        same return shape.
    classify_expression_graded(text) -> float | None
        Context-free classifier: text → graded TOPIC_TEXT stance in [−1, +1]
        (+ favors, − opposes, 0.0 = genuine ambivalent, None = UNRELATED /
        unparseable). The single classifier in the system: its sign arbitrates
        the gate and tags memories; its magnitude feeds the reward product.
        JSON-schema-constrained ordinal label.
    classify_expression(text) -> int | None
        Sign of classify_expression_graded(): +1/−1/0, or None. Thin wrapper for
        callers that need only the ternary stance (memory tagging).
    reflect(argmax_stance)
        Synthesise insights from recent memories, anchored to argmax_stance.
        Stores each insight tagged with its classified stance. Not gated.

The social reward is not a classifier but the product r = graded(expresser) *
graded(reaction) (reward_from_expressions); see network/discussion.py.
"""

import json
from typing import TYPE_CHECKING

from langchain_ollama import OllamaLLM

from config import (
    MAX_MEMORIES_SEED,
    SFT_GATE_ENABLED,
    SFT_GATE_MAX_ATTEMPTS,
    TOPIC_TEXT,
)
from memory.store import MemoryStore

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

# sentence_transformers is imported lazily in _get_st_model() so that
# importing this module stays light for consumers that never embed
# (e.g. offline log analysis).
_st_model = None

# classify_expression: a single graded, TOPIC_TEXT-anchored stance scale mapped
# to a numeric value in [-1, 1]. Ordinal but not uniformly spaced (CLEAR <->
# PREDOMINANT is a larger step than PREDOMINANT <-> SLIGHT <-> AMBIVALENT).
# AMBIVALENT -> 0.0 is a genuine neutral signal (sign 0); UNRELATED -> None is a
# distinct "no signal" case (the text does not engage the topic at all). The
# concession-counting rubric that anchors each label lives in the prompt and
# keeps the intermediate labels reachable (LLM judges under unanchored scales
# collapse to the endpoints; see the README classify_expression section).
#
# Reward is no longer a separate LLM judgement: r = graded(expresser) *
# graded(reaction), the o_i * o_j product form (see reward_from_expressions and
# the README "reward as a product of graded expressions" section).
_GRADED_STANCE_LABELS: dict[str, float | None] = {
    "CLEAR_FAVOR": 1.0,
    "PREDOMINANT_FAVOR": 0.6,
    "SLIGHT_FAVOR": 0.3,
    "AMBIVALENT": 0.0,
    "SLIGHT_OPPOSITION": -0.3,
    "PREDOMINANT_OPPOSITION": -0.6,
    "CLEAR_OPPOSITION": -1.0,
    "UNRELATED": None,
}


def _stance_sign(graded: float | None) -> int | None:
    """Collapse a graded stance to the +1/-1/0 convention (None stays None).

    The sign is what the expression gate arbitrates on and what indexes the
    Q-update and tags memories; the graded magnitude only feeds the reward
    product. AMBIVALENT (0.0) -> 0 (genuine abstention); UNRELATED / parse
    failure (None) -> None (no stance discernible).
    """
    if graded is None:
        return None
    if graded > 0:
        return 1
    if graded < 0:
        return -1
    return 0


def reward_from_expressions(
    graded_expr: float | None, graded_resp: float | None
) -> float | None:
    """Social reward r = graded(expresser) * graded(response), in [-1, 1].

    The o_i * o_j product form: two same-side stances reinforce (positive),
    opposite sides penalise (negative), and each factor's magnitude scales the
    signal by how firmly that turn committed to the topic. Symmetric by
    construction -- both factors come from the same TOPIC_TEXT-anchored classifier,
    so mutual opposition and mutual favor are the identical classification problem
    (the former relational classify_reward misread mutual rejection as
    disagreement; see the README).

    Returns None when the reaction is UNRELATED/unparseable (graded_resp is
    None) so the caller skips the TD update; a genuine AMBIVALENT reaction (0.0)
    yields 0.0, a real neutral signal that still moves Q. graded_expr is never
    None in practice (a gated turn always resolves to a sign, and the ungated arm
    collapses None to 0.0).
    """
    if graded_expr is None or graded_resp is None:
        return None
    return graded_expr * graded_resp


def _get_st_model() -> "SentenceTransformer":
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cpu")
    return _st_model


def _stance_sentence(stance: int) -> str:
    """Anchoring sentence for a stance, phrased against config.TOPIC_TEXT."""
    topic = TOPIC_TEXT[0].lower() + TOPIC_TEXT[1:]
    return f"You favor that {topic}." if stance == 1 else f"You oppose that {topic}."


def _final_fallback_reply(softmax_stance: int) -> str:
    """Fixed on-stance reply: stage 3 of the gate's fallback cascade.

    Deterministic template from softmax_stance and TOPIC_TEXT; no LLM call, no
    classification. Fires only when both LLM stages exhaust their budgets, so no
    counter-stance text can leave the gate. Tagged on-stance without re-classifying.
    """
    position = "favor" if softmax_stance == 1 else "oppose"
    return (
        f"I have thought about this carefully, and I {position} the idea that "
        f"{TOPIC_TEXT[0].lower() + TOPIC_TEXT[1:]}. That is where I currently stand."
    )


def _gate_record(
    attempts: list[dict], softmax_stance: int, fallback: bool, final: bool
) -> dict:
    """Assemble the per-turn gate record from the accumulated draft attempts.

    Persona-field telemetry unit, attached to each turn in the discussion log
    and mined post-hoc by analyze_gate.py (see README "The gate record").

    Keys:
        n_attempts:              total drafts generated (incl. the accepted one)
        passed:                  final text on-stance or ambiguous (always True
                                 under the gate; in the ungated arm flags whether
                                 the single draft was stance-conform)
        fallback_used:           cascade reached stage 2 (context-free prompt)
        final_fallback_used:     cascade reached stage 3 (fixed template)
        first_attempt_expressed: expressed stance sign of draft 1 (fidelity basis);
                                 may be None if draft 1 was unparseable
        final_expressed:         expressed stance sign of the accepted text
                                 (Q-update index; skipped when 0). Never None
        final_expressed_graded:  graded stance of the accepted text in [-1, 1]
                                 (reward factor). Never None under the gate; may
                                 be a genuine 0.0 (ambivalent accepted text)
        attempts:                [{stage, text, expressed, expressed_graded
                                 [, fail_reason]}, ...]; stage in {main, fallback,
                                 final}; expressed is the sign of expressed_graded;
                                 fail_reason in {flip, unparseable} on rejected
                                 drafts
    """
    return {
        "n_attempts": len(attempts),
        "passed": attempts[-1]["expressed"] in (softmax_stance, 0),
        "fallback_used": fallback,
        "final_fallback_used": final,
        "first_attempt_expressed": attempts[0]["expressed"],
        "final_expressed": attempts[-1]["expressed"],
        "final_expressed_graded": attempts[-1]["expressed_graded"],
        "attempts": attempts,
    }


def _invoke_labeled(
    llm: OllamaLLM, prompt: str, labels: dict[str, object], context: str
) -> str | None:
    """Invoke llm under a JSON-schema-constrained label choice.

    Passes a JSON Schema as Ollama's ``format`` parameter so the reply is always
    a JSON object with a ``label`` drawn from ``labels``, removing the free-text
    parsing-failure class entirely. The schema also asks for a one-sentence
    ``reasoning`` field, since brief reasoning before committing improves
    accuracy. A malformed reply is still possible, so the call is retried once.

    Args:
        llm: OllamaLLM instance to invoke.
        prompt: Full instruction, ending in the label choice.
        labels: Allowed label set; only its keys build the schema and validate.
        context: Caller name, prefixed to the warning for log grep-ability.

    Returns:
        Chosen label string, or None if no valid label after one retry.
    """
    schema = {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "label": {"type": "string", "enum": list(labels)},
        },
        "required": ["reasoning", "label"],
    }
    raw = ""
    for _ in range(2):
        raw = llm.invoke(prompt, format=schema)
        try:
            label = json.loads(raw)["label"]
        except (json.JSONDecodeError, KeyError, TypeError):
            label = None
        if label in labels:
            return label

    print(
        f"\n  ⚠️  {context}: no valid label in reply after retry, "
        f"falling back. Raw text (tail): {raw[-150:]!r}"
    )
    return None


class Agent:
    def __init__(self, name: str, persona: str, llm: OllamaLLM):
        """Initialise an agent with a fixed identity, persona, and isolated memory store.

        Args:
            name:    Display name used in prompts and as the ChromaDB collection key.
            persona: Role description; reaches generation only through the cached
                     stance bridges (see _mold_stance), never injected directly.
            llm:     Shared OllamaLLM instance (all agents may share one).
        """
        self.name = name
        self.persona = persona
        self.llm = llm
        self.memory = MemoryStore(name)

        # Persona seeding into memory is disabled: a "persona"-typed memory would
        # re-enter the respond()/reflect() prompts via retrieval, bypassing the
        # _mold_stance bridge (see README "Agent internal state").

        # self._store(persona, mem_type="persona")
        self._stance_bridges = self._derive_stance_bridges()

    # private helpers --------------------------------------------------------

    def _mold_stance(self, stance: int) -> str:
        """Mold the persona into a stance justification.

        Single point where persona and stance meet. respond()/reflect() only ever
        see the resulting sentence via _stance_bridges, never self.persona.
        Deliberately soft (Step A): the persona may still override the anchored
        stance at generation time (see README "Agent internal state").

        Args:
            stance: +1 (pro) or −1 (contra).

        Returns:
            A persona-grounded justification sentence for stance.
        """
        position = (
            f"favor that {TOPIC_TEXT[0].lower()+TOPIC_TEXT[1:]}"
            if stance == 1 else
            f"oppose that {TOPIC_TEXT[0].lower()+TOPIC_TEXT[1:]}"
        )
        return self.llm.invoke(
            f"You are {self.name}. {self.persona}\n\n"
            f"People often hold positions that do not fit seamlessly with all of their other "
            f"values which is human and no contradiction that needs pointing out.\n\n"
            f"In 2-3 sentences, state why you, as {self.name}, {position}. "
            f"Ground it in your character and life background but do not soften, hedge, or "
            f"argue against the position itself.\n\n"
            f"Reply with only the justification in the first person. No introduction, "
            f"no meta-commentary or remarks about the role prompt, no parenthetical asides."
        ).strip()

    def _derive_stance_bridges(self) -> dict[int, str]:
        """Cache a molded bridge sentence for each possible stance, at init."""
        return {1: self._mold_stance(1), -1: self._mold_stance(-1)}

    def _embed(self, text: str) -> list:
        """Return a vector embedding for text using sentence-transformers."""
        return _get_st_model().encode(text).tolist()

    def _score_importance(self, text: str) -> float:
        """Ask the LLM to rate how significant a memory is on a 1-10 scale.
        Returns numeric rating, or 5.0 if the response cannot be parsed.
        """
        raw = self.llm.invoke(
            f"Rate the significance of this memory for {self.name} "
            f"on a scale from 1 (trivial) to 10 (highly significant). "
            f"Reply with only a single number.\nMemory: {text}"
        ).strip()
        try:
            return float(raw.split()[0])
        except (ValueError, IndexError):
            return 5.0

    def _store(self, content: str, mem_type: str = "interaction", stance: int = 0) -> None:
        """Compute importance and embedding for content, then persist it.

        Args:
            stance: Stance this memory was formed under (+1, −1, or 0 =
                    neutral/unknown). Tags the memory for later stance-congruent
                    retrieval biasing; see memory/scoring.py.
        """
        importance = self._score_importance(text=content)
        self.memory.add(
            content=content,
            mem_type=mem_type,
            importance=importance,
            embedding=self._embed(text=content),
            stance=stance,
        )

    def _retrieve(self, context: str, stance: int | None = None) -> list[dict]:
        """Retrieve memories most relevant to context using embedding similarity.

        Args:
            stance: If given, biases retrieval toward memories tagged with
                    this stance (see memory/scoring.py). None leaves
                    retrieval stance-unbiased.
        """
        return self.memory.retrieve(self._embed(text=context), query_stance=stance)

    # reflection -------------------------------------------------------------

    def reflect(self, argmax_stance: int) -> None:
        """Two-step reflection loop inspired by Park et al. (2023).

        Step 1: Generate 2 questions worth reflecting on from the most recent
                MAX_MEMORIES_SEED raw memories.
        Step 2: For each question, retrieve the most relevant memories and ask
                the LLM to synthesise a single insight.

        Args:
            argmax_stance: The agent's deterministic Q-argmax stance (+1 or
                           −1). Consolidated opinion entering the
                           reflection, not a per-interaction softmax draw.
        """
        recent = self.memory.all_recent(limit=MAX_MEMORIES_SEED)
        if not recent:
            return
        recent_block = "\n".join(f"- {m}" for m in recent)

        stance = _stance_sentence(argmax_stance)
        stance_bridge = self._stance_bridges[argmax_stance]

        # Step 1: identify meaningful questions from recent experience
        questions_raw = self.llm.invoke(
            f"You are {self.name} and currently hold this position:\n"
            f"{stance}\n{stance_bridge}\n\n"
            f"Recent experiences:\n{recent_block}\n\n"
            f"Name 2 meaningful questions you should reflect on, one per line. "
            f"No numbering, no introduction."
        )
        questions = [
            q.strip() for q in questions_raw.strip().splitlines() if q.strip()
        ][:2]

        # Step 2: synthesise an insight for each question
        for question in questions:
            mems = self._retrieve(question, stance=argmax_stance)
            mem_block = "\n".join(f"- {m['content']}" for m in mems)

            insight = self.llm.invoke(
                f"You are {self.name} and currently hold this position:\n"
                f"{stance}\n{stance_bridge}\n\n"
                f"Question: {question}\n"
                f"Relevant memories:\n{mem_block}\n\n"
                f"State a single insight or conclusion in the first person."
            ).strip()

            # Reflection is not gated (no retry), so an unparseable
            # classification (None) collapses to a neutral (0) stance tag.
            insight_stance = self.classify_expression(insight)
            self._store(
                f"[Reflection] {insight}",
                mem_type="reflection",
                stance=insight_stance if insight_stance is not None else 0,
            )
            print(f"\n 💭 {self.name} reflects: {insight}")

    # respond / expression gate -----------------------------------------------

    def _generate_reply(
        self, message: str, speaker: str, softmax_stance: int, mems: list[dict]
    ) -> str:
        """Generate one reply draft from the soft-anchored prompt."""
        mem_block = (
            "\n".join(f"[{m['type']}] {m['content']}" for m in mems)
            if mems else "(none yet)"
        )
        stance = _stance_sentence(softmax_stance)
        stance_bridge = self._stance_bridges[softmax_stance]

        return self.llm.invoke(
            f"You are {self.name}.\n\n"
            f"Relevant memories:\n{mem_block}\n\n"
            f"{speaker} says: \"{message}\"\n\n"
            f"You currently hold this position:\n"
            f"{stance}\n{stance_bridge}\n\n"
            f"Reply as {self.name} to {speaker}'s statement. Express your position "
            f"stated above and justify it from your personal experience and character. "
            f"Answer in 2-3 sentences."
        ).strip()

    def _fallback_reply(self, message: str, speaker: str, softmax_stance: int) -> str:
        """Generate one stage-2 draft: _generate_reply minus context.

        Identical prompt except the memory block and the persona-derived stance
        bridge are dropped; no escalation in wording.
        """
        stance = _stance_sentence(softmax_stance)
        return self.llm.invoke(
            f"You are {self.name}.\n\n"
            f"{speaker} says: \"{message}\"\n\n"
            f"You currently hold this position:\n{stance}\n\n"
            f"Reply as {self.name} to {speaker}'s statement. Express your position "
            f"stated above and justify it from your personal experience and character. "
            f"Answer in 2-3 sentences."
        ).strip()

    def _sft_gate(self, message: str, speaker: str, softmax_stance: int) -> tuple[str, dict]:
        """Generate a reply expressing softmax_stance, by rejection sampling.

        Three cascade stages (see README "The mechanism"):
            1. main:     up to SFT_GATE_MAX_ATTEMPTS soft-prompt drafts (_generate_reply).
            2. fallback: up to SFT_GATE_MAX_ATTEMPTS context-free drafts (_fallback_reply).
            3. final:    fixed on-stance template (_final_fallback_reply), no LLM call.
        A draft passes iff the sign of classify_expression_graded() equals the
        drawn stance or 0 (genuine ambivalent). A flip (opposite sign, fail_reason
        "flip") or a None (UNRELATED / unparseable, fail_reason "unparseable") is
        rejected and retried; neither may enter the interaction. The gate is thus
        anchored to the sign; the graded magnitude is carried through untouched to
        feed the reward product. Each stage bails early if a draft repeats
        verbatim. Memory retrieval runs once per gate call, not per attempt.

        Returns:
            (accepted_text, gate_record): see _gate_record. Accepted text
            expresses the drawn stance or is ambiguous, never the opposite.
        """
        mems = self._retrieve(message, stance=softmax_stance)
        attempts: list[dict] = []
        draft_no = 0

        prev_text = None
        for _ in range(SFT_GATE_MAX_ATTEMPTS):                    # stage 1: soft prompt
            draft_no += 1
            print(f"    ↻ {self.name} draft {draft_no} (main)")
            text = self._generate_reply(message, speaker, softmax_stance, mems)
            graded = self.classify_expression_graded(text)
            expressed = _stance_sign(graded)
            entry = {"stage": "main", "text": text,
                     "expressed": expressed, "expressed_graded": graded}
            if expressed in (softmax_stance, 0):                  # on-stance sign or genuine ambivalent
                attempts.append(entry)
                return text, _gate_record(attempts, softmax_stance, fallback=False, final=False)
            # opposite sign, or None (unparseable/unrelated): reject and retry. An
            # off-stance or topic-disengaged draft must not enter the interaction.
            entry["fail_reason"] = "unparseable" if expressed is None else "flip"
            attempts.append(entry)
            if text == prev_text:
                break
            prev_text = text

        prev_text = None
        for _ in range(SFT_GATE_MAX_ATTEMPTS):                    # stage 2: context-free fallback
            draft_no += 1
            print(f"    ↻ {self.name} draft {draft_no} (fallback)")
            text = self._fallback_reply(message, speaker, softmax_stance)
            graded = self.classify_expression_graded(text)
            expressed = _stance_sign(graded)
            entry = {"stage": "fallback", "text": text,
                     "expressed": expressed, "expressed_graded": graded}
            if expressed in (softmax_stance, 0):
                attempts.append(entry)
                return text, _gate_record(attempts, softmax_stance, fallback=True, final=False)
            entry["fail_reason"] = "unparseable" if expressed is None else "flip"
            attempts.append(entry)
            if text == prev_text:
                break
            prev_text = text

        text = _final_fallback_reply(softmax_stance)              # stage 3: fixed template
        # Fixed on-stance template: a clean, full-conviction stance declaration,
        # tagged at graded +-1.0 without re-classifying (mirrors its +-1 sign).
        attempts.append({"stage": "final", "text": text,
                         "expressed": softmax_stance,
                         "expressed_graded": float(softmax_stance)})
        return text, _gate_record(attempts, softmax_stance, fallback=True, final=True)

    def respond(
        self,
        message: str,
        speaker: str,
        softmax_stance: int,
    ) -> tuple[str, dict]:
        """Generate a gated response to message from speaker.

        Retrieves memories, generates from the soft-anchored prompt, and enforces
        the softmax-drawn stance output-side via the expression gate (_sft_gate).
        With SFT_GATE_ENABLED = False, generation is a single ungated pass wrapped
        in a one-attempt gate record, same return shape (ablation arm). Both arms
        feed the same Q-update rule: the caller indexes on final_expressed and
        skips when it is 0 (a strategic abstention). The interaction is stored as
        a memory tagged with final_expressed. See README "Agent primitives" and
        "SFT expression gate".

        Args:
            message:        The incoming message text.
            speaker:        Agent or moderator who sent the message.
            softmax_stance: SFT stance drawn by draw_softmax_stance(): +1 or −1.

        Returns:
            (reply, gate_record): accepted reply string plus the gate record
            (see _gate_record).
        """
        if SFT_GATE_ENABLED:
            reply, record = self._sft_gate(message, speaker, softmax_stance)
        else:
            mems = self._retrieve(message, stance=softmax_stance)
            reply = self._generate_reply(message, speaker, softmax_stance, mems)
            # Ungated arm has no retry, so a None classification (UNRELATED /
            # unparseable) collapses to a neutral 0 / 0.0 (treated as an
            # abstention), unlike the gate which would retry.
            graded = self.classify_expression_graded(reply)
            expressed = _stance_sign(graded)
            attempts = [{
                "stage": "main",
                "text": reply,
                "expressed": expressed if expressed is not None else 0,
                "expressed_graded": graded if graded is not None else 0.0,
            }]
            record = _gate_record(attempts, softmax_stance, fallback=False, final=False)

        self._store(
            f"{speaker} said: '{message}'. I replied: '{reply}'",
            stance=record["final_expressed"],
        )
        return reply, record

    # classify_expression -----------------------------------------------------

    def classify_expression_graded(self, text: str) -> float | None:
        """Classify the graded stance a text takes on TOPIC_TEXT, in [-1, 1].

        The single classifier in the system. Three roles: (a) arbiter of the
        expression gate (via its sign), (b) stance tagger for stored memories
        (via its sign), and (c) the two factors of the social reward r =
        graded(expresser) * graded(reaction) (see reward_from_expressions).
        Deliberately minimal (no persona, memory, or transcript context) so
        generation and evaluation stay causally independent, avoiding self-scoring
        contamination (Chuang et al. 2024). Backed by _invoke_labeled() over
        _GRADED_STANCE_LABELS.

        Absolute, not relational: it asks only where *this one text* stands on the
        topic, never how one text reacts to another. That is what makes the reward
        symmetric -- mutual opposition and mutual favor are the identical
        classification problem (the former relational classify_reward misread
        mutual rejection as disagreement; see the README).

        Each label is anchored to a concession-counting rubric and the reasoning
        step enumerates concessions before labeling, keeping the intermediate
        labels reachable (LLM judges under unanchored scales collapse to the
        endpoints). Anchored to TOPIC_TEXT, so classifications are meaningful only
        under the topic the text was generated for.

        Args:
            text: Generated utterance to classify.

        Returns:
            float in [-1, 1] (+ favors, - opposes, 0.0 genuine ambivalent), or
            None when the text is UNRELATED to the topic or unparseable.
        """
        topic = TOPIC_TEXT[0].lower() + TOPIC_TEXT[1:]
        prompt = (
            f"Task: Assess where the following STATEMENT stands on the position "
            f"that {topic}.\n\n"
            f"Judge only the substantive position taken, not the tone, politeness, "
            f"phrasing, or quality of justification.\n"
            f"A politely phrased opposition is still opposition.\n\n"
            f"STATEMENT:\n{text}\n\n"
            f"In your reasoning, first list every concession, qualification, or point "
            f"the STATEMENT grants to the side it leans against; then choose exactly "
            f"one label:\n"
            f"CLEAR_FAVOR            = favors the position and grants the other side nothing\n"
            f"PREDOMINANT_FAVOR      = favors the position but grants the other side at least one point\n"
            f"SLIGHT_FAVOR           = favors only conditionally (\"only if ...\") or "
            f"concedes so much the lean is barely discernible\n"
            f"AMBIVALENT             = mixed or withheld judgement, no discernible lean\n"
            f"SLIGHT_OPPOSITION      = opposes only conditionally (\"unless ...\") or "
            f"concedes so much the lean is barely discernible\n"
            f"PREDOMINANT_OPPOSITION = opposes the position but grants it at least one point\n"
            f"CLEAR_OPPOSITION       = opposes the position and grants it nothing\n"
            f"UNRELATED              = does not address the position or topic at all"
        )
        label = _invoke_labeled(self.llm, prompt, _GRADED_STANCE_LABELS, "classify_expression")
        return _GRADED_STANCE_LABELS[label] if label is not None else None

    def classify_expression(self, text: str) -> int | None:
        """Sign of classify_expression_graded(): the +1/-1/0 stance convention.

        Thin wrapper for callers that only need the ternary stance (memory
        tagging in reflect(); see _stance_sign). Returns +1/-1/0, or None when the
        graded classifier returns None (UNRELATED / unparseable). Non-gate callers
        collapse that None to 0.
        """
        return _stance_sign(self.classify_expression_graded(text))
