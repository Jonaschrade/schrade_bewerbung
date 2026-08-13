"""
Memory retrieval scoring.

Each candidate memory is ranked by a weighted combination of four signals:
    Recency           exponential decay from the time the memory was stored
    Importance        LLM-rated significance (1-10), normalised to [0, 1]
    Relevance         inverted cosine distance to the query embedding
    Stance congruence whether the memory's tagged stance matches the agent's
                      current softmax-drawn stance: a motivated-recall /
                      biased-assimilation effect (see README "Memory system")

Weights are configured in config.py (WEIGHT_RECENCY, WEIGHT_IMPORTANCE,
WEIGHT_RELEVANCE, WEIGHT_STANCE_CONGRUENCE).
"""

import math
import time

from config import (
    WEIGHT_RECENCY,
    WEIGHT_IMPORTANCE,
    WEIGHT_RELEVANCE,
    WEIGHT_STANCE_CONGRUENCE,
)


def score_memory(meta: dict, distance: float, query_stance: int | None = None) -> float:
    """Compute a composite retrieval score for a single memory candidate.

    Args:
        meta:         ChromaDB metadata dict containing:
                        'ts'         (float) Unix timestamp when the memory was stored
                        'importance' (float) LLM-rated significance on a 1-10 scale
                        'stance'     (int)   stance the memory was formed under
                                             (+1, −1, or 0 = neutral/unknown)
        distance:     ChromaDB cosine distance; lower means more similar to the query.
        query_stance: Stance (+1 or −1) to bias retrieval toward (respond()
                      passes its softmax draw, reflect() its argmax stance), or
                      None if retrieval is not stance-conditioned. When None, or
                      when the memory's own stance tag is neutral (0), congruence
                      is a constant 0.5: neither boosted nor penalised.

    Returns:
        A float roughly in [0, 1]; higher values indicate the memory is more
        worth surfacing to the agent.
    """
    age_hours  = (time.time() - meta["ts"]) / 3600
    recency    = math.exp(-0.5 * age_hours)   # faster decay in the first hours
    importance = meta["importance"] / 10       # normalise to [0, 1]
    relevance  = 1 / (1 + distance)           # invert distance → similarity

    mem_stance = meta.get("stance", 0)
    if query_stance is None or mem_stance == 0:
        congruence = 0.5
    elif mem_stance == query_stance:
        congruence = 1.0
    else:
        congruence = 0.0

    return (
        WEIGHT_RECENCY           * recency    +
        WEIGHT_IMPORTANCE        * importance +
        WEIGHT_RELEVANCE         * relevance  +
        WEIGHT_STANCE_CONGRUENCE * congruence
    )
