"""
MemoryStore: per-agent ChromaDB-backed memory.

Each agent gets its own isolated collection so memories never cross between
agents.  Memories are stored with an embedding, importance score, type tag,
and Unix timestamp, which the scoring function uses to rank retrieval results.
"""

import re
import time
import unicodedata
import uuid

import chromadb

from config import MAX_MEMORIES_RETRIEVE, MEMORY_DIR, MEMORY_PERSIST
from memory.scoring import score_memory

_UMLAUT_MAP = str.maketrans({"ä":"ae","ö":"oe","ü":"ue","Ä":"Ae","Ö":"Oe","Ü":"Ue","ß":"ss"})

def _collection_name(agent_name: str) -> str:
    name = agent_name.translate(_UMLAUT_MAP)
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r"[^a-zA-Z0-9._-]", "_", name).strip("._-") or "agent"
    return f"mem_{name}"


class MemoryStore:
    def __init__(self, agent_name: str, persist: bool = MEMORY_PERSIST):
        """Create or reopen a ChromaDB collection for agent_name.

        Args:
            agent_name: Used as the collection name prefix to keep agent
                        memories isolated from each other.
            persist:    If True, data is written to MEMORY_DIR on disk;
                        otherwise an in-memory ephemeral client is used.
        """
        if persist:
            client = chromadb.PersistentClient(path=MEMORY_DIR)
        else:
            client = chromadb.EphemeralClient()
        self.col = client.get_or_create_collection(_collection_name(agent_name))

    # write ------------------------------------------------------------------

    def add(
        self,
        content: str,
        mem_type: str,
        importance: float,
        embedding: list,
        stance: int = 0,
    ) -> None:
        """Persist a single memory with its metadata.

        Args:
            content:    The memory text (interaction summary or reflection insight).
            mem_type:   'interaction' or 'reflection'. ('persona' is reserved
                        but currently unused: persona seeding at init is
                        disabled, see Agent.__init__.)
            importance: LLM-rated significance on a 1-10 scale.
            embedding:  Pre-computed vector embedding of content.
            stance:     Stance this memory was formed under: +1, −1, or 0
                        (neutral/unknown, e.g. ambiguous classification). Used at
                        retrieval time to bias toward stance-congruent memories.
        """
        self.col.add(
            documents  = [content],
            embeddings = [embedding],
            metadatas  = [{
                "type": mem_type,
                "importance": importance,
                "ts": time.time(),
                "stance": stance,
            }],
            ids        = [str(uuid.uuid4())],
        )

    # read -------------------------------------------------------------------

    def retrieve(
        self,
        query_embedding: list,
        n: int = MAX_MEMORIES_RETRIEVE,
        query_stance: int | None = None,
    ) -> list[dict]:
        """Return the top-n memories by composite score against query_embedding.

        The full collection is scored by a composite of relevance (to
        query_embedding), recency, importance, and (when query_stance is given)
        stance congruence, then cut down to n. This lets a highly important or
        very recent memory surface even if it isn't among the closest by
        embedding distance alone.

        Args:
            query_embedding: Embedding of the current message or question.
            n:               Maximum number of memories to return.
            query_stance:    The retrieving agent's current softmax-drawn
                             stance (+1 or −1), or None to leave retrieval
                             stance-unbiased.

        Returns:
            List of {"content": str, "type": str, "score": float} dicts,
            sorted by descending composite score, truncated to n.
        """
        if self.col.count() == 0:
            return []

        results = self.col.query(
            query_embeddings = [query_embedding],
            n_results        = self.col.count(),
            include          = ["documents", "metadatas", "distances"],
        )

        scored = [
            {
                "content": doc,
                "type": meta["type"],
                "score": score_memory(meta, dist, query_stance),
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

        return sorted(scored, key=lambda x: x["score"], reverse=True)[:n]

    def all_recent(self, limit: int) -> list[str]:
        """Return the limit most recently added memory strings.

        Used to seed the reflection step, where recency matters more than
        relevance to a specific query.
        """
        if self.col.count() == 0:
            return []
        # Fetch all records with timestamps, then sort by recency.
        # col.get(limit=n) returns records in ChromaDB's internal ID order,
        # which is not guaranteed to match insertion/chronological order.
        results = self.col.get(include=["documents", "metadatas"])
        pairs = list(zip(results["documents"], results["metadatas"]))
        pairs.sort(key=lambda x: x[1].get("ts", 0.0), reverse=True)
        return [doc for doc, _ in pairs[:limit]]
