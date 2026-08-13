# Plan: Parallelization Strategy

## Context

A single simulation run is sequential by construction. With the current defaults
(`config.py`: `NUM_AGENTS=20`, `INTERACTIONS_PER_ROUND=20`, `NETWORK_MAX_ROUNDS=25`)
each round draws `INTERACTIONS_PER_ROUND` **asymmetric interactions** and processes
them one at a time. Every interaction mutates shared state that later draws depend
on: the drawn expresser's Q-values (`update_q_value`), the edge's rolling reward
history (`record_reward`, which feeds the next `select_responder` draw), and, under
`GRAPH_DYNAMIC`, the graph itself (`update_edge`). This is the random sequential
update of Banisch & Olbrich (2019); the interactions inside a round cannot be
reordered or overlapped without changing the model.

This supersedes the earlier symmetric design, where a round was a set of
independent pairs (max-weight matching) that could all run concurrently, plus a
per-discussion full-transcript concordance rating. Both parallelism sources are
gone: the interaction model is now asymmetric and stateful, and edge valuation
derives from `reward_from_expressions()` products accumulated in `EdgeData.reward_history`
(no separate concordance or reward call). Two levers remain, and the primary one has moved
from *inside* a run to *across* runs:

1. **Sweep-level process parallelism** across grid points (`run_sweep.py`, already
   implemented). Each grid point is a full independent `main_network.py` process.
   This is the primary lever and a natural fit for the replication-heavy schedule
   (`test_schedule.md`, roughly 43 to 53 runs).
2. **Reflection-phase thread parallelism** within a run (optional, not yet
   implemented). The reflection step is embarrassingly parallel across agents.

**Compute environment**: SCC cluster, NVIDIA L40 GPU (48 GB VRAM), Ollama server,
`qwen3.5:35b` (Q4_K_M, ~20 GB weights, leaving ~25 GB for KV cache). Embeddings run
via `sentence-transformers` on CPU, independent of the LLM backend.

---

## Where Time Is Spent

```
Per interaction (turns_per_agent = 1):
  2 gated turns          expresser + responder, each = ≥1 generation + 1 classify_expression_graded
  0 reward LLM calls     reward_a = graded(expresser) · graded(responder), reused from the
                         gate records (no separate classifier call; reward_b not computed here)

Per round (INTERACTIONS_PER_ROUND = 20):
  20 interactions        sequential (random sequential update)
  reflection             every REFLECT_EVERY rounds: NUM_AGENTS agents × ~4 LLM calls

Per run (25 rounds):
  500 interactions       = 1 000 gated turns  (× gate multiplier below)
  5 reflection phases    = 5 × 20 × ~4 ≈ 400 LLM calls
```

> **SFT expression gate multiplier (Step B).** A *gated turn* is not a single LLM
> call: `respond()` runs the expression gate (see README "SFT expression gate"),
> where each attempt costs one generation + one `classify_expression_graded()` call,
> retried until the draft no longer flips the drawn stance. With the measured
> first-attempt fidelities, expect roughly **1.7 generations + 1.7 classifications
> per gated turn** on average (classification calls are short, so the wall-clock
> multiplier sits somewhat below the ~3.4× call multiplier), concentrated on
> persona-dispreferred draws. `SFT_GATE_ENABLED = False` recovers the
> single-call-per-turn cost (ablation arm). The gate is agent-local and adds no
> cross-agent coupling, so it does not change the parallelism picture; it only
> raises the per-turn cost.

At an order-of-magnitude ~10 s per generation call on the L40, a single run of the
default configuration lands in the **high single-digit hours**, dominated by the
sequential interaction loop. Treat this as a placeholder: measure one real run
before planning a grid, since the figure scales with `NUM_AGENTS`,
`INTERACTIONS_PER_ROUND`, `NETWORK_MAX_ROUNDS`, and the gate multiplier.

---

## Parallelism Opportunities

| Opportunity | Independence | Leverage |
|---|---|---|
| **Grid points (sweep)** | Fully independent OS processes: separate ChromaDB, separate logs, no shared Python state | **Primary lever.** Scales with GPU concurrency; matches the replicated schedule |
| **Reflections across agents** | Each agent's ChromaDB collection is isolated; the argmax stance is fixed for the round | Up to `NUM_AGENTS`× during the reflection phase |
| **Interactions within a round** | Not independent: each mutates the shared Q / reward-history / graph state the next draw reads | Not parallelisable without switching to a synchronous update (a modeling change, not an optimization) |
| **Turns within an interaction** | Strictly sequential: B answers A, and each gate retry depends on the previous draft's classification | Not parallelisable |

The single-run bottleneck is therefore intrinsic. The way to use a multi-GPU or
high-concurrency budget is to run **many independent runs at once** (the sweep),
not to speed up one run internally. Within a single run, only the reflection phase
offers safe concurrency.

---

## Hardware Budget: SCC L40 (48 GB VRAM)

Per the Ollama FAQ, parallel request processing pre-allocates KV cache for **all**
slots at once:

```
total VRAM = model weights + OLLAMA_NUM_PARALLEL × KV_per_slot(num_ctx)
```

KV cache per slot scales linearly with context length. For `qwen3.5:35b`
(architecture approximate; values similar to the qwen2.5:32b class, fp16):

```
KV_per_slot = num_ctx × 64 layers × 2 (K+V) × 8 heads × 128 dim × 2 bytes
            ≈ num_ctx × 0.25 MB/token
            = 1.0 GB  at num_ctx = 4 096
            = 2.0 GB  at num_ctx = 8 192
```

Parallelism budget for `qwen3.5:35b` (~20 GB weights, ~25 GB remaining):

| `num_ctx` | KV per slot | Max `OLLAMA_NUM_PARALLEL` | Recommended |
|---|---|---|---|
| 4 096 | ~1.0 GB | ~28 | **10** |
| 8 192 | ~2.0 GB | ~14 | **8** |
| 32 768 | ~8.0 GB | ~3 | 3 |

**Practical recommendation**: keep `qwen3.5:35b` with `LLM_NUM_CTX = 4096`
(`config.py`, overridable via the `LLM_NUM_CTX` env var; already wired into the
`OllamaLLM(num_ctx=...)` constructor in both entry points). The largest prompt is
the `reflect()` Step 1 call, which feeds `MAX_MEMORIES_SEED = 15` memories
(~130 tokens each) alongside persona and instructions, roughly **2 100 tokens** in
fully populated rounds. All other calls (`respond()` ~870 tokens,
`classify_expression_graded()` ~300 tokens, `_score_importance()` ~200 tokens) sit well below
that, so 4 096 leaves ~2 000 tokens of headroom. The only scaling risk is raising
`MAX_MEMORIES_SEED` well beyond 15. This budget supports `OLLAMA_NUM_PARALLEL = 10`
comfortably within the ~28 GB envelope.

```bash
# Start Ollama with parallel request support (see RUNBOOK.md for the full startup).
OLLAMA_NUM_PARALLEL=10 OLLAMA_MAX_LOADED_MODELS=1 GGML_CUDA_NO_VMM=1 ollama serve &
```

---

## Phase 0: Enable `OLLAMA_NUM_PARALLEL` (prerequisite)

Ollama defaults to `OLLAMA_NUM_PARALLEL=1`, serialising all GPU requests regardless
of how many clients call it. Setting it to 10 turns Ollama into a genuinely
concurrent server that batches requests on the GPU.

Phase 0 pays off **only when there are concurrent requests to batch**. A single
sequential run keeps ~1 request in flight during the interaction loop, so on its own
Phase 0 changes nothing. It is the enabler for the two real levers below:
sweep-level processes (Phase 1) and the parallel reflection phase (Phase 2).

Cost: zero code changes.

---

## Phase 1: Sweep-Level Parallelism (`run_sweep.py`, implemented)

`run_sweep.py` is the primary parallelism layer and is already built. It expands
`PARAM_GRID` into one `SIM_*` environment-variable dict per combination and launches
`main_network.py` once per combination as an independent subprocess.

```bash
python run_sweep.py --dry-run     # print the grid without running anything
python run_sweep.py               # run the grid sequentially (--workers 1)
python run_sweep.py --workers 3   # run up to 3 grid points concurrently
```

**How it works**

- `PARAM_GRID` (currently `SIM_SBM_P_INTER`, `SIM_RESPONDER_SELECTION_BETA`,
  `SIM_OPINION_BETA`, `SIM_GRAPH_DYNAMIC`) is expanded by `build_grid()` into the
  Cartesian product. `config.py` reads each `SIM_*` variable via `os.getenv(...)`,
  falling back to its normal default when unset, so running `main_network.py`
  directly is unaffected.
- `COMMON_OVERRIDES` applies to every combination (currently a small
  `SIM_NUM_AGENTS` / `SIM_NETWORK_MAX_ROUNDS` pilot configuration); adjust once the
  pipeline is validated.
- Each combination gets a tagged run directory `logs/run_sweep_<ts>_<index>/` via
  `SIM_RUN_ID`, and the full grid (parameters, return code, duration, log path) is
  written to `logs/sweeps/<ts>/manifest.json` for joining results back to the
  parameters that produced them.

**Independence.** Each grid point is a separate OS process with its own Python
interpreter, its own ephemeral ChromaDB collections, and its own log directory.
There is no shared mutable state between runs, so `--workers > 1` is safe with no
locking.

**Composition with Phase 0.** All subprocess workers talk to the same Ollama server,
and their requests are batched together at the GPU. The constraint is:

```
sweep --workers × (per-run concurrency) ≤ OLLAMA_NUM_PARALLEL
```

During the interaction loop each run has a per-run concurrency of ~1, so
`--workers ≤ OLLAMA_NUM_PARALLEL` keeps every worker's request in flight without
queuing. Two operating modes on the SCC L40 (`OLLAMA_NUM_PARALLEL = 10`):

| Goal | `--workers` | Why |
|---|---|---|
| Maximise grid throughput (replication, pilots) | up to ~10 | Many runs progress at once at the same total GPU load; the whole schedule finishes far sooner even though each run stays sequential |
| Minimise a single run's wall-clock | 1 | Only one run in flight; per-run latency is then set by the sequential loop plus optional Phase 2 |

For the replicated Stage 1 to Stage 5 schedule (dozens of independent runs), grid
throughput is the relevant metric, and Phase 1 is exactly the right tool: N workers
give roughly N× schedule throughput up to the `OLLAMA_NUM_PARALLEL` ceiling, at no
quality cost.

**Recommendation**: validate with `run_sweep.py --dry-run` and the pilot
`COMMON_OVERRIDES` first; raise `--workers` only after Phase 0 is confirmed working
via a smoke run and `nvidia-smi` shows the GPU busy across concurrent requests.

---

## Phase 2: Parallel Reflection Phase (optional, within a run)

The one place a single run can be sped up internally is the reflection phase. Every
`REFLECT_EVERY` rounds, all `NUM_AGENTS` agents reflect; each `agent.reflect()` only
reads and writes its own isolated ChromaDB collection and reads its own argmax stance
(fixed for the round), so the agents are mutually independent. The interaction loop
stays strictly sequential; only this phase is threaded.

### Changes overview

```
config.py           add MAX_REFLECT_WORKERS
network/logger.py   add a threading.Lock to _write()
main_network.py     wrap the reflection phase in a ThreadPoolExecutor
```

No changes to `agents/`, `memory/`, `network/discussion.py`, `network/edges.py`,
`network/matching.py`, `network/state.py`, and none to the interaction loop.

### `config.py` addition

```python
# -- Parallelism ----------------------------------------------------------------
# Reflection is the only within-run parallelism opportunity (the interaction loop
# is a sequential random update and cannot be parallelised). Keep
# MAX_REFLECT_WORKERS ≤ OLLAMA_NUM_PARALLEL so reflection bursts do not queue; when
# running under run_sweep.py, keep sweep_workers × MAX_REFLECT_WORKERS within it.
MAX_REFLECT_WORKERS = 10
```

### `network/logger.py`, thread-safe write

The only shared mutable state touched from reflection threads is `events.jsonl`
(via `log_reflection` → `_write`). Guard `_write` with a lock; every other method
either routes through `_write` or writes per-round files from the main thread.

```python
import threading

class SimulationLogger:
    def __init__(self, run_id: str | None = None) -> None:
        ts = run_id or str(int(time.time()))
        self.run_dir     = Path("logs") / f"run_{ts}"
        self.rounds_dir  = self.run_dir / "network_rounds"
        self.rounds_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"
        self.start_time  = time.time()
        self._lock       = threading.Lock()      # guards events.jsonl appends

    def _write(self, record: dict) -> None:
        with self._lock:
            with open(self.events_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
```

### `main_network.py`, parallel reflection block

Only the reflection block changes; the interaction loop above it is untouched.

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import MAX_REFLECT_WORKERS

# ... inside the round loop, replacing the sequential reflection block ...
if round_n % REFLECT_EVERY == 0:
    print(f"\n-- Reflection phase (round {round_n}) --")
    with ThreadPoolExecutor(max_workers=MAX_REFLECT_WORKERS) as pool:
        futures = {
            pool.submit(agent.reflect,
                        state.opinion_states[name].argmax_stance): name
            for name, agent in agents.items()
        }
        for fut in as_completed(futures):
            name = futures[fut]
            fut.result()                    # re-raises any worker exception
            logger.log_reflection(round_n, name)
```

### Thread-safety audit

| Resource | Concurrent access? | Safe? | Reason |
|---|---|---|---|
| `Agent.llm` (OllamaLLM) | Multiple reflection threads | Yes | Stateless HTTP client; each call is independent |
| `Agent.memory` (ChromaDB) | Per-agent, no sharing | Yes | Each agent owns its collection |
| `Agent.reflect()` | Different agents only | Yes | Reads and writes the calling agent's own store |
| `NetworkState` (Q-values, graph) | Read-only during reflection | Yes | No mutation happens in the reflection phase |
| `SimulationLogger._write()` | Multiple threads | Yes | Guarded by `threading.Lock` |
| `SimulationLogger.snapshot_network()` | Main thread only | Yes | Called after the reflection phase completes |

**Critical invariant**: the interaction loop, all graph mutations
(`update_edge`, `ensure_connectivity`), and `snapshot_network` stay on the main
thread. Only `agent.reflect()` runs in the pool.

**Caveat under a sweep.** With `run_sweep.py --workers K` and Phase 2 active, total
GPU concurrency during a reflection phase is `K × MAX_REFLECT_WORKERS`. Keep that
product within `OLLAMA_NUM_PARALLEL` or accept queuing during the (brief, periodic)
reflection bursts. Because reflection is a small fraction of a run, the simplest
production setup is Phase 1 alone (`MAX_REFLECT_WORKERS = 1`), reserving Phase 2 for
single-run latency-sensitive work.

---

## Implementation Order

1. **Start Ollama with `OLLAMA_NUM_PARALLEL=10`** (Phase 0, prerequisite, no code
   changes; see RUNBOOK.md for the full startup flags).
2. **Use `run_sweep.py --workers N`** (Phase 1, already implemented) for grid and
   replication throughput. Start from `--dry-run`, then a small `--workers` value,
   watching `nvidia-smi`.
3. *(Optional)* Phase 2: `config.py` `MAX_REFLECT_WORKERS`, the `threading.Lock` in
   `network/logger.py`, and the threaded reflection block in `main_network.py`.
4. Smoke test: `SIM_NUM_AGENTS=4 SIM_NETWORK_MAX_ROUNDS=3` under the sweep; confirm
   `events.jsonl` is valid JSONL (`jq -c '.' events.jsonl`), the per-run snapshots
   are structurally unchanged, and GPU utilisation is visible across concurrent
   requests.

---

## Verification Checklist

- [ ] `run_sweep.py --dry-run` prints the expected grid; a real sweep writes one
      `logs/run_sweep_<ts>_<index>/` per combination and a complete
      `logs/sweeps/<ts>/manifest.json`.
- [ ] Concurrent sweep workers do not corrupt each other: each run directory is
      self-contained and its `events.jsonl` is valid JSONL.
- [ ] (Phase 2) A sequential run and a parallel-reflection run produce the same
      `events.jsonl` structure (same fields, valid JSON, no interleaved partial
      lines) and structurally identical `network_rounds/round_NNNN.json` snapshots.
- [ ] No `RuntimeError` or ChromaDB corruption under concurrent reflection (run with
      `NUM_AGENTS=20`, `MAX_REFLECT_WORKERS=10`).
- [ ] GPU utilisation visible in `nvidia-smi` during concurrent phases (~80 to 95%).

---

## Note on vLLM (if Ollama becomes the bottleneck)

With `OLLAMA_NUM_PARALLEL=10` and ~25 GB of KV-cache headroom on the L40, Ollama
provides genuine GPU-level batching, and the practical gap to vLLM is small for
`qwen3.5:35b`. Because the primary lever is now sweep-level process parallelism, the
relevant question is how many concurrent runs the server sustains, not intra-run
speed. vLLM is worth considering only when:

1. **A larger model is needed** (e.g. `qwen3.5:72b`), too large to fit the full KV
   budget alongside the weights; vLLM's paged KV management handles this more
   efficiently.
2. **`OLLAMA_NUM_PARALLEL` proves insufficient**: if profiling shows the Ollama
   server is the bottleneck above ~10 concurrent requests (many sweep workers),
   vLLM's continuous batching (requests batched mid-generation rather than queued)
   raises that ceiling.

If switching, the code changes are localised to `agents/agent.py` (add an
OpenAI-compatible client shim and route `self.llm.invoke()` through it) and
`config.py` (add `LLM_BACKEND`, `VLLM_HOST`, `VLLM_MODEL`). Embeddings run via
`sentence-transformers` on CPU, independent of both backends, so nothing changes
there.
