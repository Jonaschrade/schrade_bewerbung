#!/bin/bash
# Batch-Runner für eine Sub-Stage.
#
# Startet EINEN GPU-Ollama-Server und läuft NREP Replikate von main_network.py
# sequentiell dagegen (Modell bleibt geladen). Ersetzt den interaktiven
# qrsh+nohup-Weg, der seit dem Cluster-Server-Update defekt ist: interaktiver
# Attach (qrsh/qlogin) und ssh-auf-Knoten scheitern, Batch nicht.
#
# Die Stage wird beim Absetzen mitgegeben, das Skript wird NICHT mehr editiert.
# Die Tabelle unten setzt daraus alle SIM_*-Parameter, sodass jeder Job seine
# Physik selbst mitbringt: config.py wird erst beim Jobstart gelesen, nicht beim
# qsub, und wäre bei mehreren gleichzeitig eingereihten Stages längst überschrieben.
#
# Das Log heißt <jobname>.o<jobid>, ist also pro Job eindeutig — kein
# Überschreiben, keine Verwechslung mit dem Lauf davor.
#
# Absetzen:  S=1-04; qsub -v STAGE=$S -N stage_$S ~/projects/thesis/simulation/run_stage.sh
# Verfolgen: tail -f /data/scc/jonas.schrade/thesis-runs/stage_<S>.o<jobid>

#$ -q gpu
#$ -l h_vmem=128G,tesla_l40=1,gpu=1
#$ -l h_rt=168:00:00
#$ -N thesis_stage
#$ -o /data/scc/jonas.schrade/thesis-runs/ -j y

set -euo pipefail
RUNDIR=/data/scc/jonas.schrade/thesis-runs

STAGE="${STAGE:-}"
[ -n "$STAGE" ] || {
  echo "ABBRUCH: STAGE nicht gesetzt."
  echo "  S=1-04; qsub -v STAGE=\$S -N stage_\$S ~/projects/thesis/simulation/run_stage.sh"
  exit 1
}

# Stage-Parameter aus test_schedule.md (dort steht die Begründung, hier nur die
# Werte). Koppelt Label und Physik: eine unbekannte Stage bricht ab, statt still
# mit den Werten der Vorgänger-Stage zu rechnen.
# Defaults = "Fixed"-Block von Stage 1; jede Stage überschreibt nur ihre Abweichung.
export SIM_SBM_P_INTRA=0.7
export SIM_OPINION_BETA=5.0
export SIM_RESPONDER_SELECTION_BETA=0.0
export SIM_GRAPH_DYNAMIC=0

case "$STAGE" in
  # Stage 1 — Phasenübergangs-Scan über P_INTER.
  1-01) export SIM_SBM_P_INTER=0.02; NREP=5 ;;
  1-02) export SIM_SBM_P_INTER=0.08; NREP=5 ;;
  1-03) export SIM_SBM_P_INTER=0.15; NREP=5 ;;
  1-04) export SIM_SBM_P_INTER=0.30; NREP=5 ;;
  1-05) export SIM_SBM_P_INTER=0.70; NREP=5 ;;

  # Stage 2 — β-Sensitivität am empirischen Übergangspunkt. Der steht erst nach
  # Stage 1 fest (Gate B), muss also beim qsub mitgegeben werden.
  2-01) export SIM_OPINION_BETA=1.0; NREP=3; NEED_P_INTER=1 ;;
  2-02) export SIM_OPINION_BETA=2.0; NREP=3; NEED_P_INTER=1 ;;
  2-03) export SIM_OPINION_BETA=5.0; NREP=3; NEED_P_INTER=1 ;;

  # Stage 3 — Responder-Selection am polarisierten Anker.
  3-01) export SIM_SBM_P_INTER=0.02; export SIM_RESPONDER_SELECTION_BETA=0.0; NREP=3 ;;
  3-02) export SIM_SBM_P_INTER=0.02; export SIM_RESPONDER_SELECTION_BETA=2.0; NREP=3 ;;

  # Stage 4 — dynamischer Graph. 4-01 ist per Design die Wiederverwendung der
  # 3-01-Replikate und hat deshalb bewusst keinen eigenen Eintrag.
  4-02) export SIM_SBM_P_INTER=0.02; export SIM_GRAPH_DYNAMIC=1; NREP=3 ;;

  # Stage 5 — Stance-Flip-Kontrolle unter permissiver Formulierung. Der Topic-Text
  # ist eine inhaltliche Entscheidung und wird deshalb nicht hier festgeschrieben.
  5-01) export SIM_SBM_P_INTER=0.02; NREP=3; NEED_TOPIC=1 ;;
  5-02) NREP=3; NEED_P_INTER=1; NEED_TOPIC=1 ;;
  5-03) export SIM_SBM_P_INTER=0.70; NREP=3; NEED_TOPIC=1 ;;

  *) echo "ABBRUCH: unbekannte Stage '$STAGE'"; exit 1 ;;
esac

# Was der Schedule offenlässt, muss beim Absetzen kommen — lieber sofort abbrechen
# als still mit dem Default aus config.py rechnen.
if [ -n "${NEED_P_INTER:-}" ] && [ -z "${SIM_SBM_P_INTER:-}" ]; then
  echo "ABBRUCH: Stage $STAGE braucht den Gate-B-Übergangspunkt aus Stage 1."
  echo "  qsub -v STAGE=$STAGE,SIM_SBM_P_INTER=<wert> -N stage_$STAGE ..."
  exit 1
fi
if [ -n "${NEED_TOPIC:-}" ] && [ -z "${SIM_TOPIC_TEXT:-}" ]; then
  echo "ABBRUCH: Stage $STAGE braucht die permissive Formulierung."
  echo "  qsub -v STAGE=$STAGE,SIM_TOPIC_TEXT='<text>' -N stage_$STAGE ..."
  exit 1
fi

echo "Stage $STAGE  NREP=$NREP  p_intra=$SIM_SBM_P_INTRA  p_inter=$SIM_SBM_P_INTER" \
     "β=$SIM_OPINION_BETA  β_sel=$SIM_RESPONDER_SELECTION_BETA" \
     "dynamic=$SIM_GRAPH_DYNAMIC  topic=${SIM_TOPIC_LABEL:-<config.py>}"

# Schutz gegen Doppelläufe: Warten mehrere Jobs derselben Stage auf verschiedenen
# Knoten, würden sie sonst gleichzeitig in dieselben logs/run_<STAGE>_r*/ schreiben.
# mkdir ist atomar (auch über NFS) — wer zuerst startet, gewinnt.
STAGE_LOCK="$RUNDIR/.stage_${STAGE}.lock"
mkdir "$STAGE_LOCK" 2>/dev/null || {
  echo "ABBRUCH: Stage $STAGE läuft bereits. Lock: $STAGE_LOCK"
  echo "Falls verwaist (Job hart abgebrochen): rmdir '$STAGE_LOCK'"
  exit 1
}
# Räumt Lock und Ollama-Daemon auf — auch bei qdel (SIGTERM → exit → EXIT-Trap).
trap 'kill "${OLLAMA_PID:-}" 2>/dev/null; rmdir "$STAGE_LOCK" 2>/dev/null' EXIT
trap 'exit 143' TERM INT

# setup_thesis_env.sh hängt an teils nicht gesetzte Variablen an (LD_LIBRARY_PATH)
# und stirbt deshalb unter 'set -u'. Nur fürs Sourcen abschalten.
set +u
source ~/setup_thesis_env.sh
set -u
which ollama | grep -q '/data/scc' || { echo "FALSCHE ollama-Binary!"; exit 1; }

# GPU-Zuteilung der starter_method übernehmen: sie reserviert die Karte (Lock
# unter /tmp/sge-gpu-lock-*) und legt ihren Index in SGE_GPU ab. Findet sie keine
# freie Karte, stirbt der Job schon vor dieser Zeile mit "Could only reserve
# 0 of 1 requested devices" (siehe RUNBOOK, "GPU-Zuteilung auf der gpu-Queue").
if [ -n "${SGE_GPU:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${SGE_GPU// /,}"
else
  # Fallback ohne starter_method: freie GPU wählen, Index 0 aus (Discovery-Falle)
  export CUDA_VISIBLE_DEVICES=$(nvidia-smi \
    --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' '$1!=0 && $2<1000 {print $1; exit}')
  [ -n "$CUDA_VISIBLE_DEVICES" ] || { echo "Keine freie GPU (außer 0)!"; exit 1; }
fi
echo "Knoten $(hostname), GPU $CUDA_VISIBLE_DEVICES (SGE_GPU='${SGE_GPU:-unset}')"

export OLLAMA_HOST=127.0.0.1:11500
pkill -u "$USER" -f ollama || true; sleep 2
GGML_CUDA_NO_VMM=1 ollama serve > "$HOME/ollama_$JOB_ID.log" 2>&1 &
OLLAMA_PID=$!
sleep 10

# HARTER CPU-Fallback-Check: lieber sofort abbrechen als tagelang auf CPU rechnen
if grep -q "library=cpu" "$HOME/ollama_$JOB_ID.log" && \
   ! grep -qi "library=cuda" "$HOME/ollama_$JOB_ID.log"; then
  echo "ABBRUCH: Ollama im CPU-Fallback (GPU 0 belegt?). Siehe ollama_$JOB_ID.log"
  kill $OLLAMA_PID; exit 1
fi
echo "Ollama auf GPU bereit."

cd "$RUNDIR"
for r in $(seq 1 $NREP); do
  echo "=== Replikat $r/$NREP (SIM_RUN_ID=${STAGE}_r${r}) ==="
  if SIM_RUN_ID=${STAGE}_r${r} python -u ~/projects/thesis/simulation/main_network.py; then
    echo "Replikat $r OK"
  else
    echo "Replikat $r FEHLGESCHLAGEN (exit $?) — weiter mit nächstem"
  fi
done

echo "Stage $STAGE fertig."
