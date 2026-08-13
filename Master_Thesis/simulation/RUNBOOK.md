# RUNBOOK: Thesis-Simulation auf SCC

Wiederkehrender Startvorgang nach Login auf `scc2` oder nach einem Crash. Das
Setup ist einmalig erfolgt und wird hier nicht beschrieben.

## Kurzfassung

Der SCC ist ein **Sun-Grid-Engine-Cluster**. Man arbeitet nie direkt auf einem
Rechenknoten, sondern reicht Jobs vom Frontend `scc2` in eine Queue ein — für
uns immer die Queue `gpu`. `$HOME` und `/data/scc` sind auf allen Knoten
dasselbe Dateisystem, deshalb lässt sich ein laufender Job bequem von `scc2`
aus mitlesen.

**Der interaktive Weg (`qrsh`) ist seit dem Cluster-Server-Update defekt**, ssh
auf Knoten ebenfalls. Alles läuft daher über Batch-Jobs:
`qsub -v STAGE=<stage> -N stage_<stage> run_stage.sh`. Das Skript setzt aus der
Stage alle Parameter, belegt eine GPU, startet einen Ollama-Server und rechnet
`NREP` Replikate sequentiell dagegen. Folge-Stages lassen sich sofort
mitschicken, per `-hold_jid` verkettet.

Zwei Eigenheiten des Clusters bestimmen fast alles Weitere:

1. **GPUs werden nicht von SGE zugeteilt**, sondern von einer `starter_method`
   über Lock-Verzeichnisse in `/tmp` des Knotens. SGEs Zähler (`qc:gpu`) und die
   Realität können auseinanderlaufen — hinterlässt ein hart beendeter Fremdjob
   sein Lock, ist die Karte für alle dauerhaft blockiert, während SGE sie als
   frei meldet. Jobs, die dorthin dispatcht werden, sterben **vor** der ersten
   Skriptzeile. Details und Gegenmittel: [GPU-Zuteilung](#hintergrund-gpu-zuteilung-auf-der-gpu-queue).
2. **`h_vmem` ist ein Consumable**, kein Hinweis. SGE rechnet mit den
   *angeforderten* Werten laufender Jobs; ein Knoten kann laut `qhost` leer
   aussehen und trotzdem kein Kontingent haben. Fremdjobs halten hier gern
   188 G bei 30 Tagen Laufzeit — Wartezeiten von Tagen bis Wochen sind normal.

**Bewährte Herangehensweise bei belegtem Cluster:** pro brauchbarem Knoten
einen Wartejob absetzen (scc213, scc214, scc192 auf Vorrat), jeweils mit
eigenem `#$ -N`. Wer zuerst startet, gewinnt; die anderen brechen dank
Stage-Lock von selbst ab, statt in dieselben Ausgabeordner zu schreiben.
Danach `qstat` beobachten und die Verlierer per `qdel` entfernen.

**Wenn etwas schiefgeht, entscheidet die erste Logzeile** — die Tabelle in
[Batch-Betrieb](#die-erste-logzeile-entscheidet) ordnet sie zu.

---

## Voraussetzungen

- venv: `/data/scc/jonas.schrade/envs/thesis`
- Ollama-User-Installation: `/data/scc/jonas.schrade/ollama/install` (**0.30.10** —
  nicht die System-Binary `/software/bin/ollama`, die ist 0.1.26 und unbrauchbar)
- Modelle unter `/data/scc/jonas.schrade/ollama/models`: `qwen3.5:35b` (primär),
  `qwen2.5:32b` und `qwen2.5:14b` (Fallbacks, letzteres deutlich schneller)
- Embeddings laufen über `sentence-transformers` auf der CPU, kein Ollama-Modell nötig
- Setup-Skript `~/setup_thesis_env.sh`, Code unter `~/projects/thesis/simulation`
- Output-Verzeichnis `/data/scc/jonas.schrade/thesis-runs`

---

## Betriebsart wählen

**Batch (`qsub`) ist der Standard** — wartet in der Queue, überlebt Disconnects,
braucht keinen Node-Login.

**Interaktiv (`qrsh`)** wird eingeplant, öffnet aber keine Shell (Exit 0,
Rücksprung auf scc2). Ob es wieder geht, klärt ein Test:

```bash
qrsh -verbose -q gpu -l h_vmem=128G,tesla_l40=1,gpu=1 ; echo "Exit: $?"
```

Ändert sich der Prompt auf einen Knotennamen, ist der interaktive Weg zurück.
`-verbose` ist Pflicht — ohne das Flag verschluckt `qrsh` auch Fehlermeldungen.

---

## Ressourcen anfordern

Die Ressourcen-Zeile ist bei beiden Betriebsarten identisch (`#$ -l` im Skript
bzw. `-l` bei `qrsh`/`qsub`):

```bash
-l h_vmem=128G,tesla_l40=1,gpu=1   # scc213 (L40S)
-l h_vmem=96G,rtx_6000=1,gpu=1     # scc214 (Blackwell); dort ist h_vmem knapp
-l h_vmem=128G,gpu=1               # beide, Scheduler wählt
```

| Knoten | GPU | VRAM | RAM | Feature | Anmerkung |
|---|---|---|---|---|---|
| scc213 | 8× L40S | 48 GiB | 1024 GB | `tesla_l40=1` | Reichlich `h_vmem`-Kontingent. GPU 0 oft fremdbelegt → Discovery-Falle; anfällig für verwaiste Locks. |
| scc214 | 8× RTX PRO 6000 Blackwell | ~96 GB (mit `nvidia-smi --query-gpu=memory.total` verifizieren) | 1536 GB | `rtx_6000=1` | Blackwell (sm_120) läuft mit unserem Ollama-Build verifiziert auf GPU. `h_vmem`-Kontingent knapp, oft nur ~96 GiB frei. |
| scc192 | 4× L40 | 45 GiB | 1024 GB | `tesla_l40=1` | VRAM reicht. Meist `disabled` (`qstat -f -q gpu@scc192`, Spalte `states` = `d`), dabei oft **völlig leer**. Lohnt einen Wartejob auf Vorrat mit `#$ -q gpu@scc192`: Batch-Jobs bleiben auf disabled Queues in `qw` und starten bei Freigabe. Root-eigener ollama auf Port 11434. |
| scc146 | 4× Tesla V100 | 16 **oder** 32 GB — ungeprüft, nur 32 GB trägt qwen3.5:35b | 188 GB | `tesla_v100=1` | Meist `adu`: das `u` heißt, der execd antwortet nicht. Nicht einplanbar. |
| scc195-199 | 8× RTX 2080 Ti | 11 GiB | 128 GB | `rtx_2080ti=1` | **Zu klein** für qwen3.5:35b. |

**Verfügbarkeit vor dem Absetzen prüfen:**

```bash
qstat -F gpu -q gpu                  # 'qc:gpu=N' = freie Karten laut SGE
qhost -F h_vmem -h scc213,scc214     # 'hc:h_vmem' = freies Speicherkontingent
qstat -u "*" -q gpu                  # wer läuft gerade
qstat -u "*" -s p -q gpu             # wer wartet, nach Priorität sortiert
```

Bei Wartenden lohnt der Blick auf ihre Anforderung — ein auf `gpu@sccXXX`
gepinnter Fremdjob konkurriert nur dort:

```bash
qstat -j <fremdjob-id> | grep -E 'hard resource_list|hard_queue_list'
```

**Vier Regeln, die je einen Fehlversuch kosten:**

- **`gpu=1` mitgeben — und nie mehr.** Der Complex hat Default 0, seine Kapazität
  hängt an der **Queue** (`qconf -sq gpu`), nicht am Host; deshalb zeigen
  `qhost -F gpu` und `qconf -se scc213` nichts an. Er steuert, **wohin** SGE
  dispatcht. Ein höherer Wert lässt die `starter_method` entsprechend viele
  **physische** Karten verlangen und scheitern.
- **`h_vmem` ist der Filter gegen zu kleine Knoten.** 128G hält den Job von
  scc195-199 fern (nur 128 GB Host-RAM). Nur zusammen mit einem Feature-Constraint
  darf der Wert sinken — `rtx_6000=1` pinnt dann selbst auf scc214.
- **Kein Host-Pin** (`-l hostname=…`): scheitert ohne Retry, sobald der Knoten
  belegt ist. Auf eine Queue-Instanz pinnen (`-q gpu@sccXXX`) ist dagegen
  unproblematisch.
- **`-l` auf der Kommandozeile ersetzt `#$ -l` im Skript nicht, es wird damit
  vereinigt.** `qsub -l rtx_6000=1` auf ein Skript mit `tesla_l40=1` ergibt beide
  Features → `no suitable queues`. Skript kopieren und per `sed` umschreiben,
  danach `grep -n '^#\$' <skript>` kontrollieren.

---

## Batch-Betrieb

Das Skript liegt versioniert unter `simulation/run_stage.sh` und kommt per
`git pull` auf den Cluster (**nicht ins Terminal pasten** — Heredoc-Paste
zerlegt lange Blöcke).

**Das Skript wird nicht mehr editiert.** Die Stage kommt beim Absetzen mit, die
Stage-Tabelle im Skript setzt daraus alle `SIM_*`-Parameter:

```bash
cd ~/projects/thesis/simulation && git pull
S=1-04; qsub -v STAGE=$S -N stage_$S ~/projects/thesis/simulation/run_stage.sh
qstat -u jonas.schrade                       # qw → r
```

`STAGE` und `-N` aus **einer** Variablen zu setzen ist Absicht: der Jobname ist
nur ein Etikett, `STAGE` bestimmt die Physik. Getrennt getippt driften sie
auseinander, und das Log heißt dann nach einer Stage, die es nicht enthält.

**Warum die Parameter nicht mehr in `config.py` gesetzt werden:** `config.py`
wird gelesen, wenn der Job **startet**, nicht wenn er eingereiht wird — bei
Wartezeiten von Tagen und mehreren eingereihten Stages ist die Datei dann längst
auf einen anderen Wert gesetzt. Alle Stages liefen mit denselben Werten unter
verschiedenen Labels, ohne jede Fehlermeldung. Der Nachweis, was ein Lauf
bedeutet, ist deshalb jetzt die Stage-Tabelle in `run_stage.sh` (versioniert)
plus die Parameterzeile im Job-Log (siehe unten). `config.py` liefert nur noch
die Defaults für interaktive Läufe.

> **Ein veralteter Checkout bleibt die gefährlichste Fehlerquelle.** Läuft eine
> alte Fassung, schreibt sie mit dem alten `SIM_RUN_ID` in die Log-Ordner einer
> bereits fertigen Stage: `events.jsonl` wird **angehängt**, `personas.json` und
> `network_rounds/*.json` **überschrieben**. Deshalb immer erst `git pull`, und
> danach die Parameterzeile im Log prüfen — nicht die Skriptdatei, sondern das,
> was der Job tatsächlich geladen hat:
>
> ```bash
> grep -m1 'p_inter=' $RUNDIR/stage_<S>.o<jobid>
> ```
>
> (Vorgefallen am 2026-08-04, Job 1202993; der Stage-Lock der Vorgänger-Stage
> hat den Schaden verhindert.)

**Mehrere Knoten gleichzeitig bewerben**, wenn der Cluster voll ist. Die Kopien
unterscheiden sich nur noch in der Ressourcenzeile — Name und Stage kommen von
der Kommandozeile:

```bash
cd ~/projects/thesis/simulation && git pull && cd ~   # zuerst! sonst kopiert man eine alte Fassung
for v in 213 214 192; do cp ~/projects/thesis/simulation/run_stage.sh ~/run_wait_$v.sh; done

sed -i -e 's|^#\$ -l h_vmem=128G,tesla_l40=1,gpu=1$|#$ -l h_vmem=96G,rtx_6000=1,gpu=1|' ~/run_wait_214.sh
sed -i -e 's|^#\$ -q gpu$|#$ -q gpu@scc192|' ~/run_wait_192.sh
grep -n '^#\$' ~/run_wait_*.sh               # Kontrolle vor dem Absetzen

S=1-04
for v in 213 214 192; do qsub -v STAGE=$S -N stage_${S}_$v ~/run_wait_$v.sh; done
```

Die `#$ -o`-Zeile bleibt unangetastet: Sie zeigt auf ein Verzeichnis, SGE
benennt die Logs nach `<jobname>.o<jobid>`. Sobald einer auf `r` springt, die
übrigen per `qdel` entfernen.

### Mehrere Stages auf Vorrat einreihen

Bei Wartezeiten von Tagen lohnt es, die Folge-Stages sofort mitzuschicken: ein
wartender Job sammelt Wartezeit-Priorität, ein nicht abgesetzter nicht.

**Sequentiell verketten, nicht parallel.** Zwei Stages auf demselben Knoten
würden sich gegenseitig zerstören: `run_stage.sh` startet Ollama auf dem festen
Port 11500 und räumt vorher mit `pkill -u $USER -f ollama` auf — der zweite Job
killt also den Daemon des ersten. Der erste läuft weiter, seine restlichen
Replikate scheitern aber alle (die Schleife bricht bei Fehlern bewusst nicht ab).
`-hold_jid` verhindert das:

```bash
J=$(qsub -terse -v STAGE=1-04 -N stage_1-04 ~/projects/thesis/simulation/run_stage.sh)
J=$(qsub -terse -v STAGE=1-05 -N stage_1-05 -hold_jid $J ~/projects/thesis/simulation/run_stage.sh)
qstat -u jonas.schrade                       # hqw = wartet auf Vorgänger
```

Der Hold löst sich, sobald der Vorgänger **endet** — unabhängig vom Exit-Status.
Stirbt eine Stage an einem Phantom-Lock, rückt die nächste also nach, statt zu
blockieren. Was ein `hqw`-Job rechnet, steht fest: die Stage-Tabelle im Skript
ist zum Zeitpunkt des `qsub` festgeschrieben.

Zwei Stages sind im Schedule bewusst unvollständig und brechen beim Start ab,
wenn die fehlende Angabe nicht mitkommt:

```bash
# Stage 2 und 5-02 brauchen den empirischen Übergangspunkt aus Gate B:
qsub -v STAGE=2-01,SIM_SBM_P_INTER=0.08 -N stage_2-01 ~/projects/thesis/simulation/run_stage.sh
# Stage 5 braucht zusätzlich die permissive Formulierung:
qsub -v STAGE=5-01,SIM_TOPIC_TEXT='<permissiver Text>' -N stage_5-01 ~/projects/thesis/simulation/run_stage.sh
```

### Das Skript

Maßgeblich ist immer die Repo-Fassung `simulation/run_stage.sh`; der Abdruck
hier dient dem Verständnis der Bausteine.

Hier nur der Teil, der pro Stage entscheidet; die übrigen Bausteine (Lock,
Traps, GPU-Zuteilung, Ollama-Start, Replikat-Schleife) sind darunter in Prosa
erklärt. Die Werte selbst stehen in `test_schedule.md`, dort mit Begründung:

```bash
STAGE="${STAGE:-}"
[ -n "$STAGE" ] || { echo "ABBRUCH: STAGE nicht gesetzt."; exit 1; }

# Defaults = "Fixed"-Block von Stage 1; jede Stage überschreibt nur ihre Abweichung.
export SIM_SBM_P_INTRA=0.7
export SIM_OPINION_BETA=5.0
export SIM_RESPONDER_SELECTION_BETA=0.0
export SIM_GRAPH_DYNAMIC=0

case "$STAGE" in
  1-01) export SIM_SBM_P_INTER=0.02; NREP=5 ;;
  # … 1-02 … 1-05
  2-01) export SIM_OPINION_BETA=1.0; NREP=3; NEED_P_INTER=1 ;;
  # … 2-02 … 5-03
  *) echo "ABBRUCH: unbekannte Stage '$STAGE'"; exit 1 ;;
esac
# NEED_P_INTER / NEED_TOPIC: bricht ab, wenn die offene Angabe fehlt.

echo "Stage $STAGE  NREP=$NREP  p_intra=… p_inter=… β=… β_sel=… dynamic=… topic=…"
```

**Was die Blöcke tun:**

- **Stage-Tabelle.** Der einzige Ort, an dem Label und Physik zusammenkommen.
  Ein Tippfehler in `STAGE` trifft keinen `case`-Zweig und bricht ab, statt die
  Werte der Vorgänger-Stage zu erben; `4-01` fehlt bewusst, weil es laut
  Schedule die 3-01-Replikate wiederverwendet. Weil die Werte im Job-Environment
  stehen und nicht in `config.py`, ist festgeschrieben, was ein Job rechnet,
  sobald er eingereiht ist — die Voraussetzung dafür, mehrere Stages auf Vorrat
  einzureihen. Die `echo`-Zeile schreibt die effektiven Werte ins Log; sie ist
  der Beleg, den man nach dem Start prüft.

- **`#$`-Header.** SGE liest diese Zeilen beim Absetzen, nicht die Shell. `h_rt`
  ist die maximale Laufzeit — die `gpu`-Queue erlaubt mindestens 30 Tage
  (`h_rt=2592000`, so laufen Fremdjobs dort); danach killt SGE den Job. `-j y`
  legt stderr mit ins stdout-Log. `-o` zeigt auf ein **Verzeichnis**, SGE
  benennt die Datei dann `<jobname>.o<jobid>` — pro Job eindeutig, damit man nie
  versehentlich das Log des vorigen Laufs liest.
- **`set -euo pipefail` + Binary-Check.** Der Job soll früh und laut scheitern.
  Die falsche `ollama`-Binary (System-0.1.26) ist die teuerste Fehlerquelle
  überhaupt, weil sie erst beim Modell-Laden stillschweigend abstürzt.
- **Stage-Lock und Traps.** Wer auf mehreren Knoten parallel wartet, riskiert
  zwei gleichzeitige Läufe in denselben Ausgabeordnern und damit vermischte
  `events.jsonl`. `mkdir` ist atomar, auch über NFS — der zweite Job bricht
  sauber ab. Die Traps räumen Lock **und** Ollama-Daemon auf (SIGTERM →
  `exit 143` → EXIT-Trap) — aber **nicht verlässlich bei `qdel`**: Bash führt
  Traps erst aus, wenn das laufende Vordergrundkommando zurückkehrt. Hängt
  `python` noch in einem LLM-Call, schiebt SGE den SIGKILL nach, bevor die Bash
  zum Aufräumen kommt (beobachtet am 2026-08-04). Nach jedem `qdel` also
  prüfen: `ls -d $RUNDIR/.stage_*.lock`, ggf. `rmdir`. Ein verwaister Lock
  blockiert nur die *eigene* Stage; die Abbruchmeldung nennt den Pfad.
- **GPU-Zuteilung.** `$SGE_GPU` kommt von der `starter_method` und hat Vorrang.
  `setup_thesis_env.sh` setzt `CUDA_VISIBLE_DEVICES=7` nur, wenn die Variable
  leer ist, überschreibt die Zuteilung also nicht — die Reihenfolge (erst
  sourcen, dann `SGE_GPU` setzen) sichert das zusätzlich ab. Der
  `nvidia-smi`-Fallback greift nur außerhalb der `gpu`-Queue und überspringt
  Index 0 bewusst, weil dort die Discovery-Falle lauert.
- **Ollama-Start.** Eigener Port (11434 kann root gehören), `GGML_CUDA_NO_VMM=1`
  gegen den 256-GB-VM-Pool, Log pro Job unter `~/ollama_$JOB_ID.log`. Der
  anschließende Check bricht ab, sobald der Server auf CPU zurückgefallen ist —
  ohne ihn würde der Job tagelang unbrauchbare Ergebnisse produzieren.
- **Replikat-Schleife.** Alle `NREP` Läufe teilen sich **einen** Server, das
  Modell bleibt geladen. Ein fehlgeschlagenes Replikat stoppt die Schleife nicht
  (`if … else`), sonst würde ein einzelner Ausreißer die ganze Stage kosten.

### Die erste Logzeile entscheidet

| Zeile | Bedeutung |
|---|---|
| `Knoten … GPU N` → `Ollama auf GPU bereit.` | Alles gut, die Replikate laufen. |
| `ERROR: Could only reserve 0 of 1 requested devices.` | Kommt **vor** jeder Skriptzeile, aus der `starter_method`: keine freie Karte laut Lock-Mechanismus. SGE reiht einmal neu ein, der Job stirbt erneut und verschwindet lautlos aus `qstat`. Siehe [GPU-Zuteilung](#hintergrund-gpu-zuteilung-auf-der-gpu-queue). |
| `ABBRUCH: … CPU-Fallback` | Discovery-Falle: GPU 0 des Knotens ist fremdbelegt. Der Job beendet sich absichtlich sofort. Auf einen anderen Knoten ausweichen. |
| `ABBRUCH: Stage … läuft bereits` | Ein Parallel-Wartejob war schneller. Genau so gedacht — diesen Job löschen. |
| `… unbound variable` | `set -u` gegen ein unvollständiges Environment; siehe Fehlersuche. |

### Andere Sub-Stage

Nichts editieren, nur `-v STAGE=` und `-N` mitgeben (siehe oben). Die
Replikatzahl kommt aus der Stage-Tabelle. Ein Ollama-Server bedient alle
Replikate **sequentiell** — nicht parallel starten (GPU-Contention,
KV-Cache-OOM). Ergebnis: `logs/run_<STAGE>_r1/` … `_r<NREP>/`.

> `SIM_RUN_ID` benennt nur den Log-Ordner und setzt **keine** Parameter. Dass
> `1-04_r3` wirklich Stage 1-04 ist, belegt die Parameterzeile im Job-Log, nicht
> der Ordnername — sie zeigt, was der Prozess geladen hat.

---

## Fortschritt verfolgen

Vier Ebenen, von grob nach fein. `python -u` im Runner sorgt dafür, dass alles
ungepuffert und damit live erscheint.

**Job lebt und was er verbraucht** — `maxvmem` ist zugleich die Zahl, an der
sich `h_vmem` für künftige Stages bemessen lässt:

```bash
qstat -j <job-id> | grep -E 'usage|maxvmem'
```

**Welches Replikat läuft:**

```bash
grep -E '=== Replikat|Replikat .* OK' /data/scc/jonas.schrade/thesis-runs/<jobname>.o<jobid>
```

**Live mitlesen.** Die volle Ausgabe ist geschwätzig (20 Personas je
Replikatstart, 20 Interaktionszeilen je Runde), deshalb meist gefiltert:

```bash
tail -f /data/scc/jonas.schrade/thesis-runs/<jobname>.o<jobid>

tail -f /data/scc/jonas.schrade/thesis-runs/<jobname>.o<jobid> \
  | grep --line-buffered -E '=== Replikat|Reflection phase|Round .* completed|Simulation complete|Replikat .* OK'
```

`--line-buffered` ist nicht optional — sonst puffert `grep` blockweise und es
kommt minutenlang nichts. Auf NFS kann `tail -f` nachhängen; dann `tail -F`.
`Round N completed in X.Xs` ist die Zeile für die Hochrechnung: 25 Runden je
Replikat, `NREP` Replikate je Stage.

**Fortschritt im Dateisystem**, unabhängig vom Log:

```bash
D=/data/scc/jonas.schrade/thesis-runs/logs/run_1-03_r1
ls $D/network_rounds/ | tail -1     # round_0007.json = Runde 7 von 25
wc -l < $D/events.jsonl             # wächst laufend
```

Stagniert `events.jsonl` über Minuten bei unverändertem Zeitstempel, hängt ein
LLM-Call.

---

## Interaktiver Betrieb

Nur nutzbar, wenn der Test oben eine Shell öffnet. `run_stage.sh` automatisiert
genau diese Schritte.

**1. Knoten anfordern** — siehe [Ressourcen](#ressourcen-anfordern):

```bash
qrsh -verbose -q gpu -l h_vmem=128G,tesla_l40=1,gpu=1
```

**2. Environment laden und Binary prüfen** (häufigste, teuerste Fehlerquelle):

```bash
source ~/setup_thesis_env.sh
which ollama && ollama --version
```

Muss `/data/scc/jonas.schrade/ollama/install/bin/ollama` und `0.30.10` zeigen.

**3. GPU-Zuteilung übernehmen.** In der `gpu`-Queue hat die `starter_method`
bereits eine Karte reserviert; eine eigene Wahl wäre ein Zugriff auf fremd
reservierte Hardware:

```bash
echo "$SGE_GPU"
# nur falls leer: freie Karte selbst suchen (memory.used deutlich unter 1 GB)
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
export CUDA_VISIBLE_DEVICES=<idx>
```

**4. Ollama starten.** Eigener Port, weil auf manchen Knoten ein root-eigener
Daemon auf 11434 läuft. Nur **eigene** Prozesse killen:

```bash
export OLLAMA_HOST=127.0.0.1:11500
pkill -u jonas.schrade -f ollama; sleep 2; pgrep -u jonas.schrade -af ollama
GGML_CUDA_NO_VMM=1 ollama serve > ~/ollama.log 2>&1 &
sleep 5; ollama list
grep "inference compute" ~/ollama.log      # erwartet: library=CUDA …
```

> `GGML_CUDA_NO_VMM=1` ist Pflicht: ggml reserviert sonst einen 256-GB-VM-Pool
> (`cuMemAddressReserve`), der am harten `ulimit -Hv` von 128 GB scheitert. Ohne
> das Flag lädt kein Modell.

**5. Simulation starten.** Outputs landen relativ zum CWD in `logs/run_<id>/`:

```bash
cd /data/scc/jonas.schrade/thesis-runs
python ~/projects/thesis/simulation/main_pairwise.py   # Smoke-Test, 2 Agents
nohup python -u ~/projects/thesis/simulation/main_network.py \
  > run_$(date +%Y%m%d_%H%M%S).out 2>&1 &               # voller Lauf, disconnect-fest
```

**6. Sauber beenden:**

```bash
pkill -u jonas.schrade -f ollama; sleep 2
exit
```

---

## config.py-Pflichtwerte

```python
LLM_MODEL   = "qwen3.5:35b"
LLM_NUM_CTX = 4096               # via os.getenv("LLM_NUM_CTX") überschreibbar
OLLAMA_HOST = "127.0.0.1:11500"  # Port wie beim Serverstart
```

`num_ctx=4096` ist kein Detail: ohne Cap wählt ggml 32 768 Tokens KV-Cache
(≈ 8 GiB) und treibt den Bedarf auf ~27 GiB; mit 4096 sinkt er auf ~20 GiB.

`reasoning=False` muss im `OllamaLLM(...)`-Konstruktor stehen (mappt auf Ollamas
`think: false`). Sonst liefert qwen3.5 `<think>…</think>`-Blöcke, und
`_score_importance()` parst den ersten Token zu einer Zahl → stiller
Fallback-Wert 5.0. `classify_reward()`/`classify_expression()` sind über das
JSON-Schema robust dagegen, profitieren aber von der kürzeren Generationszeit.

---

## Ergebnisse

```bash
ls /data/scc/jonas.schrade/thesis-runs/logs/run_<id>/
```

- `events.jsonl` — eine Zeile pro Diskussion, Kantenänderung und Reflexion; primäre Analyse-Daten
- `personas.json` — die gezogenen Personas
- `network_rounds/round_NNNN.json` — Snapshot nach jeder Runde, `round_0000` ist der Ausgangszustand

Herunterladen (im **lokalen** Terminal):

```bash
scp -r jonas.schrade@scc2.uni-konstanz.de:/data/scc/jonas.schrade/thesis-runs/logs/run_<id> .
```

---

## Jobs verwalten

```bash
qstat -u jonas.schrade              # eigene Jobs clusterweit
qdel <job-id> [<job-id> …]          # Jobs entfernen (qw wie r); -f erzwingt bei 'dr'
qstat -j <job-id> | tail -30        # Block 'scheduling info:' = warum noch qw
qacct -j <job-id>                   # abgeschlossener Job: hostname, exit_status, maxvmem
qalter -N <neuer-name> <job-id>     # Namen nachträglich ändern
```

`qstat` schneidet die Namensspalte bei zehn Zeichen ab; die vollen Namen zeigt
`qstat -r`. Manuelles Aufräumen entfällt: Die Traps im Runner beenden den
Ollama-Daemon, mit dem Job-Ende gibt SGE Knoten und GPU frei.

---

## Hintergrund: GPU-Zuteilung auf der gpu-Queue

Die Queue hat `prolog NONE`, aber eine `starter_method`
(`/data/sw/sge/scripts/start-gpu.sh` → `start-job.sh`, sourct
`start-gpu_helper.sh`; Basis: [kyamagu/sge-gpuprolog](https://github.com/kyamagu/sge-gpuprolog)).
Sie umschließt jeden Job und reserviert Karten **nicht über SGE**, sondern über
Lock-Verzeichnisse `/tmp/sge-gpu-lock-<jobid>.<taskid>-<dev>` auf dem Knoten.
Danach exportiert sie `SGE_GPU` und `CUDA_VISIBLE_DEVICES`.

Daraus folgen drei Dinge:

- Die Wunschzahl liest der Helper über `qstat.orig` (Logzeile
  `Number of requested GPUS: N`). Die Binary ist unzuverlässig und liefert
  teils nichts; dann greift der Fallback **1** — für uns der richtige Wert.
  Ein höherer `gpu=`-Request wird jedoch ausgelesen und lässt den Helper
  entsprechend viele **physische** Karten verlangen: `gpu=3` endet in
  `Could only reserve 1 of 3` und verschenkt die eine Karte, die frei war.
  **Für diese Simulation gilt immer `gpu=1`.**
- Findet der Helper keine freie Karte, bricht er **vor** dem Job-Skript ab.
  `qacct` zeigt dann `failed 0`, `exit_status 1` — formal ein Job-Exit, kein
  Prolog-Fehler.
- Der Epilog entfernt nur Locks des **eigenen** Jobs. Hart abgebrochene
  Fremdjobs hinterlassen dauerhafte Locks, die eine Karte für immer sperren.

**Verwaiste Locks erkennen.** `qstat -F gpu -q gpu` meldet freie Karten
(`qc:gpu>0`), der Helper findet trotzdem keine. Da jeder Job genau eine Karte
reserviert, muss gelten: *belegte Slots* (aus `qhost -q -h <knoten>`) + *freie
Karten* (`qc:gpu`) = *Gesamtzahl GPUs*. Geht die Rechnung nicht auf, ist die
Differenz Müll. **Kein Userspace-Fix** — die Locks gehören fremden Usern;
Betreiber-Ticket mit Job-ID, `qacct -j <id>` und den beiden Zählungen als Beleg.
Eigendiagnose nur über einen Probe-Job in eine **Nicht-GPU-Queue desselben
Knotens** (`qhost -q -h <knoten>`), da der Helper nur bei `$QUEUE == gpu` läuft;
dort ist `ls -ld /tmp/sge-gpu-lock-*` möglich. Ist die einzige andere Queue des
Knotens `disabled`, entfällt auch das.

**Trotz Phantom-Locks warten statt sterben.** Solange SGE Karten zählt, die der
Helper nicht vergeben kann, wird ein `gpu=1`-Job sofort dispatcht und stirbt.
Als Wartebedingung eignet sich nur eine Ressource, die der Helper **nicht**
liest — praktisch `h_vmem`. Den Wert so wählen, dass er jetzt unerfüllbar ist,
aber erfüllt wird, sobald ein Fremdjob endet: Dann wird zeitgleich dessen
GPU-Lock frei, und der Helper findet die Karte.

```bash
qhost -F h_vmem -h <knoten>                         # hc:h_vmem = frei
qstat -j <fremdjob-id> | grep 'hard resource_list'  # was ein Job dort hält
```

Beispiel: frei 252 G, jeder Fremdjob hält 188 G → `h_vmem` zwischen 253 G und
440 G wählen, etwa 300 G. Kostet keinen echten Speicher (`h_vmem` ist eine
Obergrenze), belegt aber Kontingent — also nicht unnötig hoch ansetzen. Gegen
Konkurrenten, die auf denselben Knoten warten, hilft der Kniff nicht; das
vorher mit `qstat -u "*" -s p -q gpu` prüfen.

---

## Fehlersuche

| Symptom | Ursache | Gegenmittel |
|---|---|---|
| `qrsh`/`qlogin` meldet „successfully scheduled", öffnet aber keine Shell | Interaktiver Attach seit Server-Update defekt (PTY-/Transport-Handoff) | Kein Userspace-Fix → Batch. Regression an den Support melden. |
| `qrsh` springt kommentarlos zurück | Ohne `-verbose` unterdrückt `qrsh` alle Scheduling-Meldungen | Immer `qrsh -verbose … ; echo $?` |
| `qrsh … could not be scheduled` / `got select timeout` | Interaktive Jobs brauchen Sofort-Dispatch ohne Reservierung; scheitert bei Last, wo ein Batch-Job durchkommt | Als `qsub`-Batch einreihen |
| `ssh sccXXX` verlangt Passwort | Host-/Key-Auth zwischen Knoten kaputt; Node-SSH bräuchte zusätzlich einen laufenden Job dort | Node-Login vermeiden → Batch |
| `error: no suitable queues` | Knoten hat das Feature nicht, **oder** `-l` von Kommandozeile und Skript wurden vereinigt | Features prüfen (`tesla_l40` vs. `rtx_6000`); Skript kopieren statt per Kommandozeile übersteuern |
| Job bleibt lange in `qw`, ohne Fehler | Meist `h_vmem`: ein Consumable, SGE rechnet mit den *angeforderten* Werten laufender Jobs, nicht mit `qhost`-MEMUSE | `qstat -j <id> \| tail -30` → `cannot run at host … offers only hc:h_vmem=<bytes>`. Warten ist korrekt, der Job startet automatisch. |
| Job verschwindet binnen Sekunden aus `qstat`, ohne zu laufen | Dispatcht und in der `starter_method` gestorben | Log lesen — der Fehler steht dort **vor** jeder Skriptausgabe. `qacct -j <id>` zeigt `failed 0`, `exit_status 1`. |
| `ERROR: Could only reserve N of M requested devices.` | Keine (bzw. zu wenige) freie GPU laut Lock-Mechanismus | `gpu=1` verwenden, nie mehr. Sonst siehe [GPU-Zuteilung](#hintergrund-gpu-zuteilung-auf-der-gpu-queue). |
| `library=cpu` bzw. `CUDA … device(s) is/are busy or unavailable` trotz freier GPU | **Discovery-Falle:** GPUs im `Exclusive_Process`-Modus; Ollamas Discovery-Probe erzwingt intern `CUDA_VISIBLE_DEVICES=0` und legt dort einen Kontext an — ist GPU 0 fremdbelegt, crasht sie, **bevor** deine Auswahl greift. Weder anderer Index/UUID noch `OLLAMA_LLM_LIBRARY=cuda_v12` helfen. | Kein Userspace-Fix. Knoten mit freier GPU 0 wählen. `run_stage.sh` erkennt es und bricht sofort ab. Prüfen: `nvidia-smi --query-gpu=index,compute_mode --format=csv,noheader` |
| `setup_thesis_env.sh: line N: LD_LIBRARY_PATH: unbound variable` | Das Setup-Skript hängt an nicht gesetzte Variablen an; `set -u` macht daraus einen Abbruch — **nach** erfolgreicher GPU-Reservierung, die Karte ist also verschenkt | In `run_stage.sh` behoben (`set +u` nur ums Sourcen). Bei eigenen Skripten genauso verfahren. |
| `offloaded 0/NN layers to GPU` | Context-/Layout-Größe, **nicht** `h_vmem` | `LLM_NUM_CTX=4096` prüfen, Server sauber neu starten |
| `ggml_aligned_malloc: insufficient memory` | `h_vmem` zu niedrig fürs Modell | Höher anfordern; realer Bedarf aus `qacct -j <id> \| grep maxvmem` eines gelungenen Laufs ablesen |
| `CUDA error: out of memory` bei `cuMemAddressReserve` | ggml reserviert 256-GB-VM-Pool gegen `ulimit -Hv` = 128 GB | `GGML_CUDA_NO_VMM=1` beim Serverstart |
| `ollama --version` zeigt `0.1.26` / `/software/bin` | Environment nicht gesourced | `source ~/setup_thesis_env.sh` |
| `address already in use` (11434) | root-Ollama belegt den Default-Port | `export OLLAMA_HOST=127.0.0.1:11500` |
| `pkill: Operation not permitted` | Versuch, den root-Prozess zu killen | `-u jonas.schrade` ergänzen |
| `could not connect to ollama app` | Daemon abgestürzt oder nicht gestartet | `tail -n 30 ~/ollama_<jobid>.log`, Port und Binary prüfen |
| Python meldet 3.13 / ImportError | System-Python statt venv | `source ~/setup_thesis_env.sh`; `which python` muss auf `/data/scc/.../thesis/bin/python` zeigen |
| Heredoc-Paste erzeugt kaputte Datei | Erster Paste ließ das Heredoc offen, der zweite wurde verschluckt | Skripte nicht ins Terminal pasten, per `git pull` holen. `head -1` muss `#!/bin/bash` sein. |
| VSCode Remote-SSH verbindet nicht | Server-Update | In den Settings `"remote.SSH.useExecServer": false` |
