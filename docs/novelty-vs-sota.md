# Novelty vs State of the Art — & Evaluation Metrics

*Towards Semantically-Aware End-of-Utterance Detection for Conversational AI Systems*
Meeting brief · sources = 5 papers read in full (2024–2026).

---

## 1. The issue
Real-time endpointing forces a **latency ↔ interruption** trade-off. A VAD + fixed silence
timeout cuts a caller who is spelling a name, dictating a number, or pausing to think — or it
over-waits. Single-signal and fixed-threshold methods can only pick one compromise.

## 2. Novelty — the unclaimed conjunction
> **A streaming EOU detector that (i) fuses acoustic + lexical evidence into a _calibrated_
> completion probability, and (ii) decides via a _cost-sensitive optimal-stopping rule_.**

The two pillars each exist **separately** in nearby 2026 work; **their conjunction is not done
by any surveyed system**, and **none reports any calibration diagnostic**.

## 3. Positioning table

| System (year) | Acoustic+lexical fusion | Decision method | Calibrated? | Cost / optimal-stopping? | Lang | Code |
|---|---|---|---|---|---|---|
| **FastTurn** (2026) | ✅ CTC→Qwen3-0.6B LLM + Conformer, MLP | 4-state **argmax** | ❌ | ❌ | zh+en | ✅ test set |
| **JAL-Turn** (2026) | ✅ SenseVoice+CPC cross-attention | binary, **fixed τ=0.5** | ❌ | ❌ | multi+ja | ❌ (data only) |
| **Endpoint Anticipation** (2026) | ❌ **acoustic-only** (Mimi codec) | per-horizon **threshold θ** | ❌ | ⚠️ **cost-aware** (latency vs compute) | en | ✅ |
| **French terminality** (2024) | ✅ wav2vec2+FlauBERT | **offline** segment classifier | ❌ | ❌ | **fr** | ✅ code+data |
| **Ours** | ✅ Smart Turn v3 + CamemBERT | **optimal-stopping** on cost | ✅ **ECE** | ✅ **explicit cost model** | **fr, streaming** | — |

## 4. Three differentiators (each vs a named baseline)
1. **vs FastTurn / JAL-Turn** (they fuse, but decide by argmax / τ=0.5, uncalibrated):
   we add **calibration** + an **optimal-stopping decision**.
2. **vs Endpoint Anticipation** (the only cost-aware system, but acoustic-only, uncalibrated,
   per-horizon threshold): we add **lexical fusion**, **calibration**, and a **sequential
   optimal-stopping** formulation with an explicit interruption-vs-latency cost.
3. **vs French terminality** (French + fusion, but offline segment classification):
   we do **streaming real-time endpointing** + calibration + cost.
4. **Field-wide gap** (survey: 72% of works don't compare to prior; ⅓ use no public corpus;
   nobody calibrates): a **calibrated, comparable evaluation** is itself a contribution.

## 5. Metrics (aligned to the field + our thesis)
| Metric | Definition | Role |
|---|---|---|
| **Cut-in / False-Alarm rate** | FP / (FP+TN) — non-endpoints wrongly triggered (interruptions) | primary error ↓ |
| **Miss rate** | FN / (TP+FN) — real turn-ends missed (→ added latency) | error ↓ |
| **F1 (shift class)** | harmonic mean precision/recall on end-of-turn | classification quality |
| **Endpoint latency** | ms from true end-of-turn to trigger; median + **p90** | speed ↓ |
| **Operating curve** | latency **vs** cut-in, swept | **headline claim (Pareto)** |
| **Calibration (ECE)** | expected calibration error + reliability plot | **our differentiator** |
| **Efficiency** | ms/decision on CPU, params, **no LLM** | vs heavy SOTA |

**Headline claim** = our optimal-stopping rule **Pareto-dominates** the baselines on
(latency vs cut-in), **plus** lower ECE (they're uncalibrated) and higher hard-case hold-rate.

## 6. Baselines (reproducible on one GPU)
- **VAP** — `github.com/ErikEkstedt/VAP` — field-standard hold/shift; first baseline.
- **Endpoint Anticipation** — `github.com/bloodraven66/EndpointAnticipation` (+ eval harness
  `Full-Duplex-Bench`) — the cost/latency competitor to beat.
- **French termClassif** — `github.com/ina-foss/termClassif` — French fusion (offline) reference.
- **Own ablations** — acoustic-only (Smart Turn v3), lexical-only (CamemBERT),
  **fusion + fixed threshold** (isolates the optimal-stopping gain), VAD + timeout (incumbent).
- **Benchmarks/data** — FastTurn test set (released), Easy-Turn / smart-turn STurn-v3 (HF),
  TurnGPT (text-only EOU-probability baseline).

## 7. Evaluation plan
1. Sweep the cost ratio `C_cut / C_lat` → traces our operating curve.
2. Plot latency-vs-cut-in for every baseline + ours; show Pareto domination.
3. Reliability diagram + ECE for ours vs baselines (they have none).
4. Per-category hold-rate on hard cases (spelling / number / date / hesitation).
5. Report CPU ms/decision + params (efficiency vs LLM-based SOTA).

## 8. Honest caveats
- **Endpoint Anticipation is close** — it is cost-aware. Our edge is *lexical fusion +
  calibration + true sequential optimal stopping*, not "cost-awareness" alone. State this plainly.
- FastTurn / JAL-Turn / EPA are **2026 pre-prints** — numbers may shift; cite as very recent.
- Everything downstream needs the **labelled audio with `t_end`** (latency + operating curve +
  calibration all derive from it). Data is the gating dependency.

---
*Refs: FastTurn arXiv:2604.01897 · JAL-Turn arXiv:2603.26515 · Endpoint Anticipation
arXiv:2606.13450 · French terminality arXiv:2406.10073 · Survey IWSDS 2025 (Castillo-López et al.).*
