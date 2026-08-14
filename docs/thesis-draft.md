# Towards Semantically-Aware End-of-Utterance Detection for Conversational AI Systems

> **MSc dissertation — methodology & results draft.**
> This is a scaffold: structure + the results we already have (real numbers from the
> experiments), with `[EXPAND]` markers for prose to write and `[TODO]` for the gold
> results still pending. Numbers are from the eot-bench French subset unless stated.

---

## Abstract  `[EXPAND — write last]`
One paragraph: problem (endpointing latency/interruption trade-off), approach
(lightweight fusion of off-the-shelf acoustic + lexical, calibrated, cost-sensitive
decision), key results (fusion beats acoustic-only and reaches commercial-grade AUC;
calibration is a universal gap), and the production-domain validation.

## 1. Introduction
- Real-time endpointing forces a **latency ↔ interruption** trade-off. `[EXPAND]`
- VAD/silence-timeout methods cut **semantically-incomplete** utterances (name
  spelling, dictated numbers/dates, thinking pauses) — frequent and high-stakes in
  medical phone intake. `[EXPAND]`
- **Contribution (scoped honestly):**
  1. A lightweight fusion of off-the-shelf acoustic (Smart Turn v3) + lexical
     (CamemBERT) that reaches commercial-grade ranking quality **without fine-tuning
     or an LLM in the loop**.
  2. A calibration analysis showing that **every** SOTA turn detector is
     miscalibrated, and a cost-sensitive decision that sets the operating point from
     an explicit interruption/latency cost.
  3. A **weak-supervision labelling** scheme (resume-after-pause) and a **French
     medical-telephony** evaluation from real production data. `[TODO: gold]`

## 2. Related Work
| System | Signals | Decision | Calibrated | Notes |
|---|---|---|---|---|
| FastTurn / JAL-Turn (2026) | acoustic + lexical fusion | argmax / fixed τ | no | LLM / cross-attention, trained joint |
| LiveKit turn-detector + eot-bench (2026) | text (+ VAD) | swept-threshold Pareto | no | the benchmark we adopt |
| Endpoint Anticipation (2026) | acoustic only | per-horizon threshold | no | cost-aware operating point |
| French terminality (2024) | audio + text | offline classifier | no | French, TV/radio, not streaming |
| **This work** | acoustic + lexical | cost-sensitive on calibrated prob | **yes** | lightweight, off-the-shelf, French medical |
- The fusion mechanism and the Pareto operating-curve framing are **not** novel
  (crowded 2026 area). The gaps we address: **calibration** (none report it),
  **lightweight off-the-shelf composition**, and the **French medical domain**. `[EXPAND]`

## 3. Method
- **Pipeline.** Silero VAD gates the stream; at each micro-pause the recent audio is
  scored by Smart Turn v3 (`p_ac`) and the caller's partial transcript by a lexical
  model (`p_lex`); the two are fused into `p_eot` and a cost-sensitive rule decides
  STOP/WAIT. `[EXPAND + architecture figure — see docs/architecture.html]`
- **Lexical model.** CamemBERT — (a) fine-tuned on historical transcripts (in-domain,
  §5.0); (b) frozen embeddings + logistic head for the benchmark (in-distribution).
- **Fusion.** Learned logistic on `[p_ac, p_lex, p_ac·p_lex]`, fit per fold.
- **Calibration.** Post-hoc isotonic regression; measured with Expected Calibration
  Error (ECE). `[EXPAND — why calibration enables a cost-based threshold]`
- **Cost-sensitive decision.** Given cost ratio λ = C_cut / C_lat, the min-cost
  operating point is chosen from the trade-off (endpoint derived, not a fixed timeout).
- **Weak-supervision labelling.** Production pauses labelled `hold`/`eot` from
  resume-after-pause (caller resumed → hold; agent replied, no resume → eot); no
  manual annotation. `[EXPAND]`

## 4. Data
- **eot-bench-data** (LiveKit; harness Apache-2.0, data CC-BY-4.0) — training/dev set.
  French subset: 400 turns / 1054 silence spans (400 eot, 654 hold).
- **Tennor production (gold)** — validation set. In-house voice pipeline instrumented
  to capture, per micro-pause, the caller-only text, ASR partial trajectory,
  timestamps, audio reference, and resume/interruption signals; GDPR-gated behind an
  org feature flag; auto-labelled. `[TODO: size + results when volume ready]`
- **Historical transcripts** — 549 calls → a fini/pas_fini lexical dataset by
  prefix-truncation (§5.0). Weak labels (the current system's turn ends).

## 5. Experiments & Results
Evaluation via the eot-bench harness: it sweeps the endpointing policy and reports
the **latency vs false-cutoff (cut-in)** Pareto frontier, plus per-span **AUC**
(finished vs still-speaking). All lexical/fusion numbers are **k-fold, no leakage**.

### 5.0 Lexical baseline (historical transcripts, in-domain)
CamemBERT fine-tuned on the fini/pas_fini set: **89% accuracy / F1 0.89, cut-in 12%**
on a held-out-by-call split (heuristic floor 63%, TF-IDF 81%).
Caveat: weak labels + mild over-fit (val loss rises after epoch 2). `[EXPAND]`

### 5.1 Fusion — helps once the lexical is in-distribution
| Model (eot-bench FR) | AUC | lat @ 5% cut-in | lat @ 10% |
|---|---|---|---|
| Smart Turn v3 (acoustic-only) | 0.814 | 873 ms | 608 ms |
| + fusion, OOD medical lexical | 0.789 | 850 ms | 659 ms |
| + fusion, in-dist TF-IDF lexical | 0.876 | 759 ms | 548 ms |
| **+ fusion, in-dist CamemBERT (frozen emb + logistic)** | **0.900** | 715 ms | 510 ms |
| Soniox (commercial STT) | 0.888 | 557 ms | 463 ms |
| LiveKit v1 (commercial) | 0.938 | 635 ms | 376 ms |
- The medical CamemBERT is **out-of-distribution** on generic French and *hurts*;
  an in-distribution lexical + learned fusion **beats acoustic-only** and reaches
  **AUC 0.900, above Soniox (0.888)** — with **no fine-tuning**.
- **Honest:** we lead on AUC but trail on **latency** (the commercial streaming STT
  commit earlier). `[EXPAND]`

### 5.2 Calibration — a universal gap
| Model | ECE (raw) | ECE (calibrated) |
|---|---|---|
| OpenAI GPT-Realtime 2 | 0.374 | 0.007 |
| Soniox | 0.294 | 0.020 |
| AssemblyAI | 0.290 | 0.031 |
| Smart Turn v3 | 0.216 | 0.079 |
| LiveKit Turn Detector v1 | 0.091 | 0.052 |
| Deepgram Flux | 0.087 | 0.083 |
**10/10** leaderboard models are miscalibrated (ECE > 0.05); post-hoc calibration
lowers ECE for all. Our learned fusion is well-calibrated by construction
(ECE 0.06). `[EXPAND — calibration is the prerequisite for a cost-based decision]`

### 5.3 Cost-sensitive operating point
| λ = C_cut / C_lat | cut-in | latency |
|---|---|---|
| 1 (interruption cheap) | 26.7% | 354 ms |
| 5 | 7.2% | 699 ms |
| 10 | 1.8% | 1080 ms |
| 50 (interruption costly) | 0.9% | 1285 ms |
The operating point is **derived from the cost**, not a fixed 500 ms timeout. A
calibrated cost-derived threshold tracks the swept oracle at moderate λ; at high λ
the oracle turns conservative through timing (action-delay/timeout), motivating a
**sequential** formulation (§7). `[EXPAND]`

### 5.4 Negative results (reported for honesty)
- **Time-aware "sequential" threshold** (folding silence into the probability) did
  **not** beat the harness sweep (0.814 → 0.798): the frontier is bounded by the
  **score quality**, and the sweep already extracts the temporal policy.
- **Dialogue-context augmentation** of the lexical (concatenated or separate
  embeddings) did **not** help (0.900 → 0.86/0.90): the completion signal is in the
  caller text, and frozen-embedding pooling dilutes it.
- Takeaway: on **generic** French we plateau at ~0.900; further gains need bigger
  models (not our contribution) or **in-domain** data.

### 5.5 Domain validation (gold)  `[TODO]`
Run the same harness on the Tennor French-medical gold set. Hypothesis: generic SOTA
detectors are **out-of-domain on medical dictation** (spelling, numbers, dates) and
our in-domain fusion wins; report per-hard-case hold-rate. `[TODO: numbers]`

## 6. Discussion & Limitations
- Generic-French plateau (~0.900); the **latency** gap to commercial systems reflects
  their larger models, not a methodological limit. `[EXPAND]`
- Weak labels (transcript truncation / resume-after-pause) — the audio gold set is
  the cleaner ground truth. `[EXPAND]`
- The differentiation is the **domain + method + the novel signal** (below), not
  beating commercial systems on a generic benchmark.

## 7. Conclusion & Future Work
- **Gold domain result** as the differentiating capstone.
- **Novel signal — ASR partial-stability dynamics:** how fast the streaming transcript
  stabilises as an *early* end-of-turn cue (lower latency). Unique to our production
  data (partial trajectories captured); untestable on eot-bench (final words only). `[EXPAND]`
- **Sequential optimal stopping:** a proper multi-step cost model (vs the myopic
  threshold) to match the oracle across all cost regimes.

## Appendix — reproducibility
Code: `github.com/jean-baptisteAZAN/EOUDetector`. Eval pipeline: `eou_detector/eval/`
(fusion adapter, calibration study, decision layer). See `docs/reproduce.md`. `[EXPAND]`
