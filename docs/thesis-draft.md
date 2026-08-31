# Dissertation plan (for supervisor feedback)

> **Superseded planning artefact, kept as a record.** This is the plan sent to the supervisor
> before writing. Following his feedback the structure was reduced to six chapters (problem
> merged into Background & Related Work; conclusions merged with future work), and the
> in-domain gold validation listed below as pending has since been run: its result is negative
> and is reported in the dissertation. See `results.html` for the current numbers.

**Towards Semantically-Aware End-of-Utterance Detection for Conversational AI Systems**
Jean-Baptiste Azan. MSc Computer Science (AI), University of Kent. Supervisor: Palani. Industry partner: Tennor.

This is my plan before I write it up in prose. I would like your view on the structure, on how I frame the contribution, and on whether anything is missing or should be cut. Target length about 7,500 words.

## The problem
- A medical voice agent answers patient phone calls and has to decide, in real time, when the caller has actually finished speaking.
- Too early: it cuts the patient off. Too late: the conversation feels slow. That is the core trade-off.
- A plain voice-activity detector, or a fixed silence timeout, only hears silence, not meaning. So it cuts people off mid-sentence exactly when they spell a name, give a number or a date in chunks, or pause to think. This is common in medical intake.

## My solution
- Keep the existing VAD as a gate. At each short pause, look at two things instead of one:
  - the sound: an off-the-shelf acoustic model (Smart Turn v3) judges if it sounds like a turn end.
  - the words: CamemBERT reads the live partial transcript and judges if the words form a complete thought.
- Combine the two into a single probability (a small learned formula), make that probability honest (calibration), then decide stop or wait using an explicit cost: how bad is cutting in versus waiting.
- Everything is off-the-shelf: no fine-tuning of the base models, no LLM in the loop. The decision never waits on the transcript (only the sound is on the critical path), so it stays fast.

## What is actually new (scoped honestly)
- The fusion idea itself is not new (busy area in 2026). What I add:
  1. Calibration. I measured that every turn detector on the public benchmark is over or under confident, and that a simple fix corrects all of them. Nobody reports this.
  2. A cost-based decision on top of the calibrated probability, instead of a magic 500 ms timeout.
  3. A real French medical-telephony setting, with data auto-labelled from the live system.

## What I have done, and what I found (public French benchmark)
- Fusion beats sound-only, once the words model fits the data:

| Model (benchmark, French) | AUC | latency @ 5% cut-in |
|---|---|---|
| Smart Turn v3 (sound only) | 0.814 | 873 ms |
| + fusion, wrong-domain words model | 0.789 | 850 ms |
| + fusion, in-domain words model | **0.897** | 715 ms |
| LiveKit v1-mini (commercial) | 0.884 | 798 ms |
| LiveKit v1 (commercial) | 0.937 | 635 ms |

- The gain is real, not noise: +0.083 AUC (0.814 to 0.897), p < 0.0001 (bootstrap resampled by call).
- Note: the streaming STT systems (Soniox, AssemblyAI, OpenAI) cannot be scored on this per-span AUC (score-point misalignment), so they are compared on latency only; Soniox commits earliest (557 ms).
- Calibration: 8 out of 8 external leaderboard models are miscalibrated (13 of 14 overall); my fusion is much better calibrated than them (ECE 0.064 versus 0.2 to 0.37).
- Cost decision: the wait time comes from the cost ratio, not a fixed timer. Cheap interruption gives cut 16.8% / 330 ms; costly interruption gives 0.2% / 1525 ms.
- The pipeline runs live (browser and command-line demo), and 45 automated tests pass.

## Honest problems and limits (I want to keep these in)
- On generic French I plateau around 0.897. Beating that needs bigger models (not my point) or in-domain data.
- I lead on accuracy (AUC) but trail the commercial systems on latency: they commit earlier.
- Weak labels are noisier than human labels, and the French benchmark subset is small (about 1,000 pauses).
- Two things I tried did not work, and I will report them: a time-aware sequential threshold, and adding dialogue context to the words model.

## What is left to do
- The in-domain gold result. I have turned on capture in production; the validation data should be ready in about a week. This is the capstone that shows it generalises to real medical calls.
- Writing the chapters in prose, and completing the references.

## Planned dissertation structure (about 7,500 words)
- 1. Introduction: the problem, my aims (O1 to O5), the contribution, and a chapter overview.
- 2. The problem: the stop/wait decision, the latency versus interruption trade-off, and why silence alone is not enough.
- 3. Literature and technology review: acoustic, lexical and joint turn detection, the benchmark, and calibration methods; where the gap is.
- 4. Approaches I considered and why I chose this one: joint model vs late fusion; fine-tuned vs frozen words model; fixed timeout vs cost decision; myopic vs sequential.
- 5. What I built: the architecture, the method (fusion, calibration, cost), the data, and the engineering.
- 6. Results, testing and verification: the tables above, the significance test, the negative results, and the software tests.
- 7. Conclusions and honest self-evaluation: objectives versus what I achieved, the limits, and what I would do differently.
- 8. Future work: the gold result, the ASR partial-stability signal, and sequential optimal stopping.
- Bibliography, and appendices (reproducibility, ethics, corpus map).

## Questions for you
- Does the contribution framing (calibration first, plus a cost decision, plus the French medical domain) read as strong enough, or should I push a different angle?
- Is the structure right, and is anything missing or worth cutting?
- Am I right to keep the negative results and the honest limits in the write-up?
