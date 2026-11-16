# Log book, COMP7300

**Jean-Baptiste Azan** · MSc Artificial Intelligence, University of Kent
**Project:** Towards Semantically-Aware End-of-Utterance Detection for Conversational AI Systems
**Supervisor:** Dr Palaniappan Ramaswamy · **Industry partner:** Tennor

---

## April

**17 April. Topic chosen and approved.**
I picked a problem I kept watching go wrong at Tennor. The agent cuts callers off when they
spell a name or dictate a number in blocks, which on a medical line is more than an annoyance.
I wanted the project pinned to that setting from the start. Approved on the first pass, so
the scope was settled early.

**Late April. Plan reviewed in writing, then a call.**
Dr Palaniappan Ramaswamy read the initial plan and sent back two concerns: whether the dataset
was real, and novelty. We spoke at 11am the following Monday to go through it. He was blunt on
the second point. Assembling a framework of existing techniques is undergraduate work; an MSc
is expected to produce something measurably better than a baseline. After that call I stopped
optimising for a clean codebase Tennor could deploy and started asking what would let me beat a
baseline on a number. The effort moved to the evaluation and the decision layer. The shipped
path stayed the simple rule-based fusion.

---

## May and June

**May to June. Reading, review, feasibility.**
I was running three things at once. Reading turn-taking detection, which produced the Chapter 2
comparison; what stuck was that no available system reports the calibration of its own
probability or ties its threshold to a cost. Preparing the critical review. And working out
with Tennor what could actually be captured from the production agent and under what
constraints.
That last thread produced the legal and governance position the dissertation states: what could
be captured, under what conditions, and what could never leave Tennor's infrastructure. At this
stage that was a restriction on the production data and nothing more. I had no other French
corpus to work on, so the plan was still to train on what Tennor could give me.
Next: get the capture specified precisely enough for Tennor to sign it off.

**12 June. Critical review submitted.**
Zink et al., *Predictive Speech Recognition and End-of-Utterance Detection* (arXiv:2409.19990).
My synthesis proposed three things: combine the acoustic endpoint with a meaning-based
completeness check, attach a confidence measure and act early only when confident, and evaluate
on speech as it arrives instead of whole recordings. Those became the three pillars of the
project. I did not plan them that way. I noticed the match once the design had settled.

**17 June. LiveKit releases Turn Detector v1 and eot-bench.**
Five days after the review, and it overlapped with my angle. Changing subject was never on the
table: the topic was approved in April and there was no time to restart. The choice was to
reproduce what LiveKit had released or to find the part it did not address. I went for the
second and narrowed the claim to calibrating the probability and deriving the endpoint from a
stated cost.

---

## Late June and July

**26 June. Pipeline skeleton.**
Goal: a runnable end-to-end path before choosing any model.
Written: `session.py`, `orchestrator.py`, `types.py`, the `VAD`, `ASR`, `AcousticEOU`,
`LexicalEOU` and `Fusion` interfaces, `ring_buffer.py`, `endpoint/controller.py`,
`fusion/rules.py`, 14 test files, and a design plan
(`plans/2026-06-26-semantic-eou-detector.md`, deleted on 18 August once the docs replaced it).
Every stage sits behind an abstract interface, so a component can be swapped without touching
the decision logic. A monolithic pipeline would have been faster to write and impossible to
A/B. This part of the design never needed reworking afterwards.
The schedule was another matter. The production data collector went live much later than
planned, partly the holiday period at Tennor, partly a slipped internal review cycle, and
everything downstream of it moved with it.
Next: put a real lexical model behind the `LexicalEOU` interface.

**3 July. First CamemBERT fine-tune. It failed.**
Goal: a lexical model that separates complete from incomplete utterances.
From `notebooks/camembert-eou/checkpoint-15/trainer_state.json`: 5 epochs, 15 optimisation
steps, eval loss 0.6937 down to 0.6864, best checkpoint at step 3. The evaluation set held 15
examples, 7 of them positive. At every checkpoint the model predicted "complete" for all 15
(TP 7, FP 8, TN 0, FN 0, recall 1.0), so its accuracy of 0.467 is simply the positive base
rate. That is a constant classifier, not a decision.
I still do not know why. The dataset was small. I am not experienced at fine-tuning and the
procedure may have been wrong. CamemBERT may not be the right model for judging completeness
from a truncated prefix.
What changed after this: I stopped treating a model I trained myself as the contribution, and
started asking how much could be pulled out of components that already exist. I did keep
fine-tuning, and the 28 July run survives, but only as one branch of a composed system.
I did not record what I changed between this run and the 28 July one, and I cannot reconstruct
it now. That is a gap in this log book.
Next: rebuild the dataset properly and try again with more examples.

**8 July. First working package.**
Two weeks of work pulled together into one runnable package: scaffolding first (README,
`demo.py`, requirements), then the 48 files of the detector itself.

**23 July. Docs and the export script.**
`scripts/export_eou_dataset.sh` and the first design documents.

**26 and 27 July. Lexical dataset package.**
`build_lexical_dataset.py` (prefix-truncation labelling, Azure-style normalisation, hard-case
tagging for spelling, numbers and dates) and `baselines.py` (heuristic and TF-IDF).
Labelling by prefix truncation is cheap and scalable. The cost is that a truncated prefix is
not always a genuine mid-turn pause.

**28 July. Second CamemBERT fine-tune, kept.**
Produced `models/camembert-eou`, fitted on Tennor medical French. It is the model in the live
demo and the model behind the 0.789 row of Table 5.1.
That row is a fusion score, not a score for the lexical branch on its own, and at 0.789 it
sits below the 0.814 acoustic baseline. A medical-fitted lexical read applied to generic French
makes the decision worse. So the model I kept here is also the project's first negative result.
I only understood that in August, when the benchmark comparison was run.
What changed since 3 July is not recorded. See the note above.

**30 July. eot-bench integrated, data strategy reversed.**
The Tennor restriction had been known since June. What I did not have until now was a French
corpus anyone outside the company could obtain. The harness is Apache-2.0 and its dataset
CC-BY-4.0, both usable and redistributable, so training and development moved onto the public
benchmark and the production data was demoted to an in-domain validation set.
The cost is that the headline result is measured on generic French, not medical French. The
gain is that every headline number is reproducible by anyone.
I put this to Dr Ramaswamy in writing on Teams the same week and he agreed the validation set
was the better setup. There was no call between 27 July and 17 August.

**31 July. Sequential stopping. Negative result.**
Goal: fold elapsed silence into the probability, so that a single threshold behaves as a
time-varying one, and beat the harness sweep.
Result: AUC 0.814 to 0.811 on the acoustic baseline. No improvement.
This one was disappointing, because the avenue closes with it. Folding time into the
probability was the obvious route to a sequential rule and the result says the route is blocked
at this level of score quality. What survives is the observation, made later, that the sweep
buys conservatism through timing and not through the threshold, which is why the full
sequential formulation is left as future work.
Next: stop trying to beat the sweep and go back to improving the score itself.

---

## August

**1 to 13 August. Fusion experiments.**
Nothing structural landed in this window; the work was experimental. This is
where the fusion variants were tried against each other before the evaluation layer was written
up on the 19th: the rule-based average, the learned logistic, TF-IDF lexical against
frozen-embedding lexical, and the dialogue-context variant that became the second negative
result.
These are the variants reported in Table 5.1 and in Section 5.6.

**14 August. Corpus dataset.**

**16 August. The two demonstrations.**
`demo.py` (microphone or replayed wav) and `webdemo.py` (browser). Both drive the same
`EouSession`, so there is a single decision path.

**18 August. Fine-tuning notebook, full test run, revised timeline.**
All 45 tests across the 14 test files run. Revised nine-week plan sent to Dr Ramaswamy. The
June design plan was dropped once the docs replaced it, and the notebook was stripped of its
outputs before being filed.

**19 August. The evaluation layer.**
The files carry this date, but the work behind them ran through the whole month. The date is
when it was tidied and written up.
Files: `indist_fusion.py` (TF-IDF), `indist_fusion_camembert.py` (frozen embeddings),
`indist_fusion_context.py`, `calibration_study.py`, `decision_layer.py`,
`eot_fusion_adapter.py`, `fusion/logistic.py`, plus the design and architecture documents.
Results settled over August: fusion 0.897 against an acoustic baseline of 0.814; the
context-augmented variant at 0.854, the second negative result; and every external detector on
the leaderboard miscalibrated, with none of them reporting it.

**20 August. Ethics statement and demonstration video.**
`docs/ethics.md` written, and the demonstration recorded.

**21 August. Corpus submitted.**

**23 August. Plain-language write-up of the method** (`docs/how-it-works.html`).

**29 and 30 August. The in-domain export and the gold result.**
Exported 709 production calls with audio; 636 of them yield 5,342 scored micro-pauses, 1,015
ends of turn against 4,327 holds.
Obstacle: the benchmark harness reads audio through the Hugging Face `Audio` feature, which
requires `torchcodec`, which does not load against `ffmpeg` 8. Pinning the whole toolchain
backwards was not worth it, so I read the recordings directly. Cost, about a day.
Result: acoustic 0.49, lexical 0.60, fusion 0.59. The acoustic branch is at chance and fusing
is worse than the lexical branch alone.
Diagnosis: 8 kHz narrowband against a model trained on 16 kHz wideband, plus a ceiling imposed
by weak supervision. At this point the bandwidth story was an inference from the sample rate; I
measured it on 2 September. I ruled out bugs first, checking the caller channel and the
timestamp alignment against the benchmark path.

**31 August. Per-case breakdown.**
Tagged every silence on the benchmark by the causal caller text before it and recomputed the
AUC within each subgroup. Fusion gains 0.050 on dictated digits against 0.094 on ordinary text,
and the benchmark holds only six spelling cases and four dates in 1,054 silences. The
motivating cases are where the lexical signal helps least, and the public benchmark cannot test
them. Results written up in `docs/results.html` and `docs/thesis-draft.md`.
This contradicts the framing of the project and I only reached it in the final week, for the
same reason as the in-domain result two days earlier: the production collector went live far
too late. Had it been running in July, both measurements would have been available while there
was still time to act on them. This is the single decision I would most want back.

---

## September

**1 September. Dissertation drafting starts.**

**2 September. Bandwidth measured.**
`eou_detector/eval/bandwidth_check.py`. Across sixty in-domain recordings, 0.32 % of the energy
sits above the 3.4 kHz telephone cut-off and 0.007 % above 4 kHz. This is the measurement
behind the claim made on 30 August, which until then rested on the sample rate alone. Started
this log book the same day.

**3 September. Abstract.**

---

## Supervision record

Weekly written updates on Teams throughout, plus these calls:

| Date | Direction | Length |
|---|---|---|
| Late April, 11am Monday | outgoing | — |
| 27 July | incoming | 17 min |
| 17 August | outgoing | 12 min |

The late-April call followed a written review of the plan, which is where the two concerns
first arrived. The written exchanges carried most of the substance: the concerns about dataset
and novelty, the move of training and development onto the public benchmark on 30 July, the
request that the accuracy gain be demonstrated statistically instead of asserted, the structure
of the dissertation, and the ethics question about the scope of the production capture.

---

# Final evaluation

**How I approached this piece of work:**
I started from a problem I kept seeing go wrong in production at Tennor, not from a gap in the
literature, and that fixed both the domain and the constraints from the first week. I built the
pipeline before choosing any model, so components could be swapped without touching the
decision logic. In late July I reversed the data strategy, once it was clear that only the
public benchmark could support a result anyone else could check. And after my supervisor's
early challenge on novelty I stopped optimising for a codebase clean enough for Tennor to
deploy, and started asking what would let me measure something past the existing baseline.

**What I found fairly easy was:**
Writing the code. I already knew the Tennor modules I was reusing, so the engineering was
rarely the obstacle. The difficulty was the layer above it.

**What I found most difficult was:**
Deciding what to build. Working out which code would actually move the result past the baseline
took most of the project's time. Two things made that harder. I am not experienced at
fine-tuning, and my first attempt on 3 July produced a constant classifier, which cost me time
before I understood that composing pre-trained components was the better route. And the
bandwidth mismatch between the 16 kHz acoustic model and the 8 kHz production audio only
surfaced at the end of August, with no time left to act on it.

**If I were to do the work again, I would do the following differently:**
I would instrument production in the first week instead of the seventh. The collector went live
late, and everything depending on it arrived in the final days: the in-domain result, and the
per-case breakdown showing that the public benchmark cannot test the cases the project exists
for. Both would have changed the shape of the work had I seen them in July. I would also check
the audio bandwidth of the target deployment before choosing an acoustic model, instead of
discovering the mismatch at validation. And I would run the per-case breakdown early; knowing
in week three that the benchmark contains six spelling cases would have told me at once what it
could and could not measure.

**What I learned is:**
That a probability has to be calibrated before a threshold placed on it means anything, and
that almost nobody checks. That a component fitted for one domain loses on another, which this
project measured twice before I believed it: the 0.789 fusion row and the in-domain collapse
to 0.59. And that a properly diagnosed negative result is worth more than a weak positive one.
I got more out of the in-domain failure than out of the benchmark gain.

**I would like specific feedback on:**
Whether the contribution is genuinely relevant against the state of the art. The field moved
during the project, and I narrowed the claim to calibration and the cost-based decision because
that is what the released systems still do not do. I would like to know whether that reads as a
real contribution or as a retreat.

**I felt this way about the work:**
The worst stretch was mid-June, when LiveKit released a system that overlapped my original
angle five days after I submitted the review. I had no better idea to replace it with, so I
kept working the problem until the part that was still mine became clear. The final week was
uncomfortable in a different way: the two results I care most about both arrived too late to
act on, and I had to write them up as findings instead of as a starting point.
