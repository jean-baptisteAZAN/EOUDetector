# Ethics and Data Handling Statement

**Project:** Towards Semantically-Aware End-of-Utterance Detection for Conversational AI Systems
**Author:** Jean-Baptiste Azan
**Industry partner:** Tennor
**Date:** August 2026

## 1. Context and partnership

This MSc project was carried out in partnership with **Tennor**, which operates a
French medical voice agent that patients call by phone. The project builds a
real-time end-of-utterance detector for that agent.

A key point for this statement: the data used to build the in-domain dataset for
this project is **internal test data**, not real patient traffic. Capture is gated
behind an organisation-level feature flag that was enabled only for **internal test
organisations**, so the recordings and transcripts used here come from test calls,
**not from real patients**.

## 2. Data used

**a. Public benchmark (no personal data).** All headline experiments and every
result reported in the dissertation run on the **LiveKit eot-bench** dataset
(licence CC-BY-4.0), a public French turn-taking benchmark. It contains no Tennor
data and no personal data. The benchmark carries the full weight of the evaluation
so that the method can be demonstrated without any patient data.

**b. Tennor internal test data.** A secondary, in-domain dataset was derived from
**internal test calls** to the Tennor agent (feature-flag gated to test
organisations). It was used only to build an in-domain lexical dataset and as an
in-domain validation set. It is **not** included in this corpus.

## 3. Governance

- Tennor is the **data controller** and operates a **GDPR-compliant, ISO/IEC
  27001-certified** environment. The author processed the data within Tennor's
  infrastructure as part of the partnership.
- The work involved **no recruitment of participants** and **no interaction with
  patients**. It reuses internal test data already generated within Tennor's system.

## 4. Data minimisation and anonymisation

- Only the **caller-side** signals needed for the task were captured per micro-pause:
  the partial transcript text, the automatic-speech-recognition partial trajectory,
  timestamps, an audio reference, and resume/interruption flags. No more than needed.
- Identifying details are **masked**: names, first names, and ages are removed from
  the captured data.
- Capture is gated behind an **organisation-level feature flag**, enabled only for
  internal test organisations, so it never ran on real patient traffic.

## 5. Reduced human exposure (weak supervision)

Labels for the in-domain dataset are produced **automatically** by weak supervision
(a pause is labelled `hold` or `eot` from whether the caller resumed speaking), so
building the dataset does **not** require a person to sit and listen to calls. This
limits human exposure to the content.

## 6. Storage and access

- The internal test data and any derived audio or transcripts are stored within
  **Tennor's infrastructure** (GDPR-compliant, ISO/IEC 27001-certified).
- Access was restricted to the **author only**, as internal project tracking.
- No data was copied into this repository, the corpus, or any public location.

## 7. What is in this corpus

This corpus contains **no patient data and no Tennor test data**. It contains: the
source code, the dataset-generation code, the evaluation on the public eot-bench
dataset, the design and planning documents, and this statement. The `.gitignore`
excludes call audio, transcripts, and model weights, and the corpus was packaged so
that none of those files are included.

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Exposure of real patient data | Only internal test data used; capture feature-flag gated to test organisations; never run on real patient traffic |
| Re-identification | Names, first names and ages masked; data kept on Tennor infrastructure; not exported; not in corpus |
| Accidental inclusion of data in the corpus | `.gitignore` rules plus packaging that excludes audio, transcripts and weights; verified before submission |
| Over-collection | Only caller-side, task-relevant fields captured |
| Human exposure to call audio | Automatic weak-supervision labelling instead of manual listening |
| Clinical impact of a wrong cut | Cost-sensitive decision and honest reporting of false-cutoff rates; the system augments, and does not replace, existing safeguards |
