# Reproducing the results

All experiments run through the LiveKit **eot-bench** harness on the public French
subset (no gold data / no PHI required). Every lexical/fusion number is k-fold
(split by call id, no leakage).

**Abbreviations.** AUC = Area Under the ROC (Receiver Operating Characteristic) Curve ·
ECE = Expected Calibration Error · TF-IDF = Term Frequency–Inverse Document Frequency ·
k-fold = k-fold cross-validation · PHI = Protected Health Information · HF = Hugging Face ·
JSONL = JSON Lines · CC-BY-4.0 = Creative Commons Attribution 4.0.

## Setup

```bash
# 1. this repo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # numpy, pandas, scikit-learn, transformers, torch, datasets, soundfile

# 2. the benchmark harness (separate repo, Apache-2.0)
git clone https://github.com/livekit/eot-bench.git
pip install -e "./eot-bench[dev]"
pip install onnxruntime transformers torch soundfile "datasets>=3.2"

# 3. HF token (dataset + model downloads)
export HF_TOKEN=hf_...        # eot-bench-data is CC-BY-4.0
```

`eot-harness` is the command-line interface (CLI). Set `SS=eot-bench/output/livekit__eot-bench-data__validation__min_silence_100ms/fr`
after the first run (the span-set directory).

## 1. Acoustic baseline (Smart Turn v3)

```bash
cd eot-bench
eot-harness predict --path livekit/eot-bench-data --name fr --split validation \
  --adapter eot_harness.smart_turn_adapter:SmartTurnAudioAdapter --output-dir output --overwrite
eot-harness compute-metrics --predictions $SS/smart_turn_audio_adapter__*/predictions.parquet \
  --output-dir $SS/smart_turn_audio_adapter__*/metrics
```
→ AUC 0.814, latency@5% 873 ms, @10% 608 ms.

## 2. In-distribution fusion (the main result)

Uses the shipped Smart Turn run for `p_ac` + a CamemBERT-embedding lexical for
`p_lex`, fused with a k-fold logistic. Run from the repo root with the package on
`PYTHONPATH`:

```bash
python -m eou_detector.eval.indist_fusion_camembert \
  --span-set $SS --ref smart_turn_audio_adapter__<hash>
eot-harness compute-metrics --predictions $SS/indist_fusion_camembert/predictions.parquet \
  --output-dir $SS/indist_fusion_camembert/metrics
eot-harness compare-models $SS        # leaderboard incl. our fusion
```
→ fusion AUC 0.900 (beats Soniox 0.888 on AUC). `indist_fusion.py` is the TF-IDF
variant (0.876).

## 3. Calibration study (all models)

```bash
python -m eou_detector.eval.calibration_study --span-set $SS --ref smart_turn_audio_adapter__<hash>
```
→ ECE (raw vs calibrated) for every model in the span-set (10/10 miscalibrated) +
the calibrated cost-derived threshold vs the swept oracle.

## 4. Decision layer (calibration + cost-sensitive operating point)

```bash
python -m eou_detector.eval.decision_layer --run $SS/indist_fusion_camembert
```
→ ECE of the fused probability + the min-cost operating point per cost ratio λ.

## Module map (`eou_detector/`)

| File | What |
|---|---|
| `dataset/build_lexical_dataset.py` | fini/pas_fini dataset from transcripts (prefix-truncation) |
| `dataset/baselines.py` | heuristic + TF-IDF lexical baselines |
| `dataset/train_camembert.py` | fine-tune CamemBERT (the in-domain lexical) |
| `eval/eot_fusion_adapter.py` | live eot-bench adapter (Smart Turn + lexical → fused p_eot) |
| `eval/indist_fusion.py` / `indist_fusion_camembert.py` | in-distribution fusion experiments |
| `eval/calibration_study.py` | ECE across the leaderboard + cost-threshold check |
| `eval/decision_layer.py` | calibration + cost-sensitive operating point |
| `eval/sequential_stopping.py` | time-aware threshold (negative result) |
| `eval/gold_to_eotbench.py` | production export → eot-bench schema (for the gold validation) |

## Gold validation (when production volume is ready)

```bash
./scripts/export_eou_dataset.sh --since <date> --with-audio        # psql → JSONL + audio
python -m eou_detector.eval.gold_to_eotbench --export data/eou_export_*/eou_calls.jsonl \
  --audio-dir data/eou_export_*/audio --out data/gold_eotbench_fr
# then point the harness --path at data/gold_eotbench_fr and rerun steps 1-4.
```

## Notes
- No PHI in the repo: `data/`, `*.jsonl`, `*.wav`, model weights are git-ignored.
- The LiveKit **turn-detector model** is license-restricted (we do not run its
  weights; its published numbers are cited). The harness + dataset are open.
