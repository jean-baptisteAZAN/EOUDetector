"""eot-bench adapter: fused acoustic + lexical end-of-turn probability.

p_ac  = Smart Turn v3 on the causal audio prefix (reuses eot-bench's shipped adapter).
p_lex = our lexical model on the caller's current (partial) text (from `messages`):
        fine-tuned CamemBERT if EOU_CAMEMBERT_PATH is set, else a French heuristic.
p_eot = fuse(p_ac, p_lex)  (weighted average for now; calibration/logistic come next).

The harness instantiates the adapter with NO args, so config comes from env vars:
  EOU_CAMEMBERT_PATH   path to a fine-tuned CamemBERT dir (optional -> heuristic fallback)
  EOU_W_AC, EOU_W_LEX  fusion weights (default 0.5 / 0.5)

Run (from the eot-bench repo, with this package importable):
  PYTHONPATH=/Users/jean-baptisteazan/Documents/EOUDetector \
  EOU_CAMEMBERT_PATH=/path/to/camembert-eou \
  eot-harness predict --path livekit/eot-bench-data --name fr --split validation \
    --adapter eou_detector.eval.eot_fusion_adapter:FusionAdapter --output-dir output --overwrite
"""
import os
import re


def _last_user_text(messages):
    if not messages:
        return ""
    for m in reversed(messages):
        if (m.get("role") or "").lower() == "user":
            return (m.get("content") or "").strip()
    return ""


# --- lexical scorers (return P(fini) in [0,1] per text) ---

_TRAILING = {
    "euh", "heu", "et", "ou", "donc", "alors", "mais", "que", "qui", "de", "du",
    "le", "la", "les", "un", "une", "mon", "ma", "je", "j", "c'est", "à", "au",
    "pour", "avec", "dans",
}
_NUMBERS = {
    "zero", "zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept",
    "huit", "neuf", "dix", "vingt", "trente", "quarante", "cinquante", "soixante",
    "cent", "mille",
}


class HeuristicLexical:
    """Cheap FR floor: low P(fini) when clearly mid-utterance."""
    name = "heuristic"

    def score(self, texts):
        out = []
        for t in texts:
            toks = re.findall(r"[a-zàâäéèêëïîôöùûüç'0-9]+", (t or "").lower())
            if not toks:
                out.append(0.5)
                continue
            singles = 0
            for w in reversed(toks):
                if len(w) == 1 and w.isalpha():
                    singles += 1
                else:
                    break
            veto = singles >= 2 or toks[-1] in _TRAILING or toks[-1] in _NUMBERS or toks[-1].isdigit()
            out.append(0.1 if veto else 0.6)
        return out


class CamembertLexical:
    """Fine-tuned CamemBERT -> P(fini). Falls back to heuristic if load fails."""
    name = "camembert"

    def __init__(self, path):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).eval()
        # index of the "fini" label
        id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.fini_idx = next((i for i, v in id2label.items() if str(v).lower() == "fini"), 1)

    def score(self, texts):
        import numpy as np
        clean = [t if t else "" for t in texts]
        enc = self.tok(clean, return_tensors="pt", truncation=True,
                       max_length=64, padding=True)
        with self._torch.no_grad():
            logits = self.model(**enc).logits
        probs = self._torch.softmax(logits, dim=-1).cpu().numpy()
        p = probs[:, self.fini_idx]
        # neutral for empty text
        return [0.5 if not t else float(pi) for t, pi in zip(clean, p)]


def _load_lexical():
    path = os.environ.get("EOU_CAMEMBERT_PATH")
    if path and os.path.isdir(path):
        try:
            return CamembertLexical(path)
        except Exception as e:  # noqa: BLE001
            print(f"[FusionAdapter] CamemBERT load failed ({e}); using heuristic")
    return HeuristicLexical()


class FusionAdapter:
    score_point = 0.2

    def __init__(self):
        from eot_harness.smart_turn_adapter import SmartTurnAudioAdapter
        self.st = SmartTurnAudioAdapter()
        self.lex = _load_lexical()
        self.w_ac = float(os.environ.get("EOU_W_AC", 0.5))
        self.w_lex = float(os.environ.get("EOU_W_LEX", 0.5))
        s = self.w_ac + self.w_lex
        self.w_ac, self.w_lex = self.w_ac / s, self.w_lex / s
        self.adapter_id = f"fusion-smartturn-{self.lex.name}-a{self.w_ac:.2f}"

    def predict_batch(self, batch):
        p_ac = self.st.predict_batch(batch)
        texts = [_last_user_text(item.get("messages")) for item in batch]
        p_lex = self.lex.score(texts)
        return [self._fuse(float(a), float(l)) for a, l in zip(p_ac, p_lex)]

    def _fuse(self, p_ac, p_lex):
        # v1: weighted average. Next: calibrated logistic on [p_ac, p_lex, p_ac*p_lex].
        return self.w_ac * p_ac + self.w_lex * p_lex

    def supports_language(self, lang_code):  # lexical is FR-tuned; acoustic is language-agnostic
        return True
