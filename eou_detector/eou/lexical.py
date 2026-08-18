"""Lexical (semantic) end-of-utterance branch — pipe 2.

Homemade, French-first, rule-based completeness scorer over the latest ASR
partial. Fully owned (no third-party model / no license constraint), deterministic,
sub-millisecond. It returns a graded probability that the utterance is COMPLETE
(p_lex) plus an explicit reason, so the fusion gets a real positive signal, not
only a veto.

Design — two layers:
  1. Strong-incomplete (veto): the caller is clearly mid-utterance. p_lex is
     pushed below the fusion veto band (< veto_lex). Covers the POC's three
     false-positive classes: spelling a name, dictating a number/date, trailing
     on a connector / hesitation.
  2. Graded completeness: when not vetoed, score how "finished" the sentence
     looks (terminal punctuation, French closing phrases, length, trailing
     function/auxiliary words) -> p_lex in [0, 1].

Extensible to other languages later by swapping the word lists behind the same
LexicalEOU interface (this implementation is French-only by design for now).
"""
import abc
import re
from typing import Optional, Tuple

from eou_detector.types import LexResult

# --- French lexical resources -------------------------------------------------

# Hesitation / filler: a turn ending here is not finished.
_FILLERS = {
    "euh", "heu", "heuh", "euhm", "hum", "hmm", "hem", "ben", "bah", "bof",
    "enfin", "disons", "genre",
}
# Conjunctions / subordinators.
_CONNECTORS = {
    "et", "ou", "ni", "mais", "or", "donc", "car", "puis", "parce", "comme",
    "lorsque", "quand", "si", "que", "qu", "qui", "quoi", "dont", "où",
}
# Prepositions.
_PREPOSITIONS = {
    "à", "a", "de", "du", "des", "en", "dans", "sur", "sous", "par", "pour",
    "avec", "sans", "chez", "vers", "entre", "depuis", "pendant", "contre",
    "selon", "malgré", "jusque", "jusqu",
}
# Determiners / articles / possessives.
_DETERMINERS = {
    "le", "la", "les", "un", "une", "mon", "ma", "mes", "ton", "ta", "tes",
    "son", "sa", "ses", "ce", "cet", "cette", "ces", "notre", "nos", "votre",
    "vos", "leur", "leurs", "au", "aux", "quel", "quelle", "quels", "quelles",
}
# Subject pronouns / elided forms.
_SUBJECT_PRON = {
    "je", "j", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles",
    "c", "ça", "n", "qu", "l", "m", "t", "s", "d",
}
# Auxiliaries / modals: trailing here usually expects a complement (soft signal,
# not a hard veto -- a bare "oui je peux" can still be complete).
_MODAL_AUX = {
    "suis", "es", "est", "sommes", "êtes", "sont", "ai", "as", "avons", "avez",
    "ont", "vais", "vas", "va", "allons", "allez", "vont", "peux", "peut",
    "pouvons", "pouvez", "peuvent", "veux", "veut", "voulons", "voulez",
    "veulent", "dois", "doit", "devons", "devez", "doivent", "voudrais",
    "voudrait", "aimerais", "aimerait", "faut", "vais",
}
# Number words: an open run (no terminal punctuation) is still being dictated.
_NUMBER_WORDS = {
    "zero", "zéro", "un", "une", "deux", "trois", "quatre", "cinq", "six",
    "sept", "huit", "neuf", "dix", "onze", "douze", "treize", "quatorze",
    "quinze", "seize", "dix-sept", "dix-huit", "dix-neuf", "vingt", "trente",
    "quarante", "cinquante", "soixante", "septante", "huitante", "nonante",
    "cent", "cents", "mille", "million", "milliard",
}
# Phrases that, at the END of an utterance, strongly indicate completion.
_CLOSERS = (
    "au revoir", "bonne journée", "bonne soirée", "bonne nuit", "à bientôt",
    "à demain", "à plus", "merci beaucoup", "merci bien", "merci", "c'est tout",
    "c'est bon", "c'est ça", "c'est exact", "c'est parfait", "c'est noté",
    "d'accord", "très bien", "parfait", "exactement", "tout à fait",
    "s'il vous plaît", "s'il te plaît", "voilà", "oui", "non", "ok", "okay",
    "d'accord merci", "je vous remercie", "avec plaisir", "pas de problème",
)

# Punctuation that ends a sentence.
_TERMINAL = ".?!…"

# Spelling boundary tolerated as "single letter" (handles "M.", "M-").
_SINGLE_LETTER = re.compile(r"^[a-zàâäéèêëïîôöùûüç]$")
# Tokens keep hyphenated compounds together ("rendez-vous", "dix-sept",
# "est-ce") so their tail isn't mistaken for a bare function word.
_TOKEN_RE = re.compile(
    r"[a-zàâäéèêëïîôöùûüç']+(?:-[a-zàâäéèêëïîôöùûüç']+)*|\d+")


def _tokens(text: str):
    return _TOKEN_RE.findall(text.lower().strip())


def _trailing_spelling(toks) -> bool:
    """True if the utterance is trailing on a spelled-out run: >=2 single
    letters ("m a r", "M. A. R") or a hyphen-joined run ("m-a-r-t-i-n")."""
    singles = 0
    for t in reversed(toks):
        if _SINGLE_LETTER.match(t):
            singles += 1
        else:
            break
    if singles >= 2:
        return True
    parts = toks[-1].split("-")
    return len(parts) >= 2 and all(_SINGLE_LETTER.match(p) for p in parts)


class LexicalEOU(abc.ABC):
    @abc.abstractmethod
    def predict(self, text: str) -> LexResult:
        """Partial transcript -> graded P(utterance complete) + veto + reason."""


class FrenchSemanticEOU(LexicalEOU):
    """Homemade French rule-based semantic EOU (pipe 2)."""

    # p_lex values; the strong-incomplete ones sit below the fusion veto band.
    _VETO_P = 0.05
    _NUMBER_P = 0.10
    _SOFT_P = 0.30        # leaning incomplete, but not a hard veto
    _NEUTRAL_P = 0.50

    def check(self, text: str) -> Tuple[bool, str]:
        """Back-compat strong-veto check: (veto, reason)."""
        res = self.predict(text)
        return res.veto, res.reason

    def predict(self, text: str) -> LexResult:
        raw = text.strip()
        if not raw:
            return LexResult(self._NEUTRAL_P, False, "empty")
        toks = _tokens(raw)
        if not toks:
            return LexResult(self._NEUTRAL_P, False, "no_tokens")

        last = toks[-1]
        joined = " ".join(toks)
        is_closer = any(joined == c or joined.endswith(" " + c)
                        for c in _CLOSERS)
        n = len(toks)

        # Explicit sentence-final punctuation from the ASR is a strong, trusted
        # completion signal -- it overrides the veto layer.
        if raw[-1] in _TERMINAL:
            return LexResult(0.85 if is_closer else 0.80, False,
                             "terminal_punctuation")

        # --- layer 1: strong-incomplete (veto) ---
        if _trailing_spelling(toks):
            return LexResult(self._VETO_P, True, "spelling_in_progress")
        if last in _FILLERS:
            return LexResult(self._VETO_P, True, "hesitation")
        if last in _CONNECTORS or last in _PREPOSITIONS or \
                last in _DETERMINERS or last in _SUBJECT_PRON:
            return LexResult(self._VETO_P + 0.03, True, "trailing_function_word")
        if last in _NUMBER_WORDS or last.isdigit():
            return LexResult(self._NUMBER_P, True, "open_number_run")

        # Trailing auxiliary/modal usually expects a complement -> leaning wait,
        # but not a hard veto (a bare "oui je peux" can be complete).
        if last in _MODAL_AUX and not is_closer:
            return LexResult(self._SOFT_P, False, "trailing_aux_verb")

        # --- layer 2: graded completeness ---
        if is_closer:
            return LexResult(0.82, False, "closing_phrase")
        score = 0.45
        if n >= 4:
            score += 0.10
        if n <= 2:
            score -= 0.10
        score = min(0.98, max(0.02, score))
        return LexResult(score, False, "leaning_complete")


class CamembertLexicalEOU(LexicalEOU):
    """Fine-tuned CamemBERT completeness classifier -- the lexical model behind
    the reported fusion result -- optionally guarded by the French heuristic veto
    for the hard incomplete classes (spelling, open numbers, hesitation, trailing
    function words).

    ``p_lex`` = model P(class 'fini' | latest partial transcript). Weights live in
    ``models/camembert-eou`` (config ``id2label {0: pas_fini, 1: fini}``). Torch and
    transformers are imported lazily so the rule-based path stays dependency-light.
    ``predict`` is called off the event loop (in an executor) by ``EouSession``.
    """

    def __init__(self, model_dir: str = "models/camembert-eou",
                 veto_guard: bool = True, max_len: int = 64,
                 device: Optional[str] = None):
        import torch
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer)
        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(model_dir)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self._model.eval()
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._max_len = max_len
        self._pos = self._positive_index()
        # Guard reuses the deterministic French rules for the three false-positive
        # classes the classifier can miss on very short partials.
        self._guard = FrenchSemanticEOU() if veto_guard else None

    def _positive_index(self) -> int:
        id2label = getattr(self._model.config, "id2label", None) or {}
        for k, v in id2label.items():
            if str(v).strip().lower() in ("fini", "complete", "eot", "end"):
                return int(k)
        return 1 if getattr(self._model.config, "num_labels", 2) > 1 else 0

    def predict(self, text: str) -> LexResult:
        raw = (text or "").strip()
        if not raw:
            return LexResult(0.5, False, "empty")
        if self._guard is not None:
            g = self._guard.predict(raw)
            if g.veto:
                return LexResult(g.p_lex, True, f"veto:{g.reason}")
        enc = self._tok(raw, truncation=True, max_length=self._max_len,
                        return_tensors="pt")
        enc = {k: v.to(self._device) for k, v in enc.items()}
        with self._torch.no_grad():
            logits = self._model(**enc).logits[0]
            p = self._torch.softmax(logits, dim=-1)[self._pos].item()
        return LexResult(float(p), False, "camembert")
