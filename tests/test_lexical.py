from eou_detector.eou.lexical import FrenchSemanticEOU, LexicalEOU


def setup_module(module):
    module.LEX = FrenchSemanticEOU()


# --- layer 1: strong-incomplete (veto), the three target false-positive classes


def test_spelling_in_progress_vetoes():
    lex = FrenchSemanticEOU()
    assert lex.check("mon nom c'est m a r")[0] is True
    assert lex.check("M. A. R")[0] is True
    assert lex.predict("mon nom c'est m a r").reason == "spelling_in_progress"


def test_open_number_run_vetoes():
    lex = FrenchSemanticEOU()
    assert lex.check("le numero c'est zero six")[0] is True
    assert lex.check("mon code est quatre vingt")[0] is True
    r = lex.predict("le numero c'est zero six")
    assert r.veto is True and r.reason == "open_number_run" and r.p_lex <= 0.1


def test_trailing_function_word_vetoes():
    lex = FrenchSemanticEOU()
    assert lex.check("je voudrais prendre un")[0] is True       # determiner
    assert lex.check("donc je")[0] is True                      # subject pronoun
    assert lex.check("c'est le numero de")[0] is True           # preposition


def test_hesitation_vetoes():
    lex = FrenchSemanticEOU()
    assert lex.check("alors euh")[0] is True
    assert lex.predict("alors euh").reason == "hesitation"


# --- layer 2: graded completeness (positive signal, not just veto)


def test_complete_sentence_scores_high_and_not_vetoed():
    lex = FrenchSemanticEOU()
    r = lex.predict("je voudrais prendre un rendez-vous demain")
    assert r.veto is False
    assert r.p_lex > 0.5


def test_closing_phrase_scores_high():
    lex = FrenchSemanticEOU()
    assert lex.predict("merci au revoir").p_lex >= 0.8
    assert lex.predict("oui c'est exact").p_lex >= 0.8
    assert lex.predict("d'accord").p_lex >= 0.8


def test_terminal_punctuation_boosts():
    lex = FrenchSemanticEOU()
    assert lex.predict("je voudrais un rendez-vous.").p_lex > \
        lex.predict("je voudrais un rendez-vous").p_lex


def test_trailing_aux_leans_incomplete_without_hard_veto():
    lex = FrenchSemanticEOU()
    r = lex.predict("est-ce que je peux")
    assert r.veto is False
    assert r.p_lex <= 0.35   # leaning wait, but not a hard veto


def test_empty_is_neutral():
    lex = FrenchSemanticEOU()
    r = lex.predict("")
    assert r.veto is False and r.p_lex == 0.5


def test_is_lexical_eou():
    assert isinstance(FrenchSemanticEOU(), LexicalEOU)
