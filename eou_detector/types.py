from dataclasses import dataclass


@dataclass
class Partial:
    text: str
    is_final: bool = False
    ts_ms: float = 0.0


@dataclass
class LexResult:
    p_lex: float
    veto: bool
    reason: str


@dataclass
class FusionInput:
    p_ac: float
    p_ac_available: bool
    p_lex: float
    p_lex_available: bool
    silence_ms: float


@dataclass
class FusionResult:
    p_eou: float
    decision: str  # "ENDPOINT" | "WAIT"
    required_silence_ms: float
    reason: str


@dataclass
class Decision:
    ts_ms: float
    p_ac: float
    p_lex: float
    p_eou: float
    decision: str
    required_silence_ms: float
    silence_ms: float
    latency_ms: float
    reason: str
    partial_text: str
    lex_reason: str = ""  # which lexical rule fired (spelling/number/closer/...)
