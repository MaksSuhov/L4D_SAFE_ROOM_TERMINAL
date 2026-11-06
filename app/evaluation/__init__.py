from app.evaluation.dataset import build_case, load_cases
from app.evaluation.domain import (
    CaseFailure,
    CaseResult,
    EvalCase,
    JudgeScore,
    Overlap,
    SourceOverlap,
    Timing,
)
from app.evaluation.interfaces import (
    ClockProtocol,
    CompletionProtocol,
    ContextBuilderProtocol,
    EmbeddingProtocol,
    JudgeProtocol,
    SimilarityProtocol,
    StreamingChatProtocol,
)
from app.evaluation.judge import LLMJudge
from app.evaluation.latency import MonotonicClock, measure_stream
from app.evaluation.metrics import content_tokens, source_overlap, stem, text_overlap, tokenize
from app.evaluation.report import EvaluationReport
from app.evaluation.runner import CaseEvaluator, EvaluationService
from app.evaluation.similarity import EmbeddingSimilarity, NullSimilarity

__all__ = [
    "CaseEvaluator",
    "CaseFailure",
    "CaseResult",
    "ClockProtocol",
    "CompletionProtocol",
    "ContextBuilderProtocol",
    "EmbeddingProtocol",
    "EmbeddingSimilarity",
    "EvalCase",
    "EvaluationReport",
    "EvaluationService",
    "JudgeProtocol",
    "JudgeScore",
    "LLMJudge",
    "MonotonicClock",
    "NullSimilarity",
    "Overlap",
    "SimilarityProtocol",
    "SourceOverlap",
    "StreamingChatProtocol",
    "Timing",
    "build_case",
    "content_tokens",
    "load_cases",
    "measure_stream",
    "source_overlap",
    "stem",
    "text_overlap",
    "tokenize",
]
