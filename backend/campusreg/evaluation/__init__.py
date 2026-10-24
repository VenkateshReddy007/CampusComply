"""Reproducible evaluation tools for reviewed CampusReg datasets."""

from campusreg.evaluation.dataset import (
    EvaluationDatasetError,
    load_dataset,
    validate_challenge_dataset,
)
from campusreg.evaluation.metrics import compute_metrics
from campusreg.evaluation.models import (
    EVALUATION_CATEGORIES,
    EVALUATION_SPLITS,
    ConversationTurn,
    EvaluationPrediction,
    EvaluationQuestion,
    ExpectedPage,
)
from campusreg.evaluation.runner import EvaluationRun, EvaluationRunner, fingerprint_config

__all__ = [
    "EVALUATION_CATEGORIES",
    "EVALUATION_SPLITS",
    "ConversationTurn",
    "EvaluationDatasetError",
    "EvaluationPrediction",
    "EvaluationQuestion",
    "EvaluationRun",
    "EvaluationRunner",
    "ExpectedPage",
    "RetrievalExperimentConfig",
    "RetrievalExperimentError",
    "build_experiment_grid",
    "compute_metrics",
    "fingerprint_config",
    "load_dataset",
    "load_experiment_plan",
    "load_pdf_manifest",
    "run_retrieval_experiments",
    "validate_challenge_dataset",
]


_RETRIEVAL_EXPERIMENT_EXPORTS = {
    "RetrievalExperimentConfig",
    "RetrievalExperimentError",
    "build_experiment_grid",
    "load_experiment_plan",
    "load_pdf_manifest",
    "run_retrieval_experiments",
}


def __getattr__(name: str):
    """Load optional experiment tooling lazily, including for its module CLI."""

    if name in _RETRIEVAL_EXPERIMENT_EXPORTS:
        from campusreg.evaluation import retrieval_experiments

        return getattr(retrieval_experiments, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
