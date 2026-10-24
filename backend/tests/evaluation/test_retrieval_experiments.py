"""Synthetic-only tests for the retrieval experiment runner."""

from __future__ import annotations

import json
import csv
import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

from campusreg.evaluation import (
    ExpectedPage,
    EvaluationQuestion,
    RetrievalExperimentConfig,
    RetrievalExperimentError,
    build_experiment_grid,
    load_experiment_plan,
    load_pdf_manifest,
    run_retrieval_experiments,
)
import campusreg.evaluation.retrieval_experiments as experiments
from campusreg.rag.models import DocumentPage


MINILM = "sentence-transformers/all-MiniLM-L6-v2"
BGE = "BAAI/bge-small-en-v1.5"


def _question(question_id: str, split: str, expected_document: str, text: str) -> EvaluationQuestion:
    return EvaluationQuestion(
        id=question_id,
        question=f"SYNTHETIC TEST FIXTURE {text} question",
        category="direct_factual",
        expected_answer="SYNTHETIC TEST FIXTURE expected answer",
        answerable=True,
        expected_document_ids=(expected_document,),
        expected_pages=(ExpectedPage(expected_document, 1),),
        expected_conflict=False,
        split=split,
    )


def _synthetic_inputs() -> tuple[list[EvaluationQuestion], list[DocumentPage]]:
    questions = [
        _question("synthetic-dev-alpha", "development", "synthetic-alpha", "alpha"),
        # The fake query embedding deliberately points to beta to exercise miss reporting.
        _question("synthetic-val-miss", "validation", "synthetic-alpha", "miss"),
    ]
    pages = [
        DocumentPage(
            page=1,
            text="SYNTHETIC TEST FIXTURE alpha source passage with enough words to index",
            source_name="SYNTHETIC TEST FIXTURE alpha.pdf",
            document_id="synthetic-alpha",
        ),
        DocumentPage(
            page=1,
            text="SYNTHETIC TEST FIXTURE beta source passage with enough words to index",
            source_name="SYNTHETIC TEST FIXTURE beta.pdf",
            document_id="synthetic-beta",
        ),
    ]
    return questions, pages


class SyntheticEmbedder:
    """Distinct deterministic vectors for synthetic topics and model IDs."""

    def __init__(self, model_name: str, constructed: list[str]) -> None:
        self.model_name = model_name
        self.constructed = constructed
        constructed.append(model_name)
        self._dimension = 2 if model_name == MINILM else 3

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            if "miss" in lowered or "beta" in lowered:
                axis = 1
            else:
                axis = 0
            vector = [0.0] * self._dimension
            vector[axis] = 1.0
            vectors.append(vector)
        return np.asarray(vectors, dtype=np.float32)


def test_default_plan_is_exact_18_configuration_retrieval_grid() -> None:
    configs = build_experiment_grid()
    assert len(configs) == 18
    assert {config.embedding_model for config in configs} == {MINILM, BGE}
    assert {config.chunk_size for config in configs} == {80, 120, 200}
    assert {config.chunk_overlap for config in configs} == {30}
    assert {config.top_k for config in configs} == {1, 3, 5}
    baseline = [config for config in configs if config.is_baseline]
    assert len(baseline) == 1
    assert baseline[0].embedding_model == MINILM
    assert (baseline[0].chunk_size, baseline[0].chunk_overlap, baseline[0].top_k) == (120, 30, 3)


def test_checked_in_plan_has_no_generation_parameters() -> None:
    root = Path(__file__).parents[3]
    configs, rule = load_experiment_plan(root / "evaluation" / "retrieval-experiment.config.json")
    plan = json.loads((root / "evaluation" / "retrieval-experiment.config.json").read_text())
    assert len(configs) == 18
    assert rule["primary"] == {"metric": "recall_at_3", "direction": "max"}
    assert "temperature" not in plan
    assert "gemma_model" not in plan
    assert plan["generation"] == "excluded"


def test_runner_separates_embedding_models_preserves_citations_and_resumes(tmp_path: Path) -> None:
    questions, pages = _synthetic_inputs()
    configs = [
        RetrievalExperimentConfig(model, 8, 2, top_k)
        for model in (MINILM, BGE)
        for top_k in (1, 3)
    ]
    constructed: list[str] = []

    def factory(model_name: str) -> SyntheticEmbedder:
        return SyntheticEmbedder(model_name, constructed)

    first = run_retrieval_experiments(
        questions,
        pages,
        output_dir=tmp_path / "run",
        configs=configs,
        embedder_factory=factory,
    )

    assert first["observation_count"] == 8
    assert len(constructed) == 2
    assert set(constructed) == {MINILM, BGE}
    assert all(row["status"] == "complete" for row in first["results"])
    dimensions = {row["embedding_model"]: row["model_dimension"] for row in first["results"]}
    assert dimensions == {MINILM: 2, BGE: 3}
    assert len(list((tmp_path / "run" / "cache").glob("index-*.npy"))) == 2
    with first["results_csv_path"].open("r", encoding="utf-8", newline="") as csv_file:
        csv_rows = list(csv.DictReader(csv_file))
    assert len(csv_rows) == 8
    assert {row["split"] for row in csv_rows} == {"development", "validation"}
    assert len({row["dataset_fingerprint"] for row in csv_rows}) == 1
    assert len({row["corpus_fingerprint"] for row in csv_rows}) == 1
    run_config = json.loads(first["experiment_config_path"].read_text(encoding="utf-8"))
    assert run_config["generation_evaluation_included"] is False
    assert run_config["held_out_test_used_for_selection"] is False
    best = json.loads(first["best_configuration_path"].read_text(encoding="utf-8"))
    assert best["status"] == "selected"
    assert best["split"] == "validation"
    assert "held-out test" in best["rationale"]

    raw = [json.loads(line) for line in first["observations_path"].read_text().splitlines()]
    alpha_hit = next(row for row in raw if row["question_id"] == "synthetic-dev-alpha")
    assert alpha_hit["retrieved_sources"][0]["document_id"] == "synthetic-alpha"
    assert alpha_hit["retrieved_sources"][0]["document_name"] == "SYNTHETIC TEST FIXTURE alpha.pdf"
    assert alpha_hit["retrieved_sources"][0]["page_number"] == 1
    assert alpha_hit["retrieved_sources"][0]["text"].startswith("SYNTHETIC TEST FIXTURE alpha")

    failures = [
        json.loads(line)
        for line in first["failure_cases_path"].read_text().splitlines()
    ]
    assert len(failures) == 2
    assert {row["question"] for row in failures} == {questions[1].question}
    assert {row["experiment_id"].rsplit("-k", 1)[-1] for row in failures} == {"1"}
    assert all(row["retrieved_sources"][0]["document_id"] == "synthetic-beta" for row in failures)

    second = run_retrieval_experiments(
        questions,
        pages,
        output_dir=tmp_path / "run",
        configs=configs,
        embedder_factory=factory,
    )
    assert second["observation_count"] == 8
    assert len(constructed) == 2  # all successful observations are cached


def test_chunk_and_corpus_changes_rebuild_the_index(tmp_path: Path) -> None:
    questions, pages = _synthetic_inputs()
    constructed: list[str] = []
    factory = lambda name: SyntheticEmbedder(name, constructed)
    output = tmp_path / "cache-invalidation"

    first = [RetrievalExperimentConfig(MINILM, 8, 2, 1)]
    run_retrieval_experiments(questions, pages, output_dir=output, configs=first, embedder_factory=factory)
    assert len(constructed) == 1

    changed_chunks = [RetrievalExperimentConfig(MINILM, 10, 2, 1)]
    run_retrieval_experiments(
        questions, pages, output_dir=output, configs=changed_chunks, embedder_factory=factory
    )
    assert len(constructed) == 2

    revised_pages = [
        DocumentPage(
            page=page.page,
            text=page.text + " SYNTHETIC TEST FIXTURE revised source version",
            source_name=page.source_name,
            document_id=page.document_id,
        )
        for page in pages
    ]
    run_retrieval_experiments(
        questions, revised_pages, output_dir=output, configs=changed_chunks, embedder_factory=factory
    )
    assert len(constructed) == 3


def test_query_dimension_mismatch_is_reported_as_failure_not_compared(tmp_path: Path) -> None:
    questions, pages = _synthetic_inputs()

    class MismatchedEmbedder:
        dimension = 2

        def encode(self, texts: Sequence[str]) -> np.ndarray:
            if "question" in texts[0].lower():
                return np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
            return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

    config = [RetrievalExperimentConfig(MINILM, 8, 2, 1)]
    result = run_retrieval_experiments(
        questions,
        pages,
        output_dir=tmp_path / "dimension-mismatch",
        configs=config,
        embedder_factory=lambda _: MismatchedEmbedder(),
    )
    observations = [
        json.loads(line)
        for line in result["observations_path"].read_text().splitlines()
    ]
    assert all(item["status"] == "error" for item in observations)
    assert all(item["failure_stage"] == "retrieval" for item in observations)
    assert all(item["error_type"] == "ValueError" for item in observations)
    assert all(item["retrieved_sources"] == [] for item in observations)


def test_expected_pages_must_exist_in_selected_corpus() -> None:
    questions, pages = _synthetic_inputs()
    questions[0] = replace(questions[0], expected_pages=(ExpectedPage("synthetic-alpha", 2),))
    with pytest.raises(RetrievalExperimentError, match="expected PDF pages are missing"):
        run_retrieval_experiments(
            questions,
            pages,
            output_dir=Path("unused-synthetic-output"),
            configs=[RetrievalExperimentConfig(MINILM, 8, 2, 1)],
            embedder_factory=lambda _: pytest.fail("input validation must happen before model creation"),
        )


def test_pdf_manifest_hash_binds_extracted_pages_to_source_version(tmp_path: Path, monkeypatch) -> None:
    content = b"SYNTHETIC TEST FIXTURE PDF bytes"
    pdf_path = tmp_path / "synthetic-source.pdf"
    pdf_path.write_bytes(content)
    manifest = tmp_path / "corpus-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "document_id": "synthetic-manifest-source",
                        "filename": "SYNTHETIC TEST FIXTURE source.pdf",
                        "path": pdf_path.name,
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        experiments,
        "read_pdf_bytes",
        lambda _bytes, filename, *, document_id: [
            DocumentPage(1, "SYNTHETIC TEST FIXTURE extracted page", filename, document_id)
        ],
    )

    pages, revisions = load_pdf_manifest(manifest)
    assert pages[0].document_id == "synthetic-manifest-source"
    assert pages[0].page_number == 1
    assert revisions[0]["sha256"] == hashlib.sha256(content).hexdigest()

    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_data["documents"][0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(manifest_data), encoding="utf-8")
    with pytest.raises(RetrievalExperimentError, match="SHA-256 mismatch"):
        load_pdf_manifest(manifest)
