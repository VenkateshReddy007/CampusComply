"""Paired no-RAG/RAG Gemma evaluation with resumable, human-reviewed outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

from campusreg.config import RAGSettings
from campusreg.evaluation.dataset import load_dataset, validate_challenge_dataset
from campusreg.evaluation.models import EvaluationQuestion
from campusreg.evaluation.retrieval_experiments import load_pdf_manifest
from campusreg.rag.documents import read_pdf_bytes
from campusreg.rag.embeddings import SentenceTransformerEmbedder
from campusreg.rag.errors import GenerationError
from campusreg.rag.generation import GoogleGenAIProvider, TextGenerator
from campusreg.rag.models import AnswerResult, AnswerStatus, RetrievedSource
from campusreg.rag.pipeline import answer as rag_answer
from campusreg.rag.retrieval import build_knowledge_base

NO_RAG_SYSTEM_INSTRUCTION = (
    "You are answering a campus-policy question without official regulation passages. "
    "Do not invent citations or claim that you checked an official source. "
    "Clearly state when a current campus-specific rule cannot be verified."
)
PROMPT_REVISION = "paired-no-rag-rag-v1"
_CITATION_PATTERN = re.compile(r"\[[^\]\n]+\]")
_SECRET_KEY = re.compile(r"api[_-]?key|secret|token|password|credential|authorization", re.I)
_LABEL_TEMPLATE: dict[str, Any] = {
    "answer_correct": None,
    "unsupported_claims_present": None,
    "unsupported_claims_count": None,
    "citation_support": None,
    "evidence_support": None,
    "appropriate_refusal": None,
    "comparison_tag": None,
    "reviewer": None,
    "reviewed_at": None,
    "notes": None,
}
_COMPARISON_TAGS = {
    "rag_helped",
    "no_change",
    "rag_misleading_evidence",
    "rag_incomplete_evidence",
    "no_rag_better",
    "unclear",
}


class PairedEvaluationError(ValueError):
    """Raised when a paired evaluation cannot be compared fairly."""


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _without_secrets(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _without_secrets(item) for key, item in value.items() if not _SECRET_KEY.search(str(key))}
    if isinstance(value, (list, tuple)):
        return [_without_secrets(item) for item in value]
    return value


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as target:
            target.write(content)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    _atomic_write(path, "".join(_canonical(record) + "\n" for record in records))


def _source_record(source: RetrievedSource) -> dict[str, Any]:
    return {
        "document_id": source.document_id,
        "document_name": source.source_name,
        "page_number": source.page_number,
        "chunk_id": source.chunk_id,
        "citation": source.citation,
        "text": source.text,
        "score": source.score,
        "retrieval_method": source.retrieval_method,
        "diagnostics": dict(source.diagnostics),
    }


def _labels(record: Mapping[str, Any]) -> dict[str, Any]:
    value = record.get("human_labels")
    labels = dict(_LABEL_TEMPLATE)
    if isinstance(value, Mapping):
        labels.update(value)
    return labels


def _has_review_metadata(labels: Mapping[str, Any]) -> bool:
    reviewer = labels.get("reviewer")
    reviewed_at = labels.get("reviewed_at")
    if not isinstance(reviewer, str) or not reviewer.strip() or not isinstance(reviewed_at, str):
        return False
    try:
        parsed = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _valid_citations(answer: str | None, sources: Sequence[Mapping[str, Any]]) -> tuple[bool | None, bool | None]:
    if not answer:
        return None, None
    cited = _CITATION_PATTERN.findall(answer)
    if not cited:
        return False, None
    allowed = {source.get("citation") for source in sources}
    return True, all(citation in allowed for citation in cited)


def _call_no_rag(
    generator: TextGenerator,
    question: EvaluationQuestion,
    settings: RAGSettings,
) -> tuple[str, str | None, str | None]:
    """Send only the question string to Gemma; no evidence is accepted here."""

    try:
        text = generator.generate(
            question.question,
            model=settings.gemma_model,
            system_instruction=NO_RAG_SYSTEM_INSTRUCTION,
            temperature=settings.temperature,
        )
    except GenerationError as exc:
        status = "incomplete_response" if exc.code == "empty_generation" else "generation_error"
        return status, None, exc.code
    except Exception:
        return "generation_error", None, "generation_failed"
    if not isinstance(text, str) or not text.strip():
        return "incomplete_response", None, "empty_generation"
    return "answered", text.strip(), None


def _rag_record(
    question: EvaluationQuestion,
    answerer: Callable[[str], AnswerResult],
    *,
    started: float,
) -> dict[str, Any]:
    try:
        result = answerer(question.question)
        if not isinstance(result, AnswerResult):
            raise TypeError("RAG answerer must return AnswerResult")
        answer_text = result.answer
        sources = [_source_record(source) for source in result.sources]
        citation_exists, citation_valid = _valid_citations(answer_text, sources)
        error_code = result.error.code if result.error else None
        status = result.status.value
        if result.status is AnswerStatus.INVALID_RESPONSE:
            status = "incomplete_response"
        return {
            "status": status,
            "answer": answer_text,
            "error_code": error_code,
            "retrieved_sources": sources,
            "citation_exists": citation_exists,
            "citation_valid": citation_valid,
            "evidence_gate": dict(result.gate_decision) if result.gate_decision is not None else None,
            "retrieval_query": result.retrieval_query,
            "latency_ms": (perf_counter() - started) * 1000.0,
        }
    except Exception:
        return {
            "status": "request_error",
            "answer": None,
            "error_code": "rag_request_failed",
            "retrieved_sources": [],
            "citation_exists": None,
            "citation_valid": None,
            "latency_ms": (perf_counter() - started) * 1000.0,
        }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            if index == len(lines) - 1:
                break
            raise PairedEvaluationError("paired result cache contains invalid JSON") from exc
        if isinstance(value, dict):
            records.append(value)
    return records


def _side_by_side(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["question_id"]), {})[str(record["condition"])] = record
    rows: list[dict[str, Any]] = []
    for question_id, pair in grouped.items():
        rag_row = pair.get("rag", {})
        no_rag_row = pair.get("no_rag", {})
        base = rag_row or no_rag_row
        row: dict[str, Any] = {
            "question_id": question_id,
            "question": base.get("question"),
            "split": base.get("split"),
            "category": base.get("category"),
            "expected_answer": base.get("expected_answer"),
            "model_id": base.get("model_id"),
            "configuration_json": _canonical(base.get("configuration", {})),
        }
        for name, record in (("rag", rag_row), ("no_rag", no_rag_row)):
            labels = _labels(record)
            row.update(
                {
                    f"{name}_status": record.get("status"),
                    f"{name}_answer": record.get("answer"),
                    f"{name}_error_code": record.get("error_code"),
                    f"{name}_latency_ms": record.get("latency_ms"),
                    f"{name}_citation_exists": record.get("citation_exists"),
                    f"{name}_citation_valid": record.get("citation_valid"),
                    f"{name}_evidence_gate_json": _canonical(record.get("evidence_gate")),
                    f"{name}_retrieval_query": record.get("retrieval_query"),
                    f"{name}_sources_json": _canonical(record.get("retrieved_sources", [])),
                    f"{name}_answer_correct": labels.get("answer_correct"),
                    f"{name}_unsupported_claims_present": labels.get("unsupported_claims_present"),
                    f"{name}_unsupported_claims_count": labels.get("unsupported_claims_count"),
                    f"{name}_citation_support": labels.get("citation_support"),
                    f"{name}_evidence_support": labels.get("evidence_support"),
                    f"{name}_appropriate_refusal": labels.get("appropriate_refusal"),
                    f"{name}_reviewer": labels.get("reviewer"),
                    f"{name}_reviewed_at": labels.get("reviewed_at"),
                    f"{name}_review_notes": labels.get("notes"),
                }
            )
        pair_labels = _labels(rag_row)
        tag = pair_labels.get("comparison_tag")
        row["comparison_tag"] = tag if tag in _COMPARISON_TAGS and _has_review_metadata(pair_labels) else None
        rows.append(row)
    return rows


def _reviewed_rate(records: Sequence[Mapping[str, Any]], label: str, *, eligible_statuses: set[str]) -> tuple[float | None, int]:
    values = [
        _labels(record).get(label)
        for record in records
        if record.get("status") in eligible_statuses
        and isinstance(_labels(record).get(label), bool)
        and _has_review_metadata(_labels(record))
    ]
    return (sum(values) / len(values) if values else None, len(values))


def _latency_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, float | int | None]:
    values = [float(record["latency_ms"]) for record in records if isinstance(record.get("latency_ms"), (float, int))]
    if len(values) < 5:
        return {"n": len(values), "median_ms": None, "p95_ms": None}
    ordered = sorted(values)
    return {
        "n": len(values),
        "median_ms": float(median(values)),
        "p95_ms": ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)],
    }


def _write_report(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Controlled no-RAG versus RAG evaluation",
        "",
        "Factual correctness, unsupported claims, citation support, evidence support, and refusal correctness are human-reviewed labels. Citation syntax and mapping are recorded separately and do not establish factual quality. No LLM judge is used as ground truth.",
        "",
    ]
    for condition in ("no_rag", "rag"):
        rows = [record for record in records if record.get("condition") == condition]
        answered = [record for record in rows if record.get("status") == "answered"]
        failed = sum(record.get("status") in {"generation_error", "request_error", "incomplete_response", "generation_error"} for record in rows)
        correct, correct_n = _reviewed_rate(rows, "answer_correct", eligible_statuses={"answered"})
        unsupported, unsupported_n = _reviewed_rate(rows, "unsupported_claims_present", eligible_statuses={"answered"})
        support, support_n = _reviewed_rate(rows, "citation_support", eligible_statuses={"answered"})
        refusals, refusal_n = _reviewed_rate(
            rows, "appropriate_refusal", eligible_statuses={"answered", "abstained"}
        )
        citation_valid = [record.get("citation_valid") for record in answered if record.get("citation_valid") is not None]
        latency = _latency_summary(rows)
        lines.extend(
            [
                f"## {condition.replace('_', ' ').title()}",
                "",
                f"- Requests: {len(rows)}; API/request/incomplete failures: {failed}.",
                f"- Human answer correctness: {correct if correct is not None else 'not reviewed'} (n={correct_n}).",
                f"- Human unsupported-claim rate: {unsupported if unsupported is not None else 'not reviewed'} (n={unsupported_n}).",
                f"- Human citation-support rate: {support if support is not None else 'not reviewed'} (n={support_n}).",
                f"- Human appropriate-refusal rate: {refusals if refusals is not None else 'not reviewed'} (n={refusal_n}).",
                f"- Structurally valid citations among answered outputs: {sum(citation_valid)}/{len(citation_valid)} reviewed by the deterministic citation map; this is not source support.",
                f"- Latency summary: median={latency['median_ms']} ms, p95={latency['p95_ms']} ms (n={latency['n']}; requires at least 5 outputs).",
                "",
            ]
        )
    tags = {
        "rag_helped": "RAG helped",
        "no_change": "No measured quality difference",
        "rag_misleading_evidence": "RAG retrieved misleading evidence",
        "rag_incomplete_evidence": "RAG evidence was incomplete",
        "no_rag_better": "No-RAG was better",
        "unclear": "Reviewer marked unclear",
    }
    pairs = _side_by_side(records)
    records_by_pair = {
        (str(record.get("question_id")), str(record.get("condition"))): record
        for record in records
    }
    reviewed_pairs = [
        row for row in pairs
        if isinstance(row.get("rag_answer_correct"), bool)
        and isinstance(row.get("no_rag_answer_correct"), bool)
        and _has_review_metadata(_labels(records_by_pair[(str(row["question_id"]), "rag")]))
        and _has_review_metadata(_labels(records_by_pair[(str(row["question_id"]), "no_rag")]))
    ]
    correctness_delta = (
        sum(int(row["rag_answer_correct"]) - int(row["no_rag_answer_correct"]) for row in reviewed_pairs)
        / len(reviewed_pairs)
        if reviewed_pairs
        else None
    )
    unsupported_pairs = [
        row for row in pairs
        if isinstance(row.get("rag_unsupported_claims_present"), bool)
        and isinstance(row.get("no_rag_unsupported_claims_present"), bool)
        and _has_review_metadata(_labels(records_by_pair[(str(row["question_id"]), "rag")]))
        and _has_review_metadata(_labels(records_by_pair[(str(row["question_id"]), "no_rag")]))
    ]
    unsupported_delta = (
        sum(
            int(row["rag_unsupported_claims_present"])
            - int(row["no_rag_unsupported_claims_present"])
            for row in unsupported_pairs
        ) / len(unsupported_pairs)
        if unsupported_pairs
        else None
    )
    lines.extend(
        [
            "## Paired human-reviewed differences",
            "",
            f"- Answer-correctness difference (RAG minus no-RAG): {correctness_delta if correctness_delta is not None else 'not reviewed'} (paired n={len(reviewed_pairs)}).",
            f"- Unsupported-claim presence difference (RAG minus no-RAG): {unsupported_delta if unsupported_delta is not None else 'not reviewed'} (paired n={len(unsupported_pairs)}). Negative values mean fewer RAG outputs were labeled with unsupported claims.",
            "",
        ]
    )
    for tag, title in tags.items():
        examples = [row for row in pairs if row.get("comparison_tag") == tag]
        lines.extend([f"## {title} examples", ""])
        if not examples:
            lines.extend(["No human-reviewed example has this comparison label yet.", ""])
        else:
            for row in examples[:5]:
                lines.append(f"- `{row['question_id']}` ({row['split']}): {row['question']}")
            lines.append("")
    _atomic_write(path, "\n".join(lines).rstrip() + "\n")


def run_paired_evaluation(
    questions: Sequence[EvaluationQuestion],
    *,
    rag_answerer: Callable[[str], AnswerResult],
    generator: TextGenerator,
    settings: RAGSettings,
    output_dir: str | Path,
    run_config: Mapping[str, Any],
    force: bool = False,
) -> dict[str, Any]:
    """Run both conditions over the exact same ordered questions and config."""

    if not questions:
        raise PairedEvaluationError("paired evaluation needs at least one question")
    if len({question.id for question in questions}) != len(questions):
        raise PairedEvaluationError("question IDs must be unique across the paired dataset")
    if run_config.get("gemma_model") != settings.gemma_model or run_config.get("temperature") != settings.temperature:
        raise PairedEvaluationError("run_config model and generation settings must match RAGSettings")
    if not run_config.get("corpus_revision") or not run_config.get("prompt_revision"):
        raise PairedEvaluationError("run_config must identify the source corpus and prompt revision")

    dataset_hash = _fingerprint([asdict(question) for question in questions])
    config = dict(_without_secrets(run_config))
    config["dataset_fingerprint"] = dataset_hash
    config["gemma_model"] = settings.gemma_model
    config["temperature"] = settings.temperature
    config_hash = _fingerprint(config)
    root = Path(output_dir)
    raw_path = root / "paired_predictions.jsonl"
    cached = {} if force else {
        (str(row.get("question_id")), str(row.get("condition"))): row
        for row in _read_jsonl(raw_path)
        if row.get("cache_key") and row.get("status") not in {"generation_error", "request_error"}
    }
    record_map: dict[tuple[str, str], dict[str, Any]] = {}
    for key, row in cached.items():
        if row.get("config_fingerprint") == config_hash:
            record_map[key] = row

    def base_record(question: EvaluationQuestion, condition: str) -> dict[str, Any]:
        key = _fingerprint(
            {
                "question_id": question.id,
                "question": question.question,
                "expected_answer": question.expected_answer,
                "condition": condition,
                "config_fingerprint": config_hash,
            }
        )
        return {
            "cache_key": key,
            "config_fingerprint": config_hash,
            "condition": condition,
            "model_id": settings.gemma_model,
            "question_id": question.id,
            "question": question.question,
            "expected_answer": question.expected_answer,
            "answerable": question.answerable,
            "category": question.category,
            "split": question.split,
            "configuration": config,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "human_labels": dict(_LABEL_TEMPLATE),
        }

    for question in questions:
        for condition in ("no_rag", "rag"):
            pair_key = (question.id, condition)
            if pair_key in record_map:
                continue
            record = base_record(question, condition)
            started = perf_counter()
            if condition == "no_rag":
                status, answer_text, error_code = _call_no_rag(generator, question, settings)
                measured = (perf_counter() - started) * 1000.0
                sources: list[dict[str, Any]] = []
                citation_exists, citation_valid = _valid_citations(answer_text, sources)
                payload = {
                    "status": status,
                    "answer": answer_text,
                    "error_code": error_code,
                    "retrieved_sources": sources,
                    "citation_exists": citation_exists,
                    "citation_valid": citation_valid,
                    "latency_ms": measured,
                }
            else:
                payload = _rag_record(question, rag_answerer, started=started)
            record.update(payload)
            record_map[pair_key] = record
            _write_jsonl(raw_path, list(record_map.values()))

    records = list(record_map.values())
    rows = _side_by_side(records)
    side_by_side_path = root / "side_by_side_results.csv"
    fields = list(rows[0]) if rows else []
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    _atomic_write(side_by_side_path, stream.getvalue())
    report_path = root / "paired_report.md"
    _write_report(report_path, records)
    return {
        "predictions_path": raw_path,
        "side_by_side_csv_path": side_by_side_path,
        "report_path": report_path,
        "records": records,
        "question_count": len(questions),
        "cached_or_completed_calls": len(records),
    }


def run_smoke_test() -> dict[str, Any]:
    """Exercise both paths with synthetic text and a mocked generator only."""

    from tempfile import TemporaryDirectory

    from campusreg.evaluation.models import EvaluationQuestion
    from campusreg.rag.models import ErrorInfo

    question = EvaluationQuestion(
        id="SYNTHETIC-PAIRED-SMOKE-1",
        question="SYNTHETIC TEST FIXTURE: what does the sample passage say?",
        category="direct_factual",
        expected_answer="SYNTHETIC TEST FIXTURE: expected phrase",
        answerable=True,
        expected_document_ids=("synthetic-source",),
        expected_pages=(),
        expected_conflict=False,
        split="development",
    )
    source = RetrievedSource(
        page=1,
        text="SYNTHETIC TEST FIXTURE passage available only to the RAG path.",
        source_name="SYNTHETIC TEST FIXTURE source.pdf",
        score=0.91,
        document_id="synthetic-source",
        chunk_id="synthetic-chunk",
    )

    class FakeGenerator:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def generate(self, prompt: str, **kwargs: Any) -> str:
            self.prompts.append(prompt)
            assert "SYNTHETIC TEST FIXTURE passage" not in prompt
            assert kwargs["model"] == "synthetic-model"
            return "SYNTHETIC TEST FIXTURE general response"

    fake = FakeGenerator()

    def fake_rag(_question: str) -> AnswerResult:
        return AnswerResult(
            status=AnswerStatus.ANSWERED,
            answer=f"SYNTHETIC TEST FIXTURE answer {source.citation}",
            sources=(source,),
        )

    with TemporaryDirectory(prefix="campusreg-paired-smoke-") as temporary:
        result = run_paired_evaluation(
            [question],
            rag_answerer=fake_rag,
            generator=fake,
            settings=RAGSettings(gemma_model="synthetic-model"),
            output_dir=temporary,
            run_config={
                "gemma_model": "synthetic-model",
                "temperature": 0.2,
                "corpus_revision": "synthetic-fixture-v1",
                "prompt_revision": PROMPT_REVISION,
            },
        )
        if len(fake.prompts) != 1:
            raise AssertionError("the no-RAG condition must make one question-only generator call")
        if not result["side_by_side_csv_path"].is_file() or not result["report_path"].is_file():
            raise AssertionError("paired smoke outputs were not written")
        return {"status": "passed", "calls": len(fake.prompts), "conditions": 2}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Controlled no-RAG versus RAG Gemma evaluation.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="run synthetic mocked checks; no API calls")
    mode.add_argument("--full", action="store_true", help="explicitly run against reviewed data and Gemma")
    parser.add_argument("--dataset")
    parser.add_argument("--corpus-manifest")
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    if args.smoke:
        result = run_smoke_test()
        print(f"Paired evaluation smoke test {result['status']} ({result['conditions']} conditions).")
        return 0
    if not args.dataset or not args.corpus_manifest or not args.output_dir:
        parser.error("--full requires --dataset, --corpus-manifest, and --output-dir")

    validate_challenge_dataset(args.dataset)
    questions = load_dataset(args.dataset)
    pages, sources = load_pdf_manifest(args.corpus_manifest)
    corpus_revision = _fingerprint(sources)
    settings = replace(RAGSettings.from_env(), corpus_revision=corpus_revision)
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    knowledge_base = build_knowledge_base(pages, "paired evaluation corpus", embedder, settings)
    generator = GoogleGenAIProvider.from_env()
    answerer = lambda question: rag_answer(question, knowledge_base, embedder, generator, settings)
    run_paired_evaluation(
        questions,
        rag_answerer=answerer,
        generator=generator,
        settings=settings,
        output_dir=args.output_dir,
        run_config={
            "gemma_model": settings.gemma_model,
            "temperature": settings.temperature,
            "embedding_model": settings.embedding_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "top_k": settings.top_k,
            "retrieval_method": settings.retrieval_method,
            "candidate_count": settings.candidate_count,
            "rrf_constant": settings.rrf_constant,
            "corpus_revision": corpus_revision,
            "source_documents": sources,
            "prompt_revision": PROMPT_REVISION,
        },
    )
    print(f"Paired results saved under {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
