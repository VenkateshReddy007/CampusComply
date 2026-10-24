"""Fixed-passage, resumable Gemma model comparison.

The runner uses identical question, retrieved passage text, citation rules, and
generation parameters for both model IDs. Model outputs are never replaced by
another model when a request fails.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

from campusreg.config import RAGSettings
from campusreg.evaluation.models import EvaluationQuestion
from campusreg.rag.errors import GenerationError
from campusreg.rag.generation import GoogleGenAIProvider, TextGenerator
from campusreg.rag.models import RetrievedSource

MODEL_IDS = ("gemma-4-26b-a4b-it", "gemma-4-31b-it")
SYSTEM_INSTRUCTION = """Answer the question using ONLY the fixed evidence passages.
Every factual claim must cite the exact supplied document and page as
[document name, document ID, p. N]. Do not invent citations. If the evidence
does not establish an answer, say: "The answer could not be found in the document."""
PROMPT_REVISION = "fixed-evidence-gemma-comparison-v1"
_CITATIONS = re.compile(r"\[[^\]\n]+\]")
_SECRET = re.compile(r"api[_-]?key|secret|token|password|credential|authorization", re.I)


class ModelComparisonError(ValueError):
    pass


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _without_secrets(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _without_secrets(v) for k, v in value.items() if not _SECRET.search(str(k))}
    if isinstance(value, (tuple, list)):
        return [_without_secrets(item) for item in value]
    return value


def _source(source: RetrievedSource) -> dict[str, Any]:
    return {
        "document_id": source.document_id,
        "document_name": source.source_name,
        "page_number": source.page_number,
        "chunk_id": source.chunk_id,
        "citation": source.citation,
        "text": source.text,
    }


def _prompt(question: str, sources: Sequence[Mapping[str, Any]]) -> str:
    evidence = "\n\n".join(f"{row['citation']}\n{row['text']}" for row in sources)
    return f"Fixed evidence passages:\n{evidence}\n\nQuestion: {question}"


def _call(
    question: EvaluationQuestion,
    model_id: str,
    source_rows: Sequence[Mapping[str, Any]],
    provider: TextGenerator,
    *,
    temperature: float,
) -> dict[str, Any]:
    started = perf_counter()
    timestamp = datetime.now(timezone.utc).isoformat()
    try:
        output = provider.generate(
            _prompt(question.question, source_rows),
            model=model_id,
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=temperature,
        )
        if not isinstance(output, str) or not output.strip():
            raise ValueError("provider returned an incomplete response")
        answer = output.strip()
        citation_tokens = _CITATIONS.findall(answer)
        allowed = {row["citation"] for row in source_rows}
        status = "answered"
        error_code = None
        citation_exists: bool | None = bool(citation_tokens)
        citation_valid: bool | None = bool(citation_tokens) and all(c in allowed for c in citation_tokens)
    except GenerationError as exc:
        answer = None
        status = "incomplete_response" if exc.code == "empty_generation" else "api_error"
        error_code = exc.code
        citation_exists = None
        citation_valid = None
    except Exception as exc:
        answer = None
        status = "incomplete_response" if isinstance(exc, ValueError) else "api_error"
        error_code = "empty_or_invalid_response" if isinstance(exc, ValueError) else "generation_failed"
        citation_exists = None
        citation_valid = None
    return {
        "model_id": model_id,
        "question_id": question.id,
        "question": question.question,
        "split": question.split,
        "expected_answer": question.expected_answer,
        "answerable": question.answerable,
        "timestamp": timestamp,
        "status": status,
        "answer": answer,
        "error_code": error_code,
        "latency_ms": (perf_counter() - started) * 1000,
        "citation_exists": citation_exists,
        "citation_valid": citation_valid,
        "fixed_evidence": list(source_rows),
        "human_labels": {
            "answer_correct": None,
            "unsupported_claims_present": None,
            "unsupported_claims_count": None,
            "citation_support": None,
            "evidence_support": None,
            "appropriate_refusal": None,
            "reviewer": None,
            "reviewed_at": None,
            "notes": None,
        },
    }


def _reviewed_bool(row: Mapping[str, Any], field: str) -> bool | None:
    labels = row.get("human_labels")
    value = labels.get(field) if isinstance(labels, Mapping) else None
    if not isinstance(value, bool):
        return None
    reviewer = labels.get("reviewer")
    reviewed_at = labels.get("reviewed_at")
    if not isinstance(reviewer, str) or not reviewer.strip() or not isinstance(reviewed_at, str):
        return None
    try:
        if datetime.fromisoformat(reviewed_at.replace("Z", "+00:00")).tzinfo is None:
            return None
    except ValueError:
        return None
    return value


def _percentiles(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = sorted(float(row["latency_ms"]) for row in rows if isinstance(row.get("latency_ms"), (int, float)))
    if len(values) < 5:
        return {"n": len(values), "median_ms": None, "p95_ms": None}
    return {
        "n": len(values),
        "median_ms": float(median(values)),
        "p95_ms": values[max(0, math.ceil(0.95 * len(values)) - 1)],
    }


def _write_report(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Gemma fixed-evidence model comparison",
        "",
        "Both models receive the same fixed passages, exact question, system instructions, and temperature. Correctness, unsupported claims, citation support, evidence support, and refusal quality require human review. Citation syntax is only a structural check.",
        "",
    ]
    for model_id in MODEL_IDS:
        model_rows = [row for row in rows if row.get("model_id") == model_id]
        completed = [row for row in model_rows if row.get("status") == "answered"]
        reviewed_correct = [v for row in model_rows if (v := _reviewed_bool(row, "answer_correct")) is not None]
        reviewed_unsupported = [v for row in model_rows if (v := _reviewed_bool(row, "unsupported_claims_present")) is not None]
        reviewed_support = [v for row in model_rows if (v := _reviewed_bool(row, "citation_support")) is not None]
        latency = _percentiles(model_rows)
        lines.extend([
            f"## `{model_id}`", "",
            f"- Completed: {len(completed)}/{len(model_rows)}; API errors or incomplete responses: {len(model_rows) - len(completed)}.",
            f"- Human answer correctness: {sum(reviewed_correct) / len(reviewed_correct) if reviewed_correct else 'not reviewed'} (n={len(reviewed_correct)}).",
            f"- Human unsupported-claim rate: {sum(reviewed_unsupported) / len(reviewed_unsupported) if reviewed_unsupported else 'not reviewed'} (n={len(reviewed_unsupported)}).",
            f"- Human citation-support rate: {sum(reviewed_support) / len(reviewed_support) if reviewed_support else 'not reviewed'} (n={len(reviewed_support)}).",
            f"- Structural citation validity: {sum(bool(row.get('citation_valid')) for row in completed)}/{sum(row.get('citation_valid') is not None for row in completed)}; not source support.",
            f"- Latency: median={latency['median_ms']} ms, p95={latency['p95_ms']} ms (n={latency['n']}; minimum 5).", "",
        ])
    paired = []
    lookup = {(row["question_id"], row["model_id"]): row for row in rows}
    for qid in sorted({str(row["question_id"]) for row in rows}):
        left = lookup.get((qid, MODEL_IDS[0]))
        right = lookup.get((qid, MODEL_IDS[1]))
        if left and right and left.get("fixed_evidence") == right.get("fixed_evidence"):
            a, b = _reviewed_bool(left, "answer_correct"), _reviewed_bool(right, "answer_correct")
            if a is not None and b is not None:
                paired.append((a, b))
    complete = all((qid, model) in lookup and lookup[(qid, model)].get("status") == "answered"
                   for qid in {str(row["question_id"]) for row in rows} for model in MODEL_IDS)
    lines.extend([
        "## Comparison status", "",
        f"- Pair-complete: {complete}. A missing or failed model call makes the run incomplete; no substitution is made.",
        f"- Paired human correctness comparison: {sum(int(b) - int(a) for a, b in paired) / len(paired) if paired else 'not reviewed'} (model 31B minus 26B A4B; n={len(paired)}).",
        "- No quality conclusion is available until the paired outputs have human review. This report does not treat plausible wording or citation formatting as correctness.", "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_fixed_evidence_comparison(
    questions: Sequence[EvaluationQuestion],
    evidence: Mapping[str, Sequence[RetrievedSource]],
    *,
    provider: TextGenerator,
    settings: RAGSettings,
    output_dir: str | Path,
    run_config: Mapping[str, Any],
    model_ids: Sequence[str] = MODEL_IDS,
    force: bool = False,
) -> dict[str, Any]:
    """Compare models over precisely the same questions and retrieved passages."""
    if tuple(model_ids) != MODEL_IDS:
        raise ModelComparisonError(f"model_ids must be exactly {MODEL_IDS!r} in the baseline comparison")
    if not questions or len({q.id for q in questions}) != len(questions):
        raise ModelComparisonError("comparison needs a non-empty set of unique question IDs")
    if set(evidence) != {q.id for q in questions}:
        raise ModelComparisonError("fixed evidence must be provided for every and only selected question")
    config = _without_secrets(dict(run_config))
    config.update({"temperature": settings.temperature, "prompt_revision": PROMPT_REVISION, "model_ids": list(MODEL_IDS)})
    config_hash = _hash({"config": config, "questions": [asdict(q) for q in questions],
                         "evidence": {q.id: [_source(s) for s in evidence[q.id]] for q in questions}})
    root = Path(output_dir)
    raw_path = root / "model_comparison_predictions.jsonl"
    existing: dict[tuple[str, str], dict[str, Any]] = {}
    if raw_path.exists() and not force:
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("config_fingerprint") == config_hash and row.get("cache_key"):
                existing[(str(row.get("question_id")), str(row.get("model_id")))] = row
    output = []
    for question in questions:
        fixed_rows = [_source(source) for source in evidence[question.id]]
        fixed_hash = _hash(fixed_rows)
        for model_id in MODEL_IDS:
            key = (question.id, model_id)
            prior = existing.get(key)
            cache_key = _hash({"config": config_hash, "question": question.id,
                               "model_id": model_id, "evidence": fixed_hash})
            if prior is not None and prior.get("cache_key") == cache_key:
                output.append(prior)
                continue
            row = _call(question, model_id, fixed_rows, provider, temperature=settings.temperature)
            row.update({"cache_key": cache_key, "config_fingerprint": config_hash,
                        "configuration": config, "fixed_evidence_fingerprint": fixed_hash})
            output.append(row)
            root.mkdir(parents=True, exist_ok=True)
            raw_path.write_text("".join(_canonical(item) + "\n" for item in output), encoding="utf-8")
    root.mkdir(parents=True, exist_ok=True)
    raw_path.write_text("".join(_canonical(row) + "\n" for row in output), encoding="utf-8")
    csv_path = root / "model_comparison_side_by_side.csv"
    fields = ["question_id", "question", "split", "expected_answer", "fixed_evidence_fingerprint"]
    for model_id in MODEL_IDS:
        fields.extend([f"{model_id}_status", f"{model_id}_answer", f"{model_id}_error_code",
                       f"{model_id}_timestamp", f"{model_id}_latency_ms", f"{model_id}_citation_valid",
                       f"{model_id}_answer_correct", f"{model_id}_unsupported_claims_present",
                       f"{model_id}_citation_support", f"{model_id}_evidence_support",
                       f"{model_id}_appropriate_refusal"])
    lookup = {(row["question_id"], row["model_id"]): row for row in output}
    with csv_path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for question in questions:
            left = lookup[(question.id, MODEL_IDS[0])]
            row = {"question_id": question.id, "question": question.question, "split": question.split,
                   "expected_answer": question.expected_answer,
                   "fixed_evidence_fingerprint": left.get("fixed_evidence_fingerprint")}
            for model_id in MODEL_IDS:
                record = lookup[(question.id, model_id)]
                labels = record.get("human_labels", {})
                for field in fields:
                    prefix = model_id + "_"
                    if field.startswith(prefix):
                        key = field[len(prefix):]
                        row[field] = labels.get(key) if key in labels else record.get(key)
            writer.writerow(row)
    _write_report(root / "model_comparison_report.md", output)
    complete = all(row.get("status") == "answered" for row in output)
    return {"complete": complete, "config_fingerprint": config_hash,
            "prediction_count": len(output), "predictions_path": raw_path,
            "csv_path": csv_path, "report_path": root / "model_comparison_report.md"}


class _SmokeProvider:
    def generate(self, prompt: str, *, model: str, system_instruction: str, temperature: float) -> str:
        if "SYNTHETIC TEST FIXTURE" not in prompt:
            raise AssertionError("smoke input must be synthetic")
        return "SYNTHETIC TEST FIXTURE answer [synthetic.pdf, document synthetic-id, p. 1]"


def _smoke(output_dir: str | Path) -> None:
    from campusreg.evaluation.models import EvaluationQuestion

    questions = [EvaluationQuestion(
        id=f"synthetic-{i}", question=f"SYNTHETIC TEST FIXTURE question {i}",
        category="direct_factual", expected_answer="SYNTHETIC TEST FIXTURE expected answer",
        answerable=True, expected_document_ids=("synthetic-id",),
        expected_pages=(), expected_conflict=False, split="development",
    ) for i in (1, 2)]
    source = RetrievedSource(page=1, text="SYNTHETIC TEST FIXTURE passage", source_name="synthetic.pdf",
                            score=0.5, document_id="synthetic-id", chunk_id="synthetic-chunk")
    result = run_fixed_evidence_comparison(
        questions, {question.id: [source] for question in questions}, provider=_SmokeProvider(),
        settings=RAGSettings.baseline(), output_dir=output_dir,
        run_config={"corpus_revision": "synthetic-corpus", "retrieval_revision": "synthetic-retrieval"},
        force=True,
    )
    if not result["complete"] or result["prediction_count"] != 4:
        raise RuntimeError("model comparison smoke check failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--corpus-manifest", type=Path)
    parser.add_argument("--split", choices=("development", "validation", "test"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results/model-comparison"))
    parser.add_argument("--force", action="store_true", help="ignore cached completed calls and rerun them")
    args = parser.parse_args()
    if args.smoke:
        _smoke(args.output_dir)
        print("fixed-evidence model-comparison smoke passed (4 mocked calls)")
        return
    if args.dataset is None or args.corpus_manifest is None or args.split is None:
        parser.error("--full requires --dataset, --corpus-manifest, and --split")
    from campusreg.evaluation.dataset import load_dataset, validate_challenge_dataset
    from campusreg.evaluation.retrieval_experiments import load_pdf_manifest
    from campusreg.rag.embeddings import SentenceTransformerEmbedder
    from campusreg.rag.retrieval import build_knowledge_base, retrieve

    settings = RAGSettings.from_env()
    provider = GoogleGenAIProvider.from_env()
    validate_challenge_dataset(args.dataset)
    questions = load_dataset(args.dataset, split=args.split)
    pages, revision_documents = load_pdf_manifest(args.corpus_manifest)
    actual_documents = {page.document_id for page in pages}
    actual_pages = {(page.document_id, page.page_number) for page in pages}
    expected_documents = {doc_id for question in questions for doc_id in question.expected_document_ids}
    if expected_documents - actual_documents:
        raise ModelComparisonError("held-out questions reference documents missing from the verified corpus")
    expected_pages = {
        (page.document_id, page.page_number)
        for question in questions for page in question.expected_pages
    }
    if expected_pages - actual_pages:
        raise ModelComparisonError("held-out questions reference physical PDF pages missing from the verified corpus")
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    knowledge_base = build_knowledge_base(pages, "fixed model-comparison corpus", embedder, settings)
    fixed_evidence = {question.id: retrieve(question.question, knowledge_base, embedder, settings)
                      for question in questions}
    result = run_fixed_evidence_comparison(
        questions, fixed_evidence, provider=provider, settings=settings, output_dir=args.output_dir,
        run_config={"corpus_revision": _hash(revision_documents), "embedding_model": settings.embedding_model,
                    "chunk_size": settings.chunk_size, "chunk_overlap": settings.chunk_overlap,
                    "top_k": settings.top_k, "temperature": settings.temperature,
                    "retrieval_method": settings.retrieval_method,
                    "candidate_count": settings.candidate_count,
                    "rrf_constant": settings.rrf_constant},
        force=args.force,
    )
    print(f"comparison complete={result['complete']} predictions={result['prediction_count']}")


if __name__ == "__main__":
    main()
