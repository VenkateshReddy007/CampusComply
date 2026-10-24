"""Validation-only evidence-threshold calibration and held-out reporting."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from campusreg.rag.evidence_gate import EvidenceGatePolicy, configuration_id


class GateCalibrationError(ValueError):
    """Raised when validation evidence is insufficient for threshold selection."""


def _fingerprint(value: Any) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as target:
            target.write(text)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _retrieval_config(record: Mapping[str, Any]) -> dict[str, Any]:
    config = record.get("configuration")
    if not isinstance(config, Mapping):
        raise GateCalibrationError("RAG records must preserve their retrieval configuration")
    required = ("embedding_model", "chunk_size", "chunk_overlap", "top_k")
    if any(name not in config for name in required):
        raise GateCalibrationError("retrieval configuration is missing embedding or chunk settings")
    method = config.get("retrieval_method", "dense")
    result = {
        "embedding_model": str(config["embedding_model"]),
        "chunk_size": int(config["chunk_size"]),
        "chunk_overlap": int(config["chunk_overlap"]),
        "top_k": int(config["top_k"]),
        "retrieval_method": str(method),
        "corpus_revision": str(config.get("corpus_revision", "")),
    }
    if not result["corpus_revision"]:
        raise GateCalibrationError("retrieval configuration must identify the source corpus revision")
    if method == "hybrid":
        result.update(
            {
                "candidate_count": int(config.get("candidate_count", 20)),
                "rrf_constant": int(config.get("rrf_constant", 60)),
            }
        )
    return result


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


def _require_review(row: Mapping[str, Any], names: Sequence[str]) -> Mapping[str, Any]:
    labels = row.get("human_labels")
    if not isinstance(labels, Mapping) or not _has_review_metadata(labels):
        raise GateCalibrationError(
            f"validation question {row.get('question_id')!r} lacks human-review metadata"
        )
    if any(not isinstance(labels.get(name), bool) for name in names):
        raise GateCalibrationError(
            f"validation question {row.get('question_id')!r} lacks required reviewed labels: {', '.join(names)}"
        )
    return labels


def _retrieval_score(row: Mapping[str, Any]) -> float | None:
    sources = row.get("retrieved_sources") or []
    scores = [float(source["score"]) for source in sources if isinstance(source, Mapping) and "score" in source]
    if any(not math.isfinite(value) for value in scores):
        raise GateCalibrationError("retrieval scores must be finite")
    return max(scores) if scores else None


def _candidate_metrics(
    rows: Sequence[Mapping[str, Any]], threshold: float, *, max_false_accept_rate: float,
    max_unsupported_answer_rate: float,
) -> dict[str, Any]:
    accepted = [row for row in rows if _retrieval_score(row) is not None and _retrieval_score(row) >= threshold]
    answerable = [row for row in rows if row["answerable"] is True]
    unanswerable = [row for row in rows if row["answerable"] is False]
    accepted_answerable = [row for row in accepted if row["answerable"] is True]
    accepted_unanswerable = [row for row in accepted if row["answerable"] is False]
    answered_accepted = [row for row in accepted if row.get("status") == "answered"]
    false_accept_rate = len(accepted_unanswerable) / len(unanswerable)
    false_refusal_rate = (len(answerable) - len(accepted_answerable)) / len(answerable)
    answer_labels = [
        bool(row["human_labels"]["answer_correct"])
        for row in accepted_answerable
        if row.get("status") == "answered"
    ]
    unsupported_labels = [bool(row["human_labels"]["unsupported_claims_present"]) for row in answered_accepted]
    unsupported_rate = mean(unsupported_labels) if unsupported_labels else 0.0
    answer_correctness = mean(answer_labels) if answer_labels else None
    coverage = len(accepted) / len(rows)
    feasible = (
        false_accept_rate <= max_false_accept_rate
        and unsupported_rate <= max_unsupported_answer_rate
    )
    return {
        "threshold": threshold,
        "accepted_count": len(accepted),
        "answerable_count": len(answerable),
        "unanswerable_count": len(unanswerable),
        "false_accept_count": len(accepted_unanswerable),
        "false_accept_rate": false_accept_rate,
        "false_refusal_count": len(answerable) - len(accepted_answerable),
        "false_refusal_rate": false_refusal_rate,
        "coverage": coverage,
        "answer_correctness_on_accepted_answerable": answer_correctness,
        "answer_correctness_n": len(answer_labels),
        "unsupported_answer_rate": unsupported_rate,
        "unsupported_answer_n": len(unsupported_labels),
        "feasible_under_documented_limits": feasible,
    }


def _select_candidate(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    feasible = [row for row in candidates if row["feasible_under_documented_limits"]]
    if not feasible:
        return None
    return sorted(
        feasible,
        key=lambda row: (
            -float(row["coverage"]),
            -float(row["answer_correctness_on_accepted_answerable"] or 0.0),
            float(row["false_refusal_rate"]),
            float(row["threshold"]),
        ),
    )[0]


def calibrate_evidence_gates(
    records: Sequence[Mapping[str, Any]],
    *,
    output_dir: str | Path,
    max_false_accept_rate: float,
    max_unsupported_answer_rate: float,
) -> dict[str, Any]:
    """Select per-config thresholds from reviewed validation records only.

    The operating rule first enforces explicit false-accept and unsupported-answer
    rate limits, then maximizes validation coverage and answer correctness.
    """

    for name, value in (
        ("max_false_accept_rate", max_false_accept_rate),
        ("max_unsupported_answer_rate", max_unsupported_answer_rate),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between zero and one")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    configs: dict[str, dict[str, Any]] = {}
    data_fingerprints: dict[str, str] = {}
    for row in records:
        if row.get("condition") != "rag":
            continue
        if row.get("split") != "validation":
            continue
        if not isinstance(row.get("answerable"), bool):
            raise GateCalibrationError("validation RAG records need a reviewed answerability label")
        config = _retrieval_config(row)
        config_id = configuration_id(config)
        grouped[config_id].append(row)
        configs[config_id] = config
        config_fingerprint = str(row.get("configuration", {}).get("dataset_fingerprint", ""))
        if config_fingerprint:
            data_fingerprints[config_id] = config_fingerprint
    if not grouped:
        raise GateCalibrationError("no validation RAG records were supplied")
    question_sets: list[set[str]] = []
    output_rows: list[dict[str, Any]] = []
    selected_policies: list[dict[str, Any]] = []
    configurations: dict[str, Any] = {}
    for config_id, rows in grouped.items():
        question_ids = [str(row.get("question_id")) for row in rows]
        if len(set(question_ids)) != len(question_ids):
            raise GateCalibrationError("validation question IDs must be unique per retrieval config")
        question_sets.append(set(question_ids))
        if not any(row["answerable"] for row in rows) or not any(not row["answerable"] for row in rows):
            raise GateCalibrationError("each calibration config needs both answerable and unanswerable validation questions")
        for row in rows:
            if row.get("status") not in {"answered", "abstained"}:
                raise GateCalibrationError("failed or incomplete validation calls cannot calibrate an evidence gate")
            if row["answerable"]:
                _require_review(row, ("answer_correct", "unsupported_claims_present"))
            else:
                _require_review(row, ("appropriate_refusal",))
            if row.get("status") == "answered":
                _require_review(row, ("unsupported_claims_present",))
        scores = [_retrieval_score(row) for row in rows]
        finite_scores = sorted({score for score in scores if score is not None})
        if not finite_scores:
            raise GateCalibrationError("no validation retrieval scores are available for threshold calibration")
        thresholds = finite_scores + [math.nextafter(finite_scores[-1], math.inf)]
        candidate_rows = [
            _candidate_metrics(
                rows,
                threshold,
                max_false_accept_rate=max_false_accept_rate,
                max_unsupported_answer_rate=max_unsupported_answer_rate,
            )
            for threshold in thresholds
        ]
        selected = _select_candidate(candidate_rows)
        config = configs[config_id]
        config_fp = data_fingerprints.get(config_id) or _fingerprint(question_ids)
        score_kind = "cosine_similarity" if config["retrieval_method"] == "dense" else "reciprocal_rank_fusion_score"
        policy = None
        if selected is not None:
            policy = EvidenceGatePolicy(
                configuration_id=config_id,
                embedding_model=config["embedding_model"],
                retrieval_method=config["retrieval_method"],
                score_kind=score_kind,
                threshold=float(selected["threshold"]),
                validation_dataset_fingerprint=config_fp,
                max_false_accept_rate=max_false_accept_rate,
                max_unsupported_answer_rate=max_unsupported_answer_rate,
                validation_metrics=dict(selected),
            )
            selected_policies.append(policy.to_dict())
        configurations[config_id] = {
            "retrieval_configuration": config,
            "status": "selected" if policy is not None else "not_calibrated",
            "policy": policy.to_dict() if policy else None,
            "selection_rule": {
                "constraints": {
                    "false_accept_rate_at_most": max_false_accept_rate,
                    "unsupported_answer_rate_at_most": max_unsupported_answer_rate,
                },
                "within_constraints": [
                    "maximize validation coverage",
                    "maximize human-reviewed answer correctness",
                    "minimize false-refusal rate",
                    "choose lower threshold on exact ties",
                ],
            },
            "candidate_metrics": candidate_rows,
            "score_kind": score_kind,
            "validation_question_count": len(rows),
            "validation_dataset_fingerprint": config_fp,
        }
        for candidate in candidate_rows:
            output_rows.append({"configuration_id": config_id, **candidate})
    if any(candidate != question_sets[0] for candidate in question_sets[1:]):
        raise GateCalibrationError("all retrieval configs must use the same validation question IDs")
    value = {
        "schema_version": 1,
        "calibration_split": "validation",
        "held_out_test_used_for_calibration": False,
        "score_is_probability": False,
        "policies": selected_policies,
        "configurations": configurations,
    }
    root = Path(output_dir)
    _atomic_text(root / "evidence_gate_calibration.json", json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    columns = [
        "configuration_id", "threshold", "accepted_count", "answerable_count", "unanswerable_count",
        "false_accept_count", "false_accept_rate", "false_refusal_count", "false_refusal_rate", "coverage",
        "answer_correctness_on_accepted_answerable", "answer_correctness_n", "unsupported_answer_rate",
        "unsupported_answer_n", "feasible_under_documented_limits",
    ]
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(output_rows)
    _atomic_text(root / "validation_gate_metrics.csv", stream.getvalue())
    report = [
        "# Evidence gate calibration",
        "",
        "Thresholds were selected from validation rows only. A retrieval score is not a calibrated probability of factual correctness. The configured limits for false acceptance and unsupported answers are explicit trade-offs; they are not claims about actual risk outside this reviewed sample.",
        "",
    ]
    for config_id, detail in configurations.items():
        report.extend([f"## {config_id}", ""])
        if detail["policy"] is None:
            report.extend(["No threshold satisfied the documented risk limits; this configuration has no deployable calibrated gate.", ""])
        else:
            metrics = detail["policy"]["validation_metrics"]
            report.extend(
                [
                    f"Selected threshold: `{detail['policy']['threshold']}` using {detail['score_kind']}; this is a score cutoff, not a confidence probability.",
                    f"False accepts: {metrics['false_accept_rate']:.4f}; false refusals: {metrics['false_refusal_rate']:.4f}; coverage: {metrics['coverage']:.4f}.",
                    f"Human answer correctness among accepted answerable outputs: {metrics['answer_correctness_on_accepted_answerable']} (n={metrics['answer_correctness_n']}); unsupported-answer rate: {metrics['unsupported_answer_rate']:.4f} (n={metrics['unsupported_answer_n']}).",
                    "",
                ]
            )
    report.extend(["Held-out test metrics must be calculated after this calibration is frozen. Test observations cannot change these thresholds.", ""])
    _atomic_text(root / "evidence_gate_calibration.md", "\n".join(report))
    return {"calibration_path": root / "evidence_gate_calibration.json", "metrics_path": root / "validation_gate_metrics.csv", "report_path": root / "evidence_gate_calibration.md", **value}


def evaluate_gate_on_test(
    policies: Sequence[EvidenceGatePolicy],
    records: Sequence[Mapping[str, Any]],
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Score frozen policies against held-out test records without retuning."""

    by_config: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        if row.get("condition") == "rag" and row.get("split") == "test":
            config_id = configuration_id(_retrieval_config(row))
            by_config[config_id].append(row)
    rows: list[dict[str, Any]] = []
    for policy in policies:
        group = by_config.get(policy.configuration_id, [])
        if not group:
            continue
        answerable = [row for row in group if row.get("answerable") is True]
        unanswerable = [row for row in group if row.get("answerable") is False]
        if not answerable or not unanswerable:
            raise GateCalibrationError("held-out test evaluation needs both answerable and unanswerable questions")
        accepted = [row for row in group if (score := _retrieval_score(row)) is not None and score >= policy.threshold]
        accepted_ids = {str(row["question_id"]) for row in accepted}
        accepted_answerable = [row for row in answerable if str(row["question_id"]) in accepted_ids]
        accepted_unanswerable = [row for row in unanswerable if str(row["question_id"]) in accepted_ids]
        correct_labels: list[bool] = []
        unsupported_labels: list[bool] = []
        for row in accepted_answerable:
            labels = row.get("human_labels", {})
            if isinstance(labels, Mapping) and isinstance(labels.get("answer_correct"), bool) and _has_review_metadata(labels):
                correct_labels.append(bool(labels["answer_correct"]))
        for row in accepted:
            labels = row.get("human_labels", {})
            if row.get("status") == "answered" and isinstance(labels, Mapping) and isinstance(labels.get("unsupported_claims_present"), bool) and _has_review_metadata(labels):
                unsupported_labels.append(bool(labels["unsupported_claims_present"]))
        rows.append(
            {
                "configuration_id": policy.configuration_id,
                "embedding_model": policy.embedding_model,
                "retrieval_method": policy.retrieval_method,
                "threshold_frozen_from_validation": policy.threshold,
                "test_question_count": len(group),
                "false_accept_rate": len(accepted_unanswerable) / len(unanswerable),
                "false_refusal_rate": (len(answerable) - len(accepted_answerable)) / len(answerable),
                "coverage": len(accepted) / len(group),
                "answer_correctness_human": mean(correct_labels) if correct_labels else None,
                "answer_correctness_n": len(correct_labels),
                "unsupported_answer_rate_human": mean(unsupported_labels) if unsupported_labels else None,
                "unsupported_answer_n": len(unsupported_labels),
            }
        )
    root = Path(output_dir)
    from io import StringIO

    fields = list(rows[0]) if rows else [
        "configuration_id", "embedding_model", "retrieval_method", "threshold_frozen_from_validation",
        "test_question_count", "false_accept_rate", "false_refusal_rate", "coverage",
        "answer_correctness_human", "answer_correctness_n", "unsupported_answer_rate_human", "unsupported_answer_n",
    ]
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(root / "heldout_gate_metrics.csv", stream.getvalue())
    report = [
        "# Held-out evidence gate evaluation",
        "",
        "Thresholds were frozen from validation and were not changed using test results. Human-labeled metrics appear only where reviewed labels are available.",
        "",
    ]
    for row in rows:
        report.extend(
            [
                f"## {row['configuration_id']}",
                "",
                f"Frozen validation threshold: `{row['threshold_frozen_from_validation']}`.",
                f"Test false-accept rate: {row['false_accept_rate']:.4f}; false-refusal rate: {row['false_refusal_rate']:.4f}; coverage: {row['coverage']:.4f}.",
                f"Human answer correctness: {row['answer_correctness_human']} (n={row['answer_correctness_n']}); unsupported-answer rate: {row['unsupported_answer_rate_human']} (n={row['unsupported_answer_n']}).",
                "",
            ]
        )
    if not rows:
        report.extend(["No matching frozen policy and test records were supplied; no test metric is available.", ""])
    _atomic_text(root / "heldout_gate_report.md", "\n".join(report))
    return {"metrics_path": root / "heldout_gate_metrics.csv", "report_path": root / "heldout_gate_report.md", "results": rows}


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Calibrate and evaluate config-specific CampusReg evidence gates.")
    commands = parser.add_subparsers(dest="command", required=True)
    calibrate = commands.add_parser("calibrate", help="select thresholds using reviewed validation rows only")
    calibrate.add_argument("--predictions", required=True, help="paired_predictions.jsonl with human labels")
    calibrate.add_argument("--output-dir", required=True)
    calibrate.add_argument("--max-false-accept-rate", type=float, required=True)
    calibrate.add_argument("--max-unsupported-answer-rate", type=float, required=True)
    heldout = commands.add_parser("test", help="score frozen calibration policies on held-out test rows")
    heldout.add_argument("--calibration", required=True)
    heldout.add_argument("--predictions", required=True)
    heldout.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    records = [
        json.loads(line)
        for line in Path(args.predictions).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.command == "calibrate":
        result = calibrate_evidence_gates(
            records,
            output_dir=args.output_dir,
            max_false_accept_rate=args.max_false_accept_rate,
            max_unsupported_answer_rate=args.max_unsupported_answer_rate,
        )
        selected = len(result["policies"])
        print(f"Saved calibrated policies for {selected} retrieval configuration(s).")
        return 0
    value = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    policies = [EvidenceGatePolicy.from_dict(row) for row in value.get("policies", [])]
    result = evaluate_gate_on_test(policies, records, output_dir=args.output_dir)
    print(f"Saved held-out evidence gate report with {len(result['results'])} measured configuration(s).")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
