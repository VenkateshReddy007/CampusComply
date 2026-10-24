"""FastAPI integration tests using labelled synthetic PDFs and model doubles."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest
from fastapi.testclient import TestClient

from campusreg.api.app import APIServices, create_app
from campusreg.config import RAGSettings
from campusreg.rag.conflicts import ConflictAssessment, ConflictDetector, ConflictStatus
from campusreg.rag.conversation import ConversationRAGService
from campusreg.rag.pipeline import MultiDocumentRAGEngine
from campusreg.storage.sqlite_store import SQLiteDocumentStore


def synthetic_pdf(text: str) -> bytes:
    """Build a one-page PDF containing only clearly synthetic fixture text."""
    safe_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream_data = f"BT /F1 11 Tf 36 720 Td ({safe_text}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        f"<< /Length {len(stream_data)} >>\nstream\n".encode() + stream_data + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    output = bytearray(b"%PDF-1.4\n% synthetic test fixture\n")
    offsets = [0]
    for object_id, body in enumerate(objects, 1):
        offsets.append(len(output))
        output.extend(f"{object_id} 0 obj\n".encode() + body + b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode())
    output.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return bytes(output)


class FakeEmbedder:
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            [[1.0, 0.0] if any(term in text.casefold() for term in ("attendance", "exam", "hostel")) else [0.0, 1.0]
             for text in texts], dtype=np.float32,
        )


class FakeGenerator:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str, **kwargs: object) -> str:
        self.prompts.append(prompt)
        citations = list(dict.fromkeys(re.findall(r"\[[^\]\n]+\]", prompt)))
        return "SYNTHETIC TEST FIXTURE evidence is shown in the provided passages. " + " ".join(citations)


class FailingGenerator:
    def generate(self, prompt: str, **kwargs: object) -> str:
        raise RuntimeError("synthetic provider failure; do not expose internals")


class FakeComparator:
    def compare(self, a, b, **kwargs):
        return ConflictAssessment(
            "SYNTHETIC FIXTURE attendance eligibility",
            ConflictStatus.POTENTIAL_CONTRADICTION,
            "SYNTHETIC FIXTURE comparator detected different wording; human review is needed.",
            a.text,
            b.text,
            limitations=("Synthetic test only.",),
        )


class FakeRewriter:
    def rewrite(self, question, history):
        return "SYNTHETIC attendance exam requirement"


def make_services(tmp_path: Path, *, generator=None, requires_api_key: bool = False):
    settings = RAGSettings(chunk_size=80, chunk_overlap=0, top_k=8, relevance_threshold=0.4)
    store = SQLiteDocumentStore(tmp_path / "synthetic-api.sqlite3")
    selected_generator = generator or FakeGenerator()
    engine = MultiDocumentRAGEngine(settings, FakeEmbedder(), selected_generator, store)
    conversations = ConversationRAGService(store, engine, FakeRewriter())
    return APIServices(engine, conversations, ConflictDetector(FakeComparator()), requires_api_key)


def upload_two(client: TestClient) -> list[dict]:
    first = synthetic_pdf("SYNTHETIC FIXTURE attendance exam rule requires 75 percent attendance for eligibility.")
    second = synthetic_pdf("SYNTHETIC FIXTURE attendance exam rule requires 80 percent attendance for eligibility.")
    response = client.post("/api/documents", files=[
        ("files", ("synthetic-rule-alpha.pdf", first, "application/pdf")),
        ("files", ("synthetic-rule-beta.pdf", second, "application/pdf")),
    ])
    assert response.status_code == 201, response.text
    return response.json()["items"]


def test_health_ready_and_upload_listing_delete_and_evidence(tmp_path: Path) -> None:
    client = TestClient(create_app(make_services(tmp_path)))

    assert client.get("/api/health").json() == {"status": "ok"}
    assert client.get("/api/ready").json()["status"] in {"ready", "degraded"}
    uploaded = upload_two(client)
    assert [item["status"] for item in uploaded] == ["indexed", "indexed"]
    listed = client.get("/api/documents").json()
    assert [item["filename"] for item in listed] == ["synthetic-rule-alpha.pdf", "synthetic-rule-beta.pdf"]
    source = client.post("/api/questions", json={"question": "SYNTHETIC attendance exam eligibility"}).json()
    assert source["status"] == "answered"
    assert source["answer"]
    assert len(source["supporting_passages"]) == 2
    assert all(item["text"] in "\n".join(generator_prompt for generator_prompt in client.app.state.services.engine.generator.prompts) for item in source["supporting_passages"])
    first_source = source["supporting_passages"][0]
    evidence = client.get(f"/api/documents/{first_source['document_id']}/evidence/{first_source['chunk_id']}")
    assert evidence.status_code == 200
    assert evidence.json()["text"] == first_source["text"]
    reindexed = client.post(f"/api/documents/{listed[0]['id']}/reindex")
    assert reindexed.status_code == 200
    assert reindexed.json()["filename"] == listed[0]["filename"]
    assert client.delete(f"/api/documents/{listed[0]['id']}").json()["deleted"] is True
    assert len(client.get("/api/documents").json()) == 1


def test_unsupported_question_abstains_and_conflict_endpoint_exposes_two_real_fixture_passages(tmp_path: Path) -> None:
    client = TestClient(create_app(make_services(tmp_path)))
    upload_two(client)

    unsupported = client.post("/api/questions", json={"question": "SYNTHETIC astronomy galaxy facts"})
    assert unsupported.status_code == 200
    assert unsupported.json()["status"] == "abstained"
    assert unsupported.json()["answer"] == "The answer could not be found in the document."
    assert unsupported.json()["supporting_passages"] == []

    response = client.get("/api/conflicts")
    assert response.status_code == 200
    report = next(item for item in response.json()["reports"] if item["status"] == "potential_contradiction")
    assert report["passage_a"]["document_name"] == "synthetic-rule-alpha.pdf"
    assert report["passage_b"]["document_name"] == "synthetic-rule-beta.pdf"
    assert report["passage_a"]["page"] == report["passage_b"]["page"] == 1
    assert "SYNTHETIC FIXTURE" in report["passage_a"]["text"]
    assert report["human_review_required"] is True
    assert response.json()["authority_decided"] is False


def test_conversations_are_isolated_and_can_be_cleared(tmp_path: Path) -> None:
    client = TestClient(create_app(make_services(tmp_path)))
    upload_two(client)
    first = client.post("/api/conversations").json()["conversation_id"]
    second = client.post("/api/conversations").json()["conversation_id"]
    asked = client.post(
        f"/api/conversations/{first}/questions", json={"question": "SYNTHETIC attendance exam rules"}
    )
    assert asked.status_code == 200
    assert asked.json()["conversation_id"] == first
    follow_up = client.post(
        f"/api/conversations/{first}/questions", json={"question": "What about the deadline?"}
    )
    assert follow_up.status_code == 200
    assert follow_up.json()["retrieval_query"] == "SYNTHETIC attendance exam requirement"
    isolated = client.post(
        f"/api/conversations/{second}/questions", json={"question": "What about the deadline?"}
    )
    assert isolated.json()["status"] == "abstained"
    history = client.get(f"/api/conversations/{first}").json()["messages"]
    assert len(history) == 4
    assert client.delete(f"/api/conversations/{first}/messages").json()["cleared"] is True
    assert client.get(f"/api/conversations/{first}").json()["messages"] == []


def test_missing_api_configuration_is_a_structured_error_not_an_answer(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMMA_API_KEY", raising=False)
    monkeypatch.delenv("gemma_API_Key", raising=False)
    services = make_services(tmp_path, requires_api_key=True)
    client = TestClient(create_app(services))
    upload_two(client)

    response = client.post("/api/questions", json={"question": "SYNTHETIC attendance requirements"})

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "missing_api_configuration"
    assert "answer" not in response.json()
    assert "GOOGLE_API_KEY" not in client.get("/api/ready").text


def test_generation_failure_never_returns_a_normal_answer(tmp_path: Path) -> None:
    client = TestClient(create_app(make_services(tmp_path, generator=FailingGenerator())))
    upload_two(client)

    response = client.post("/api/questions", json={"question": "SYNTHETIC attendance exam rules"})

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "generation_error"
    assert body["answer"] is None
    assert body["citations"] == []
    assert body["error"]["code"] == "generation_failed"


def test_upload_mime_and_development_evaluation_access_are_restricted(tmp_path: Path, monkeypatch) -> None:
    client = TestClient(create_app(make_services(tmp_path), dev_evaluation_enabled=True))
    wrong_mime = client.post("/api/documents", files=[("files", ("synthetic.txt", b"fixture", "text/plain"))])
    assert wrong_mime.status_code == 201
    assert wrong_mime.json()["items"][0]["error"]["code"] == "invalid_mime_type"
    assert client.get("/api/development/evaluations").status_code == 401

    monkeypatch.setenv("CAMPUSREG_DEV_TOKEN", "synthetic-test-token")
    assert client.get("/api/development/evaluations", headers={"Authorization": "Bearer synthetic-test-token"}).status_code == 200
