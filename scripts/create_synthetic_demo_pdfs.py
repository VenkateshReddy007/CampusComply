"""Create clearly synthetic local PDFs for software workflow checks only."""

from __future__ import annotations

from pathlib import Path


def _pdf(text: str) -> bytes:
    safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    content = f"BT /F1 11 Tf 36 720 Td ({safe}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    data = bytearray(b"%PDF-1.4\n% SYNTHETIC SOFTWARE TEST FIXTURE ONLY\n")
    offsets = [0]
    for number, body in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(f"{number} 0 obj\n".encode() + body + b"\nendobj\n")
    xref = len(data)
    data.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        data.extend(f"{offset:010d} 00000 n \n".encode())
    data.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return bytes(data)


def main() -> None:
    directory = Path("data/synthetic_demo")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "synthetic-alpha.pdf").write_bytes(_pdf(
        "SYNTHETIC SOFTWARE TEST FIXTURE ONLY. NOT A REAL CAMPUS POLICY. "
        "For a fake attendance exam eligibility rule, the practice cohort requires 70 percent attendance. "
        "This invented sentence exists only to test upload, retrieval, citations, and conflict displays."
    ))
    (directory / "synthetic-beta.pdf").write_bytes(_pdf(
        "SYNTHETIC SOFTWARE TEST FIXTURE ONLY. NOT A REAL CAMPUS POLICY. "
        "For a fake attendance exam eligibility rule, the practice cohort requires 75 percent attendance. "
        "This invented sentence exists only to test upload, retrieval, citations, and conflict displays."
    ))
    print(f"Created two synthetic software-only PDFs under {directory}.")
    print("Do not present these as real institutional regulations or use them for policy evaluation.")


if __name__ == "__main__":
    main()
