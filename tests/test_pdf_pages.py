"""Official sources that are PDFs are read as text, not reported unreachable (live: Theft Act 1968 PDF)."""

from __future__ import annotations

import httpx

from backend.evidence import sources
from backend.evidence.sources import check_support, fetch_page, pdf_to_text

LINE = "A person is guilty of theft if he dishonestly appropriates property belonging to another."


def make_pdf(text: str) -> bytes:
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


def test_pdf_text_is_extracted():
    assert "dishonestly appropriates" in pdf_to_text(make_pdf(LINE))


def test_garbage_or_empty_pdf_gives_empty_text_not_an_exception():
    assert pdf_to_text(b"%PDF-1.4 not really") == ""
    assert pdf_to_text(b"") == ""


async def test_fetch_page_reads_a_pdf_and_the_claim_is_confirmed(monkeypatch):
    pdf = make_pdf(LINE)

    def handler(request):
        return httpx.Response(200, content=pdf, headers={"content-type": "application/pdf"})

    real = httpx.AsyncClient

    class Client(real):
        def __init__(self, *a, **k):
            k["transport"] = httpx.MockTransport(handler)
            super().__init__(*a, **k)

    monkeypatch.setattr(sources.httpx, "AsyncClient", Client)
    page = await fetch_page("https://www.legislation.gov.uk/ukpga/1968/60/pdfs/ukpga_19680060_en.pdf")
    assert page.ok and "dishonestly appropriates" in page.text
    out = check_support("A person is guilty of theft if he dishonestly appropriates property belonging to another.", page)
    assert out["status"].value == "confirmed"