"""pdf2txt — pure-Python PDF to plaintext (pypdf), no system deps (no poppler).

Usage:  python tools/pdf2txt.py <in.pdf> [out.txt]
Writes page-delimited UTF-8 text so a long report can be read in chunks. Built because the
harness PDF reader needs poppler's pdftoppm, which isn't installed here.
"""

import sys

from pypdf import PdfReader


def main():
    if len(sys.argv) < 2:
        print("usage: python tools/pdf2txt.py <in.pdf> [out.txt]")
        sys.exit(2)
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else src.rsplit(".", 1)[0] + ".txt"
    reader = PdfReader(src)
    parts = []
    for i, page in enumerate(reader.pages):
        parts.append(f"\n\n===== PAGE {i + 1} =====\n")
        parts.append(page.extract_text() or "")
    text = "".join(parts)
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"{len(reader.pages)} pages, {len(text)} chars -> {out}")


if __name__ == "__main__":
    main()
