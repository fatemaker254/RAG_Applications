"""
Step 1: Extract text from PDFs in DATA_DIR and split into chunks.

Chunking strategy:
- Detect section/sub-section headers using the PDF's actual font
  formatting (bold lines), not just text patterns. This is far more
  reliable than regex guessing, and correctly catches sub-module
  headers like "Registration", "Investigation", "Citizen Interface"
  in addition to top-level numbered sections like "1. INTRODUCTION".
- Lines are grouped with a vertical-position tolerance to handle
  small-caps/drop-cap styling that some PDFs use (where the first
  letter renders as a separate, slightly offset text run).
- If a document has no detectable bold headers at all, falls back to
  fixed-size chunking with overlap.

Each chunk keeps metadata: source file name + the section/sub-section
label it came from, so a generated test case can be traced back to
exactly where it came from.
"""

import os
import glob
import pdfplumber
from config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP

# Vertical-pixel tolerance for grouping characters into the same line.
# Handles PDFs where a heading's first letter is a slightly different
# size/baseline than the rest of the word (drop-cap / small-caps style).
LINE_TOLERANCE = 4

# A bold line only counts as a header if it's short - long bold
# sentences (rare, but possible) shouldn't be treated as headers.
MAX_HEADER_LENGTH = 90

# Fraction of non-space characters in a line that must be in a bold
# font for the whole line to count as bold.
BOLD_FRACTION_THRESHOLD = 0.8

# Minimum characters for something to count as a usable chunk. Filters
# out stray fragments, page numbers, and near-empty lines.
MIN_CHUNK_LENGTH = 60


def _group_chars_into_lines(chars, tolerance=LINE_TOLERANCE):
    """Group a page's characters into lines using vertical position,
    with a tolerance so drop-cap/small-caps fragments merge correctly."""
    chars_sorted = sorted(chars, key=lambda c: (c["top"], c["x0"]))
    lines = []
    current = []
    current_top = None
    for c in chars_sorted:
        if current_top is None or abs(c["top"] - current_top) <= tolerance:
            current.append(c)
            current_top = c["top"] if current_top is None else current_top
        else:
            lines.append(current)
            current = [c]
            current_top = c["top"]
    if current:
        lines.append(current)
    return lines


def _is_bold_line(line_chars) -> bool:
    non_space = [c for c in line_chars if c["text"].strip()]
    if not non_space:
        return False
    bold_count = sum(1 for c in non_space if "bold" in c["fontname"].lower())
    return (bold_count / len(non_space)) > BOLD_FRACTION_THRESHOLD


def extract_lines_with_headers(path: str):
    """
    Returns an ordered list of (text, is_header) tuples for the whole PDF.
    is_header is True when the line is bold and short enough to plausibly
    be a section/sub-section title rather than a bolded sentence.
    """
    lines_out = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            grouped = _group_chars_into_lines(page.chars)
            for line_chars in grouped:
                line_chars_sorted = sorted(line_chars, key=lambda c: c["x0"])
                text = "".join(c["text"] for c in line_chars_sorted).strip()
                if not text:
                    continue
                is_header = _is_bold_line(line_chars_sorted) and len(text) <= MAX_HEADER_LENGTH
                lines_out.append((text, is_header))
    return lines_out


def fixed_size_chunks(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Fallback: fixed-size sliding window chunking."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def chunk_document(lines, source_name: str):
    """
    Walks through (text, is_header) lines and groups them into chunks
    per header. Nested headers (e.g. a top-level section followed by
    sub-module headers within it) each start a fresh, topically-pure
    chunk - this is what fixes cross-topic bleed during retrieval.
    """
    chunks = []
    current_label = "General"
    current_lines = []
    found_any_header = False

    def flush():
        text = "\n".join(current_lines).strip()
        if len(text) < MIN_CHUNK_LENGTH:
            return
        if len(text) > CHUNK_SIZE * 1.5:
            for sub in fixed_size_chunks(text):
                if len(sub.strip()) >= MIN_CHUNK_LENGTH:
                    chunks.append({"text": sub, "source": source_name, "section": current_label})
        else:
            chunks.append({"text": text, "source": source_name, "section": current_label})

    for text, is_header in lines:
        if is_header:
            flush()
            current_label = text
            current_lines = []
            found_any_header = True
        else:
            current_lines.append(text)
    flush()

    if not found_any_header:
        # No bold headers detected anywhere -> pure fixed-size chunking
        full_text = "\n".join(t for t, _ in lines)
        chunks = [
            {"text": sub, "source": source_name, "section": "General"}
            for sub in fixed_size_chunks(full_text)
            if len(sub.strip()) >= MIN_CHUNK_LENGTH
        ]

    return chunks


def load_and_chunk_all_pdfs(data_dir: str = DATA_DIR):
    """Reads every PDF in data_dir, extracts + chunks it."""
    all_chunks = []
    pdf_paths = glob.glob(os.path.join(data_dir, "*.pdf"))

    if not pdf_paths:
        print(f"No PDFs found in '{data_dir}'. Add your requirement PDFs there.")
        return all_chunks

    for path in pdf_paths:
        name = os.path.basename(path)
        print(f"Extracting: {name}")
        lines = extract_lines_with_headers(path)
        doc_chunks = chunk_document(lines, name)
        print(f"  -> {len(doc_chunks)} chunks")
        all_chunks.extend(doc_chunks)

    return all_chunks


if __name__ == "__main__":
    chunks = load_and_chunk_all_pdfs()
    print(f"\nTotal chunks across all PDFs: {len(chunks)}")
    if chunks:
        print("\nSample sections found:")
        for c in chunks:
            print(f"  - {c['section']}")