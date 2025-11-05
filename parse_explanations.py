"""Utilities for parsing explanation PDFs and attaching them to question data."""
from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

import pdfplumber


LOGGER = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Normalize text by removing whitespace, punctuation, and lowercasing."""
    if text is None:
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.lower()
    normalized = re.sub(r"\s+", "", normalized)
    normalized = re.sub(r"[^0-9a-z가-힣]", "", normalized)
    return normalized


def build_signature(question_text: str, question_number: Optional[int]) -> str:
    normalized = normalize_text(question_text)
    if question_number is not None:
        return f"{normalized}#{question_number}"
    return normalized


@dataclass
class QuestionRecord:
    index: int
    question_number: Optional[int]
    signature: str
    normalized_text: str
    data: dict


@dataclass
class Chunk:
    question_number: Optional[int]
    text_lines: List[str] = field(default_factory=list)
    page_number: int = 0
    source_pdf: str = ""
    bbox: Optional[List[float]] = None

    def append_line(self, line: dict) -> None:
        text = line.get("text", "")
        if text:
            self.text_lines.append(text)
        x0 = float(line.get("x0", 0))
        x1 = float(line.get("x1", 0))
        top = float(line.get("top", 0))
        bottom = float(line.get("bottom", 0))
        if self.bbox is None:
            self.bbox = [x0, top, x1, bottom]
        else:
            self.bbox[0] = min(self.bbox[0], x0)
            self.bbox[1] = min(self.bbox[1], top)
            self.bbox[2] = max(self.bbox[2], x1)
            self.bbox[3] = max(self.bbox[3], bottom)

    @property
    def text(self) -> str:
        return "\n".join(self.text_lines).strip()

    def signature(self) -> str:
        base = normalize_text(self.text)
        if self.question_number is not None:
            return f"{base}#{self.question_number}"
        return base


QUESTION_HEADING_PATTERNS = [
    re.compile(r"^(?P<num>\d{1,3})[\.:]\s*"),
    re.compile(r"^\((?P<num>\d{1,3})\)"),
    re.compile(r"^(?P<num>\d{1,3})\s+번"),
]


def parse_question_heading(text: str) -> Optional[int]:
    stripped = text.strip()
    for pattern in QUESTION_HEADING_PATTERNS:
        match = pattern.match(stripped)
        if match:
            try:
                return int(match.group("num"))
            except (TypeError, ValueError):
                return None
    return None


def detect_two_column_layout(lines: Sequence[dict], page_width: float) -> bool:
    """Heuristic to determine whether a page uses a two-column layout."""
    if not lines:
        return False

    x_positions = [float(line.get("x0", 0.0)) for line in lines]
    left_positions = [x for x in x_positions if x < page_width * 0.5]
    right_positions = [x for x in x_positions if x >= page_width * 0.5]

    if len(left_positions) < 3 or len(right_positions) < 3:
        return False

    left_max = max(left_positions)
    right_min = min(right_positions)
    column_gap = right_min - left_max

    if column_gap <= page_width * 0.05:
        return False

    LOGGER.debug(
        "Detected potential two-column layout (gap=%.2f, width=%.2f)",
        column_gap,
        page_width,
    )
    return True


def order_page_lines(lines: Sequence[dict], page_width: float) -> List[dict]:
    """Return page lines ordered according to detected layout."""
    if detect_two_column_layout(lines, page_width):
        left_column = [line for line in lines if float(line.get("x0", 0.0)) < page_width * 0.5]
        right_column = [line for line in lines if float(line.get("x0", 0.0)) >= page_width * 0.5]

        left_column.sort(key=lambda l: (l.get("top", 0), l.get("x0", 0)))
        right_column.sort(key=lambda l: (l.get("top", 0), l.get("x0", 0)))

        LOGGER.debug(
            "Page width %.2f -> treating as two columns (left=%d, right=%d)",
            page_width,
            len(left_column),
            len(right_column),
        )

        return left_column + right_column

    ordered = list(lines)
    ordered.sort(key=lambda l: (l.get("top", 0), l.get("x0", 0)))
    return ordered


def extract_chunks_from_page(page, source_pdf: str) -> Iterator[Chunk]:
    lines = page.extract_text_lines() or []
    ordered_lines = order_page_lines(lines, page.width)

    current_chunk: Optional[Chunk] = None

    for line in ordered_lines:
        text = (line.get("text") or "").strip()
        if not text:
            continue

        heading_number = parse_question_heading(text)

        if heading_number is not None:
            if current_chunk is not None and current_chunk.text_lines:
                yield current_chunk
            current_chunk = Chunk(
                question_number=heading_number,
                page_number=int(page.page_number) + 1,
                source_pdf=source_pdf,
            )
            current_chunk.append_line(line)
        else:
            if current_chunk is None:
                # Ignore text before the first heading on a page
                continue
            current_chunk.append_line(line)

    if current_chunk is not None and current_chunk.text_lines:
        yield current_chunk


def extract_pdf_chunks(pdf_path: Path) -> List[Chunk]:
    chunks: List[Chunk] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for chunk in extract_chunks_from_page(page, str(pdf_path)):
                chunks.append(chunk)
    LOGGER.info("Extracted %d chunks from %s", len(chunks), pdf_path)
    return chunks


def build_question_index(records: Sequence[dict]) -> Dict[str, List[QuestionRecord]]:
    index_by_number: Dict[str, List[QuestionRecord]] = defaultdict(list)
    for idx, record in enumerate(records):
        content = record.get("content", {})
        question_text = content.get("question_text", "")
        question_number = content.get("question_number")
        signature = build_signature(question_text, question_number)
        normalized_text = normalize_text(question_text)
        question_record = QuestionRecord(
            index=idx,
            question_number=question_number,
            signature=signature,
            normalized_text=normalized_text,
            data=record,
        )
        index_key = str(question_number) if question_number is not None else "__none__"
        index_by_number[index_key].append(question_record)
    return index_by_number


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def match_chunks(
    chunks: Iterable[Chunk],
    index_by_number: Dict[str, List[QuestionRecord]],
    min_ratio: float = 0.45,
) -> Dict[int, List[dict]]:
    matches: Dict[int, List[dict]] = defaultdict(list)

    all_records = [record for records in index_by_number.values() for record in records]

    for chunk in chunks:
        normalized_chunk_text = normalize_text(chunk.text)
        question_number_key = (
            str(chunk.question_number)
            if chunk.question_number is not None
            else "__none__"
        )
        candidates = index_by_number.get(question_number_key, [])
        best_record: Optional[QuestionRecord] = None
        best_score = -1.0

        # First try question-number-specific matches
        for candidate in candidates:
            score = similarity(normalized_chunk_text, candidate.normalized_text)
            if score > best_score:
                best_score = score
                best_record = candidate

        # Fallback to global search if necessary
        if best_record is None or best_score < min_ratio:
            for candidate in all_records:
                score = similarity(normalized_chunk_text, candidate.normalized_text)
                if score > best_score:
                    best_score = score
                    best_record = candidate

        if best_record is None or best_score < min_ratio:
            LOGGER.warning(
                "Could not confidently match chunk for question %s (score=%.2f)",
                chunk.question_number,
                best_score,
            )
            continue

        explanation_payload = {
            "text": chunk.text,
            "source_pdf": chunk.source_pdf,
            "page": chunk.page_number,
            "bbox": chunk.bbox,
            "similarity": best_score,
        }
        matches[best_record.index].append(explanation_payload)
        LOGGER.debug(
            "Matched chunk (question %s) to record %d with score %.2f",
            chunk.question_number,
            best_record.index,
            best_score,
        )

    return matches


def attach_explanations(records: List[dict], matches: Dict[int, List[dict]]) -> List[dict]:
    result = []
    for idx, record in enumerate(records):
        record_copy = json.loads(json.dumps(record))  # deep copy via JSON for immutability
        content = record_copy.setdefault("content", {})
        if matches.get(idx):
            content["explanations"] = matches[idx]
        result.append(record_copy)
    return result


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse explanation PDFs and merge into JSON data.")
    parser.add_argument("--json", required=True, type=Path, help="Path to the source JSON file.")
    parser.add_argument(
        "--pdf",
        required=True,
        nargs="+",
        type=Path,
        help="One or more PDF files containing explanations.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write the augmented JSON file. If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.45,
        help="Minimum similarity score required to attach an explanation.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (e.g., INFO, DEBUG).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_arguments(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    with args.json.open("r", encoding="utf-8") as fp:
        records = json.load(fp)

    index_by_number = build_question_index(records)

    all_chunks: List[Chunk] = []
    for pdf_path in args.pdf:
        if not pdf_path.exists():
            LOGGER.error("PDF not found: %s", pdf_path)
            continue
        all_chunks.extend(extract_pdf_chunks(pdf_path))

    matches = match_chunks(all_chunks, index_by_number, min_ratio=args.min_score)
    augmented_records = attach_explanations(records, matches)

    output_json = json.dumps(augmented_records, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(output_json, encoding="utf-8")
    else:
        print(output_json)


if __name__ == "__main__":
    main()
