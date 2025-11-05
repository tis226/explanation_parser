"""Visualize question explanation chunks with bounding boxes anchored to JSON question text."""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pdfplumber

try:
    from PIL import Image  # noqa: F401 - optional dependency required for rendering previews
except Exception:  # pragma: no cover - Pillow is optional in the runtime environment
    Image = None

from parse_explanations import normalize_text, order_page_lines, parse_question_heading

LOGGER = logging.getLogger(__name__)


@dataclass
class QuestionInfo:
    index: int
    question_number: Optional[int]
    normalized_signature: str
    question_text: str


@dataclass
class LineInfo:
    text: str
    page_number: int
    x0: float
    top: float
    x1: float
    bottom: float
    normalized: str


@dataclass
class VisualChunk:
    pdf_path: Path
    page_number: int
    question_number: Optional[int]
    question_index: Optional[int]
    label: str
    bbox: Sequence[float]


def build_question_map(records: Sequence[dict]) -> Dict[str, List[QuestionInfo]]:
    """Group questions by their number for lookup when parsing PDF chunks."""

    questions_by_number: Dict[str, List[QuestionInfo]] = defaultdict(list)

    for idx, record in enumerate(records):
        content = record.get("content", {})
        question_text = content.get("question_text", "")
        question_number = content.get("question_number")
        signature_seed = f"{question_number or ''}{question_text}"
        normalized_signature = normalize_text(signature_seed)
        question_info = QuestionInfo(
            index=idx,
            question_number=question_number,
            normalized_signature=normalized_signature,
            question_text=question_text,
        )
        key = str(question_number) if question_number is not None else "__none__"
        questions_by_number[key].append(question_info)

    return questions_by_number


def iter_pdf_chunks(pdf_path: Path) -> Iterable[List[LineInfo]]:
    """Yield ordered lists of lines representing question chunks from a PDF."""

    current_lines: List[LineInfo] = []
    current_question_active = False

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_number = int(page.page_number) + 1
            raw_lines = page.extract_text_lines() or []
            ordered_lines = order_page_lines(raw_lines, page.width)

            for raw_line in ordered_lines:
                text = (raw_line.get("text") or "").strip()
                if not text:
                    continue

                heading_number = parse_question_heading(text)
                line_info = LineInfo(
                    text=text,
                    page_number=page_number,
                    x0=float(raw_line.get("x0", 0.0)),
                    top=float(raw_line.get("top", 0.0)),
                    x1=float(raw_line.get("x1", 0.0)),
                    bottom=float(raw_line.get("bottom", 0.0)),
                    normalized=normalize_text(text),
                )

                if heading_number is not None:
                    if current_question_active and current_lines:
                        yield current_lines
                    current_lines = [line_info]
                    current_question_active = True
                else:
                    if not current_question_active:
                        continue
                    current_lines.append(line_info)

        if current_question_active and current_lines:
            yield current_lines


def find_matching_question(
    chunk_lines: Sequence[LineInfo],
    questions_by_number: Dict[str, List[QuestionInfo]],
) -> Optional[QuestionInfo]:
    """Match a chunk to a question using normalized question-number signatures."""

    if not chunk_lines:
        return None

    heading_line = chunk_lines[0]
    heading_number = parse_question_heading(heading_line.text)
    key = str(heading_number) if heading_number is not None else "__none__"
    candidates = questions_by_number.get(key, [])

    if not candidates:
        LOGGER.warning("No question candidates found for heading %s", heading_number)
        return None

    combined_normalized = "".join(line.normalized for line in chunk_lines)

    for candidate in candidates:
        if candidate.normalized_signature and candidate.normalized_signature in combined_normalized:
            return candidate

    # Fall back to the first candidate if no exact normalized match is found
    LOGGER.warning(
        "Could not find exact normalized match for question %s; falling back to first candidate.",
        heading_number,
    )
    return candidates[0]


def determine_start_index(
    chunk_lines: Sequence[LineInfo],
    normalized_signature: str,
) -> int:
    """Find the first line index where the normalized signature begins within the chunk."""

    if not normalized_signature:
        return 0

    normalized_segments = [line.normalized for line in chunk_lines]

    for end_idx in range(len(normalized_segments)):
        combined = ""
        for start_idx in range(end_idx + 1):
            combined = "".join(normalized_segments[start_idx : end_idx + 1])
            if normalized_signature in combined:
                return start_idx

    return 0


def compute_bounding_boxes(
    chunk_lines: Sequence[LineInfo],
    normalized_signature: str,
    question_number: Optional[int],
    question_index: Optional[int],
    pdf_path: Path,
) -> List[VisualChunk]:
    """Compute per-page bounding boxes for a matched chunk."""

    if not chunk_lines:
        return []

    start_idx = determine_start_index(chunk_lines, normalized_signature)
    relevant_lines = chunk_lines[start_idx:]

    if not relevant_lines:
        return []

    trimmed_lines: List[LineInfo] = []
    for idx, line in enumerate(relevant_lines):
        if idx > 0 and parse_question_heading(line.text) is not None:
            break
        trimmed_lines.append(line)

    if not trimmed_lines:
        return []

    lines_by_page: Dict[int, List[LineInfo]] = defaultdict(list)
    for line in trimmed_lines:
        lines_by_page[line.page_number].append(line)

    visual_chunks: List[VisualChunk] = []
    label = normalize_text(f"{question_number or ''}")
    if question_index is not None:
        label = f"Q{question_number or '?'}#{question_index}"

    for page_number, lines in lines_by_page.items():
        x0 = min(line.x0 for line in lines)
        top = min(line.top for line in lines)
        x1 = max(line.x1 for line in lines)
        bottom = max(line.bottom for line in lines)
        visual_chunks.append(
            VisualChunk(
                pdf_path=pdf_path,
                page_number=page_number,
                question_number=question_number,
                question_index=question_index,
                label=label,
                bbox=(x0, top, x1, bottom),
            )
        )

    return visual_chunks


def render_previews(
    visual_chunks: Sequence[VisualChunk],
    output_dir: Path,
    resolution: int = 144,
) -> None:
    """Render PNG previews for the computed bounding boxes."""

    if Image is None:
        LOGGER.error("Pillow is required to render previews. Install pillow and retry.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    chunks_by_pdf: Dict[Path, Dict[int, List[VisualChunk]]] = defaultdict(lambda: defaultdict(list))
    for chunk in visual_chunks:
        chunks_by_pdf[chunk.pdf_path][chunk.page_number].append(chunk)

    for pdf_path, pages in chunks_by_pdf.items():
        if not pdf_path.exists():
            LOGGER.warning("Skipping preview rendering for missing PDF: %s", pdf_path)
            continue
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page_chunks in pages.items():
                page_index = page_number - 1
                if page_index < 0 or page_index >= len(pdf.pages):
                    LOGGER.warning(
                        "Page number %s is out of bounds for %s (total pages %s)",
                        page_number,
                        pdf_path,
                        len(pdf.pages),
                    )
                    continue
                page_image = pdf.pages[page_index].to_image(resolution=resolution)
                for chunk in page_chunks:
                    x0, top, x1, bottom = chunk.bbox
                    page_image.draw_rect((x0, top, x1, bottom), stroke="blue", stroke_width=2)
                    label_position = (x0, max(top - 12, 0))
                    page_image.draw.text(label_position, chunk.label, fill="blue")
                output_path = output_dir / f"{pdf_path.stem}_page_{page_number:03d}.png"
                page_image.save(str(output_path))
                LOGGER.debug("Saved preview %s", output_path)


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate bounding-box previews for explanation chunks based on normalized question text."
        )
    )
    parser.add_argument("--json", required=True, type=Path, help="Path to the source JSON file.")
    parser.add_argument(
        "--pdf",
        required=True,
        nargs="+",
        type=Path,
        help="One or more PDF files containing explanations.",
    )
    parser.add_argument(
        "--preview-dir",
        required=True,
        type=Path,
        help="Directory to write PNG previews with bounding boxes.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=144,
        help="Rendering resolution (dots per inch) for preview images.",
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

    questions_by_number = build_question_map(records)

    visual_chunks: List[VisualChunk] = []

    skip_keywords = ["박우찬"]

    for pdf_path in args.pdf:
        if not pdf_path.exists():
            LOGGER.error("PDF not found: %s", pdf_path)
            continue

        if any(keyword in pdf_path.name for keyword in skip_keywords):
            LOGGER.info(
                "Skipping PDF %s because it uses a two-column layout not yet supported.",
                pdf_path,
            )
            continue

        for chunk_lines in iter_pdf_chunks(pdf_path):
            question = find_matching_question(chunk_lines, questions_by_number)
            heading_number = parse_question_heading(chunk_lines[0].text) if chunk_lines else None
            if question is None:
                LOGGER.warning(
                    "Skipping chunk for question %s because no matching JSON record was found.",
                    heading_number,
                )
                continue

            chunk_visuals = compute_bounding_boxes(
                chunk_lines,
                question.normalized_signature,
                question.question_number,
                question.index,
                pdf_path,
            )
            if not chunk_visuals:
                LOGGER.warning(
                    "No bounding boxes generated for question %s in %s.",
                    question.question_number,
                    pdf_path,
                )
                continue
            visual_chunks.extend(chunk_visuals)

    if not visual_chunks:
        LOGGER.warning("No visual chunks were generated; nothing to render.")
        return

    render_previews(visual_chunks, args.preview_dir, resolution=args.resolution)


if __name__ == "__main__":
    main()
