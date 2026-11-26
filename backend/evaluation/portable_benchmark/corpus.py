import argparse
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from importlib.metadata import version
from itertools import groupby
import json
from pathlib import Path
from textwrap import TextWrapper
from unicodedata import normalize

import pymupdf

from app.chunking.section_classifier import SectionClassifier
from app.chunking.text_chunker import TextChunk, TextChunker
from evaluation.portable_benchmark.pdf_output import open_pdf_output
from evaluation.portable_benchmark.schema import CorpusManifest, ExtractionSpec, PortableDataset


CORPUS_DIR = Path(__file__).parent


@dataclass(frozen=True)
class Passage:
    passage_id: str
    chunk: TextChunk


def decode_json(data, source="Report"):
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{source}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError(f"{source}: nonfinite JSON number {value!r}")

    def finite_float(value):
        number = float(value)
        if not isfinite(number):
            nonfinite(value)
        return number

    return json.loads(data, object_pairs_hook=unique_keys, parse_constant=nonfinite, parse_float=finite_float)


def read_json(path: Path):
    return decode_json(path.read_text(encoding="utf-8"), path.name)


def json_digest(value) -> str:
    canonical = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("ascii")).hexdigest()


def corpus_fingerprint(manifest: CorpusManifest) -> str:
    identity = manifest.model_dump(exclude={"corpus_fingerprint"})
    identity["documents"].sort(key=lambda document: document["key"])
    return json_digest(identity)


def protocol_fingerprint(manifest: CorpusManifest) -> str:
    return json_digest(manifest.model_dump(include={"extraction", "chunking", "passage_id_scheme"}))


def logical_passage_id(document_key: str, protocol: str, ordinal: int, text: str) -> str:
    normalized = " ".join(normalize("NFC", text).split())
    text_digest = sha256(normalized.encode("utf-8")).hexdigest()
    return f"p1:{document_key}:{ordinal}:{protocol}:{text_digest}"


def canonical_source(path: Path) -> str:
    text = path.read_bytes().decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip() or any(char != "\n" and not " " <= char <= "~" for char in text):
        raise ValueError(f"Source must be non-empty printable ASCII text encoded as UTF-8: {path.name}")
    return text


def load_corpus(root: Path) -> tuple[CorpusManifest, dict[str, str]]:
    root = root.resolve(strict=True)
    manifest = CorpusManifest.model_validate(read_json(root / "manifest.json"))
    sources = {}
    for document in manifest.documents:
        path = root / document.source_path
        if path.parent.is_symlink() or path.is_symlink() or path.resolve() != path:
            raise ValueError(f"Source path must not traverse links: {document.source_path}")
        if not path.is_file():
            raise ValueError(f"Missing source file: {document.source_path}")
        text = canonical_source(path)
        if sha256(text.encode("utf-8")).hexdigest() != document.source_sha256:
            raise ValueError(f"Source SHA-256 mismatch: {document.source_path}")
        sources[document.key] = text

    if manifest.corpus_fingerprint != corpus_fingerprint(manifest):
        raise ValueError("Manifest corpus fingerprint mismatch")
    return manifest, sources


def layout_pages(source: str, extraction: ExtractionSpec) -> list[list[str]]:
    wrapper = TextWrapper(width=extraction.line_width, break_on_hyphens=False)
    lines = [wrapped for line in source.splitlines() for wrapped in (wrapper.wrap(line) or [""])]
    size = extraction.lines_per_page
    return [lines[start:start + size] for start in range(0, len(lines), size)]


def render_pdf(source: str, extraction: ExtractionSpec) -> bytes:
    with pymupdf.open() as document:
        for lines in layout_pages(source, extraction):
            page = document.new_page(width=612, height=792)
            for row, line in enumerate(lines):
                if line:
                    page.insert_text((42, 48 + 13 * row), line, fontname="cour", fontsize=10)
        return document.tobytes(deflate=True, no_new_id=True)


def passages_for_pages(manifest: CorpusManifest, document_key: str, pages: list[dict]) -> list[Passage]:
    expected_version = manifest.chunking.tokenizer_package_version
    if version("tiktoken") != expected_version:
        raise ValueError(f"Passage protocol requires tiktoken {expected_version}")
    classifier = SectionClassifier()
    chunker = TextChunker(
        chunk_size=manifest.chunking.chunk_size,
        chunk_overlap=manifest.chunking.chunk_overlap,
        encoding_name=manifest.chunking.tokenizer,
    )
    protocol = protocol_fingerprint(manifest)
    segments = classifier.annotate_pages(pages)

    def section_on_page(segment):
        return segment["page_number"], segment["section"], segment["retrievable"]

    joined = [
        {
            "page_number": number,
            "section": section,
            "retrievable": retrievable,
            "text": " ".join(segment["text"] for segment in group),
        }
        for (number, section, retrievable), group in groupby(segments, section_on_page)
    ]
    return [
        Passage(logical_passage_id(document_key, protocol, chunk.chunk_index, chunk.text), chunk)
        for chunk in chunker.chunk_pages(joined)
    ]


def passage_catalog(manifest: CorpusManifest, sources: dict[str, str]) -> list[Passage]:
    catalog = []
    for document in sorted(manifest.documents, key=lambda spec: spec.key):
        pages = [
            {"page_number": number, "text": "\n".join(lines)}
            for number, lines in enumerate(layout_pages(sources[document.key], manifest.extraction), 1)
        ]
        catalog.extend(passages_for_pages(manifest, document.key, pages))
    return catalog


def load_dataset(
    root: Path, manifest: CorpusManifest, catalog: list[Passage], *, dataset_path: Path | None = None,
) -> PortableDataset:
    path = dataset_path if dataset_path is not None else root / "dataset.json"
    dataset = PortableDataset.model_validate(read_json(path))
    if dataset.corpus_fingerprint != manifest.corpus_fingerprint:
        raise ValueError("Dataset corpus fingerprint mismatch")

    eligible = {passage.passage_id for passage in catalog if passage.chunk.retrievable}
    for query in dataset.queries:
        for judgment in query.judgments:
            if judgment.passage_id not in eligible:
                raise ValueError(
                    f"{query.query_id}: unknown or non-retrievable passage {judgment.passage_id}"
                )
    return dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validate the portable AURA benchmark corpus.")
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR)
    parser.add_argument("--pdf-dir", type=Path, help="Generate text PDFs in a new directory")
    args = parser.parse_args(argv)
    try:
        manifest, sources = load_corpus(args.corpus_dir)
        catalog = passage_catalog(manifest, sources)
        dataset = load_dataset(args.corpus_dir, manifest, catalog)
        if args.pdf_dir is not None:
            with open_pdf_output(args.pdf_dir) as open_pdf:
                for key, source in sorted(sources.items()):
                    content = render_pdf(source, manifest.extraction)
                    with open_pdf(f"{key}.pdf") as file:
                        file.write(content)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(f"{manifest.corpus_name} {manifest.corpus_version}")
    print(f"Corpus SHA-256: {manifest.corpus_fingerprint}")
    print(f"Documents: {len(sources)}. Passages: {len(catalog)}. Queries: {len(dataset.queries)}.")
    print(f"Dataset review status: {dataset.review_status}.")


if __name__ == "__main__":
    main()
