"""Build and verify the portable ingestion inputs before any database writes."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from runpy import run_path

from app.embeddings.embedding_service import EmbeddingService
from app.services.pdf_extractor import extract_pdf_text
from evaluation.portable_benchmark.corpus import (
    Passage, load_corpus, load_dataset, passage_catalog, passages_for_pages,
    protocol_fingerprint, read_json, render_pdf,
)
from evaluation.portable_benchmark.pdf_output import open_pdf_output
from evaluation.portable_benchmark.seed_infra import SeedError


TEST_MODEL_PREPARATION = Path(__file__).resolve().parents[2] / "tests" / "prepare_embedding_model.py"


@dataclass(frozen=True)
class PreparedDocument:
    key: str
    pages: list[dict]
    passages: list[Passage]
    pdf_bytes: int
    pdf_sha256: str


@dataclass(frozen=True)
class SeedCorpus:
    fingerprint: str
    protocol: str
    corpus_version: str
    extraction: str
    chunking: str
    documents: list[PreparedDocument]


def verify_passage_ids(actual, reference, fingerprint):
    if (
        not isinstance(reference, dict) or type(reference.get("schema_version")) is not int
        or reference != {"schema_version": 1, "corpus_fingerprint": fingerprint, "passage_ids": actual}
        or len(actual) != 40
        or len(set(actual)) != 40
    ):
        raise SeedError("Passage identities differ from the reviewed 40-passage catalog")


def prepare_corpus(root, pdf_dir):
    manifest, sources = load_corpus(root)
    projected = passage_catalog(manifest, sources)
    load_dataset(root, manifest, projected)
    reference = read_json(root / "passage_ids.json")
    verify_passage_ids([item.passage_id for item in projected], reference, manifest.corpus_fingerprint)

    prepared = []
    with open_pdf_output(pdf_dir) as open_pdf:
        for key, source in sorted(sources.items()):
            content = render_pdf(source, manifest.extraction)
            with open_pdf(f"{key}.pdf") as file:
                file.write(content)
            extracted = extract_pdf_text(pdf_dir / f"{key}.pdf")
            pages = extracted["pages"]
            passages = passages_for_pages(manifest, key, pages)
            prepared.append(PreparedDocument(key, pages, passages, len(content), sha256(content).hexdigest()))
    verify_passage_ids(
        [passage.passage_id for document in prepared for passage in document.passages],
        reference, manifest.corpus_fingerprint,
    )
    return SeedCorpus(
        manifest.corpus_fingerprint, protocol_fingerprint(manifest), manifest.corpus_version,
        manifest.extraction.protocol, manifest.chunking.protocol, prepared,
    )


class LocalEmbeddingService(EmbeddingService):
    def __init__(self, model_path):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            str(model_path), device="cpu", local_files_only=True, trust_remote_code=False, token=False,
        )


def verified_model(directory):
    # Load the checked-in preparation helper by path. The tests directory can
    # otherwise resolve to a different installed package of the same name.
    prepare = run_path(str(TEST_MODEL_PREPARATION))
    manifest = prepare["load_manifest"]()
    if manifest["repo_id"] != EmbeddingService.MODEL_NAME:
        raise SeedError("The verified model manifest does not match AURA's MiniLM model")
    path = prepare["validate_model"](directory)
    return path, manifest
