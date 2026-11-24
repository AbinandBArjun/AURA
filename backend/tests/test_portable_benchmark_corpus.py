import errno
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
from pydantic import ValidationError

from app.services.pdf_extractor import extract_pdf_text
from evaluation.portable_benchmark import corpus
from evaluation.portable_benchmark.schema import CorpusManifest


FIXTURE_TEXT = (
    "Synthetic evaluation fixture\n\n"
    "Introduction\n"
    "The fictional archive had a separate recovery key.\n\n"
    "Discussion\n"
    "Reviewers checked amendments after the recovery exercise.\n\n"
    "References\n"
    "Fictional references are excluded from retrieval.\n"
)


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def draft(tmp_path):
    data = corpus.read_json(corpus.CORPUS_DIR / "manifest.json")
    data["documents"] = [
        dict(
            data["documents"][0],
            key=key,
            source_path=f"sources/{key}.txt",
            source_sha256=sha256(FIXTURE_TEXT.encode("utf-8")).hexdigest(),
        )
        for key in ("document-one", "document-two")
    ]
    data["corpus_fingerprint"] = corpus.corpus_fingerprint(CorpusManifest.model_validate(data))
    (tmp_path / "sources").mkdir()
    for doc in data["documents"]:
        (tmp_path / doc["source_path"]).write_bytes(FIXTURE_TEXT.encode("utf-8"))
    write_json(tmp_path / "manifest.json", data)
    dataset = {
        "schema_version": 1,
        "corpus_fingerprint": data["corpus_fingerprint"],
        "review_status": "draft",
        "queries": [
            {
                "query_id": "recovery-control",
                "category": "cybersecurity",
                "query": "What control separated emergency access?",
                "judgments": [],
            }
        ],
    }
    write_json(tmp_path / "dataset.json", dataset)
    return tmp_path, data, dataset


def test_corpus_and_draft_are_valid():
    manifest, sources = corpus.load_corpus(corpus.CORPUS_DIR)
    catalog = corpus.passage_catalog(manifest, sources)
    dataset = corpus.load_dataset(corpus.CORPUS_DIR, manifest, catalog)

    assert len(sources) == 8
    assert len({doc.domain for doc in manifest.documents}) == 4
    assert len(catalog) == len({entry.passage_id for entry in catalog}) == 40
    assert all(sum(f":{key}:" in item.passage_id for item in catalog) > 1 for key in sources)
    assert len(dataset.queries) == 36
    assert dataset.review_status == "draft"
    assert all(not query.judgments for query in dataset.queries)


def test_fingerprint_and_ids_survive_relocation_crlf_and_manifest_order(draft, tmp_path):
    original, data, _ = draft
    manifest, sources = corpus.load_corpus(original)
    catalog = corpus.passage_catalog(manifest, sources)
    destination = tmp_path / "copy"
    shutil.copytree(original, destination, ignore=shutil.ignore_patterns("copy"))
    data["documents"].reverse()
    write_json(destination / "manifest.json", data)
    for path in (destination / "sources").iterdir():
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))

    relocated, relocated_sources = corpus.load_corpus(destination)
    assert relocated.corpus_fingerprint == manifest.corpus_fingerprint
    assert corpus.passage_catalog(relocated, relocated_sources) == catalog


def test_passage_ids_normalize_unicode_and_space_and_distinguish_identity():
    key, protocol = "fictional", "a" * 64
    entry = corpus.logical_passage_id(key, protocol, 0, "Cafe\u0301 \n sample")
    assert entry == corpus.logical_passage_id(key, protocol, 0, "Caf\u00e9 sample")
    assert entry != corpus.logical_passage_id(key, protocol, 0, "Caf\u00e9 example")
    assert entry != corpus.logical_passage_id("other", protocol, 0, "Caf\u00e9 sample")
    assert entry != corpus.logical_passage_id(key, "b" * 64, 0, "Caf\u00e9 sample")
    assert entry != corpus.logical_passage_id(key, protocol, 1, "Caf\u00e9 sample")


@pytest.mark.parametrize("path", ["../outside.txt", "sources/../outside.txt", "/source.txt", "C:/source.txt",
                                  "sources\\source.txt", "sources/subfolder/source.txt"])
def test_unsafe_source_paths_are_rejected(draft, path):
    root, manifest, _ = draft
    manifest["documents"][0]["source_path"] = path
    write_json(root / "manifest.json", manifest)

    with pytest.raises(ValidationError, match="source_path"):
        corpus.load_corpus(root)


@pytest.mark.parametrize(("key", "message"), [("key", "Duplicate document key"),
                                               ("source_path", "Duplicate document source path")])
def test_duplicate_document_fields_are_rejected(draft, key, message):
    root, manifest, _ = draft
    manifest["documents"][1][key] = manifest["documents"][0][key]
    write_json(root / "manifest.json", manifest)

    with pytest.raises(ValidationError, match=message):
        corpus.load_corpus(root)


@pytest.mark.parametrize(
    ("mutation", "error", "message"),
    [
        ("missing", ValueError, "Missing source file"),
        ("changed", ValueError, "Source SHA-256 mismatch"),
        ("bad-hash", ValidationError, "source_sha256"),
        ("bad-fingerprint", ValueError, "Manifest corpus fingerprint mismatch"),
        ("bad-version", ValidationError, "schema_version"),
        ("boolean-version", ValidationError, "schema_version"),
    ],
)
def test_manifest_validation_fails_for_invalid_inputs(draft, mutation, error, message):
    root, manifest, _ = draft
    source = root / manifest["documents"][0]["source_path"]
    if mutation == "missing":
        source.unlink()
    elif mutation == "changed":
        source.write_bytes(source.read_bytes() + b"Changed\n")
    elif mutation == "bad-hash":
        manifest["documents"][0]["source_sha256"] = "not-a-digest"
    elif mutation == "bad-fingerprint":
        manifest["corpus_fingerprint"] = "1" * 64
    else:
        manifest["schema_version"] = True if mutation == "boolean-version" else 2
    write_json(root / "manifest.json", manifest)

    with pytest.raises(error, match=message):
        corpus.load_corpus(root)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("foreign-corpus", "Dataset corpus fingerprint mismatch"),
        ("unknown-passage", "unknown or non-retrievable passage"),
        ("excluded-passage", "unknown or non-retrievable passage"),
        ("invalid-grade", "grade"),
        ("unreviewed", "Reviewed dataset contains pending judgments"),
        ("self-reviewed", "different reviewer"),
        ("duplicate-query", "Duplicate query ID"),
    ],
)
def test_dataset_validation_fails_for_invalid_judgments(draft, mutation, message):
    root, _, data = draft
    manifest, sources = corpus.load_corpus(root)
    catalog = corpus.passage_catalog(manifest, sources)
    eligible = next(item for item in catalog if item.chunk.retrievable)
    excluded = next(item for item in catalog if not item.chunk.retrievable)
    judgment = {
        "passage_id": eligible.passage_id,
        "grade": 2,
        "method": "human",
        "assessor_id": "assessor",
        "review_status": "pending",
    }
    data["queries"][0]["judgments"].append(judgment)
    if mutation == "foreign-corpus":
        data["corpus_fingerprint"] = "1" * 64
    elif mutation == "unknown-passage":
        judgment["passage_id"] = corpus.logical_passage_id("absent", "a" * 64, 0, "example")
    elif mutation == "excluded-passage":
        judgment["passage_id"] = excluded.passage_id
    elif mutation == "invalid-grade":
        judgment["grade"] = 4
    elif mutation == "unreviewed":
        data["review_status"] = "reviewed"
    elif mutation == "self-reviewed":
        judgment.update(review_status="reviewed", reviewer_id="assessor")
    else:
        data["queries"].append(data["queries"][0])
    write_json(root / "dataset.json", data)

    with pytest.raises(ValueError, match=message):
        corpus.load_dataset(root, manifest, catalog)


def test_schema_accepts_independently_reviewed_synthetic_fixture(draft):
    root, _, data = draft
    manifest, sources = corpus.load_corpus(root)
    catalog = corpus.passage_catalog(manifest, sources)
    eligible = [item for item in catalog if item.chunk.retrievable]
    data["review_status"] = "reviewed"
    data["queries"][0]["judgments"] = [
        {
            "passage_id": item.passage_id,
            "grade": grade,
            "method": "human",
            "assessor_id": "assessor",
            "review_status": "reviewed",
            "reviewer_id": "reviewer",
        }
        for item, grade in zip(eligible[:2], (2, 0), strict=True)
    ]
    write_json(root / "dataset.json", data)

    dataset = corpus.load_dataset(root, manifest, catalog)
    assert [judgment.grade for judgment in dataset.queries[0].judgments] == [2, 0]


def test_generated_pdf_is_repeatable_and_extracts_the_page_projection(tmp_path):
    manifest, sources = corpus.load_corpus(corpus.CORPUS_DIR)
    for key, text in sources.items():
        pdf = tmp_path / f"{key}.pdf"
        data = corpus.render_pdf(text, manifest.extraction)
        pdf.write_bytes(data)
        assert corpus.render_pdf(text, manifest.extraction) == data
        extracted = extract_pdf_text(pdf)["pages"]
        expected = corpus.layout_pages(text, manifest.extraction)
        assert [page["text"].splitlines() for page in extracted] == [
            [line for line in page if line] for page in expected
        ]


def test_duplicate_json_object_keys_are_rejected(draft):
    root, _, _ = draft
    (root / "manifest.json").write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key"):
        corpus.load_corpus(root)


def test_pdf_cli_writes_repeatable_pdfs(draft, capsys):
    root, _, _ = draft
    manifest, sources = corpus.load_corpus(root)
    outputs = [root / "pdfs-first", root / "pdfs-second"]
    for output in outputs:
        corpus.main(["--corpus-dir", str(root), "--pdf-dir", str(output)])
        assert sorted(path.name for path in output.iterdir()) == [f"{key}.pdf" for key in sorted(sources)]

    for key, source in sources.items():
        first, second = (output / f"{key}.pdf" for output in outputs)
        assert first.read_bytes() == second.read_bytes()
        assert [page["text"].splitlines() for page in extract_pdf_text(first)["pages"]] == [
            [line for line in page if line] for page in corpus.layout_pages(source, manifest.extraction)
        ]
    assert "Documents: 2." in capsys.readouterr().out


def test_pdf_cli_preserves_file_created_during_rendering(draft, monkeypatch, capsys):
    root, _, _ = draft
    output = root / "pdfs"
    destination = output / "document-one.pdf"
    canary = b"existing destination contents"
    render = corpus.render_pdf

    def render_with_collision(source, extraction):
        destination.write_bytes(canary)
        return render(source, extraction)

    monkeypatch.setattr(corpus, "render_pdf", render_with_collision)
    with pytest.raises(SystemExit) as failure:
        corpus.main(["--corpus-dir", str(root), "--pdf-dir", str(output)])
    assert failure.value.code == 2
    output_log = capsys.readouterr()
    assert "PDF destination already exists" in output_log.err
    assert "document-one.pdf" in output_log.err
    assert not output_log.out
    assert destination.read_bytes() == canary
    assert not (output / "document-two.pdf").exists()


def test_pdf_cli_refuses_preexisting_output_directory(draft):
    root, _, _ = draft
    output = root / "pdfs"
    output.mkdir()
    destination = output / "document-one.pdf"
    destination.write_bytes(b"pre-existing PDF data")

    result = subprocess.run(
        [sys.executable, "-m", "evaluation.portable_benchmark.corpus",
         "--corpus-dir", str(root), "--pdf-dir", str(output)],
        cwd=Path(corpus.__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert str(output) in result.stderr
    assert not result.stdout
    assert destination.read_bytes() == b"pre-existing PDF data"


def test_pdf_cli_reports_partial_generation_without_cleaning_other_files(
    draft, monkeypatch, capsys
):
    root, _, _ = draft
    output = root / "pdfs"
    unrelated = root / "keep.txt"
    unrelated.write_bytes(b"untouched")
    render = corpus.render_pdf

    def fail_after_first_pdf(source, extraction):
        if (output / "document-one.pdf").exists():
            raise OSError("simulated PDF generation failure")
        return render(source, extraction)

    monkeypatch.setattr(corpus, "render_pdf", fail_after_first_pdf)
    with pytest.raises(SystemExit) as failure:
        corpus.main(["--corpus-dir", str(root), "--pdf-dir", str(output)])
    assert failure.value.code == 2
    output_log = capsys.readouterr()
    assert "simulated PDF generation failure" in output_log.err
    assert not output_log.out
    assert [item.name for item in output.iterdir()] == ["document-one.pdf"]
    assert unrelated.read_bytes() == b"untouched"


def symlink_or_skip(link, target, *, directory=False):
    try:
        link.symlink_to(target, target_is_directory=directory)
    except OSError as error:
        if error.errno not in (errno.EACCES, errno.EPERM, errno.ENOSYS, errno.ENOTSUP):
            if getattr(error, "winerror", None) != 1314:
                raise
        pytest.skip("This test environment cannot create symbolic links")


def test_pdf_cli_rejects_symlinked_output_parent(draft, capsys):
    root, _, _ = draft
    other = root / "unrelated"
    other.mkdir()
    link = root / "redirect"
    symlink_or_skip(link, other, directory=True)

    with pytest.raises(SystemExit) as failure:
        corpus.main(["--corpus-dir", str(root), "--pdf-dir", str(link / "pdfs")])
    assert failure.value.code == 2
    assert not capsys.readouterr().out
    assert not list(other.iterdir())


def test_pdf_cli_does_not_follow_a_destination_symlink(draft, monkeypatch, capsys):
    root, _, _ = draft
    output = root / "pdfs"
    canary = root / "unrelated.txt"
    canary.write_bytes(b"untouched link target")
    render = corpus.render_pdf

    def render_with_symlink(source, extraction):
        symlink_or_skip(output / "document-one.pdf", canary)
        return render(source, extraction)

    monkeypatch.setattr(corpus, "render_pdf", render_with_symlink)
    with pytest.raises(SystemExit) as failure:
        corpus.main(["--corpus-dir", str(root), "--pdf-dir", str(output)])
    assert failure.value.code == 2
    assert not capsys.readouterr().out
    assert canary.read_bytes() == b"untouched link target"


def test_pdf_cli_detects_or_blocks_replaced_output_directory(draft, monkeypatch, capsys):
    root, _, _ = draft
    output = root / "pdfs"
    displaced = root / "displaced"
    render = corpus.render_pdf
    replacement = None

    def render_with_replacement(source, extraction):
        nonlocal replacement
        if replacement is None:
            try:
                output.rename(displaced)
            except PermissionError:
                if os.name != "nt":
                    raise
                replacement = False
            else:
                replacement = True
                output.mkdir()
                (output / "document-one.pdf").write_bytes(b"unrelated directory")
        return render(source, extraction)

    monkeypatch.setattr(corpus, "render_pdf", render_with_replacement)
    try:
        corpus.main(["--corpus-dir", str(root), "--pdf-dir", str(output)])
    except SystemExit as error:
        assert replacement is True
        assert error.code == 2
        output_log = capsys.readouterr()
        assert "PDF output directory changed" in output_log.err
        assert not output_log.out
        assert (output / "document-one.pdf").read_bytes() == b"unrelated directory"
        assert not list(displaced.iterdir())
    else:
        assert os.name == "nt"
        assert replacement is False
        assert not displaced.exists()
        assert len(list(output.glob("*.pdf"))) == 2


def test_pdf_cli_rejects_parent_path_traversal(draft, capsys):
    root, _, _ = draft
    collision = root / "pdfs"
    with pytest.raises(SystemExit) as failure:
        corpus.main(["--corpus-dir", str(root), "--pdf-dir", str(root / "parent" / ".." / "pdfs")])
    assert failure.value.code == 2
    output_log = capsys.readouterr()
    assert "without '..' path components" in output_log.err
    assert not output_log.out
    assert not collision.exists()
