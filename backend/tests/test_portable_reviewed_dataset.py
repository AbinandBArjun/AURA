from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from evaluation.portable_benchmark import export_reviewed_dataset as export
from evaluation.portable_benchmark.corpus import load_dataset, read_json


@pytest.fixture(scope="module")
def reviewed_artifacts():
    manifest, catalog, draft = export.canonical_inputs()
    root = export.CORPUS_DIR
    dataset = load_dataset(root, manifest, catalog, dataset_path=root / export.DATASET_NAME)
    record = read_json(root / export.RECORD_NAME)
    return dataset, catalog, draft, record


def test_reviewed_dataset_exhaustively_matches_canonical_inputs_and_approval(reviewed_artifacts):
    dataset, _, draft, record = reviewed_artifacts
    export.validate_artifacts(*reviewed_artifacts)
    assert draft.schema_version == 1 and draft.review_status == "draft"
    assert all(not query.judgments for query in draft.queries)
    assert sum(len(query.judgments) for query in dataset.queries) == 1440
    assert len(record["flagged_resolutions"]) == 41
    for query in dataset.queries:
        for judgment in query.judgments:
            assert judgment.assessor_id == "openai-codex-ai-assisted-draft"
            assert judgment.reviewer_id == "aura-project-owner"
            assert judgment.ai_provenance.model is None
            assert judgment.human_review.approval_reference == export.RECORD_NAME


def test_reviewed_dataset_export_recreates_included_bytes(reviewed_artifacts):
    dataset, catalog, draft, record = reviewed_artifacts
    pairs = [
        ((query.query_id, judgment.passage_id), judgment.grade)
        for query in dataset.queries for judgment in query.judgments
    ]
    regenerated = export.build_dataset(catalog, draft, dict(reversed(pairs)), record)
    assert export.export_bytes(regenerated) == (
        export.CORPUS_DIR / export.DATASET_NAME
    ).read_text(encoding="ascii").encode("ascii")


@pytest.mark.parametrize(("defect", "message"), [
    ("missing judgment", "exactly cover"),
    ("wrong reviewer", "provenance mismatch"),
    ("wrong flag grade", "inconsistent flagged resolution"),
    ("duplicate flag", "inconsistent flagged resolution"),
    ("wrong confirmed count", "record counts mismatch"),
])
def test_reviewed_artifact_validation_rejects_inconsistencies(reviewed_artifacts, defect, message):
    original, catalog, draft, original_record = reviewed_artifacts
    dataset = original.model_copy(deep=True)
    record = deepcopy(original_record)
    if defect == "missing judgment":
        dataset.queries[0].judgments.pop()
    elif defect == "wrong reviewer":
        dataset.queries[0].judgments[0].reviewer_id = "other-reviewer"
    elif defect == "wrong flag grade":
        flag = record["flagged_resolutions"][0]
        flag["reviewed_grade"] = flag["ai_alternative_grade"]
    elif defect == "duplicate flag":
        record["flagged_resolutions"][1] = record["flagged_resolutions"][0]
    else:
        record["counts"]["grade_changes"] = 1
    with pytest.raises(ValueError, match=message):
        export.validate_artifacts(dataset, catalog, draft, record)


def test_workbook_must_match_pinned_hash_before_parsing(tmp_path):
    replacement = tmp_path / "aura-relevance-assessment-draft.xlsx"
    replacement.write_bytes(b"not the original workbook")
    with pytest.raises(ValueError, match="Source workbook SHA-256 mismatch"):
        export.workbook_rows(replacement)


@pytest.fixture
def approval_fixture(reviewed_artifacts, tmp_path, monkeypatch):
    _, catalog, draft, record = reviewed_artifacts
    manifest, _, _ = export.canonical_inputs()
    shutil.copyfile(export.CORPUS_DIR / export.DATASET_NAME, tmp_path / export.DATASET_NAME)
    monkeypatch.setattr(export, "CORPUS_DIR", tmp_path)
    monkeypatch.setattr(export, "canonical_inputs", lambda: (manifest, catalog, draft))
    return deepcopy(record), tmp_path / export.RECORD_NAME


@pytest.mark.parametrize(("field", "value", "missing", "message"), [
    ("approval_statement", None, True, "approval_statement"),
    ("approval_statement", " ", False, "approval_statement"),
    ("approval_statement", "Approved.", False, "approval_statement"),
    ("confirmed_interpretations", None, True, "confirmed interpretations"),
    ("reviewer_id", "someone-else", False, "reviewer_id"),
    ("approved_on", "2026-11-28", False, "approved_on"),
    ("decision", "rejected", False, "decision"),
    ("ai_provenance", {"provider": "OpenAI", "tool": "Codex", "model": "invented"}, False, "ai_provenance"),
    ("source_workbook", {}, False, "source_workbook"),
    ("corpus_fingerprint", "0" * 64, False, "corpus_fingerprint"),
    ("dataset_fingerprint", None, True, "fingerprint mismatch"),
    ("dataset_fingerprint", "0" * 64, False, "fingerprint mismatch"),
    ("counts", None, True, "counts mismatch"),
    ("approval_source", "", False, "approval_source"),
    ("review_procedure", None, True, "review_procedure"),
    ("limitations", [], False, "limitations"),
])
def test_invalid_approval_record_fails_before_publication(
    approval_fixture, tmp_path, capsys, field, value, missing, message,
):
    record, path = approval_fixture
    if missing:
        del record[field]
    else:
        record[field] = value
    path.write_text(json.dumps(record), encoding="utf-8")
    output = tmp_path / "export.json"
    assert export.main([]) == 2
    assert export.main(["--workbook", str(tmp_path / "unread.xlsx"), "--output", str(output)]) == 2
    captured = capsys.readouterr()
    assert captured.err.count(message) == 2 and not captured.out
    assert not output.exists() and not list(tmp_path.glob(".aura-reviewed-*"))


@pytest.mark.parametrize(("mutation", "message"), [
    ("missing", "confirmed interpretations"),
    ("unexpected", "confirmed interpretations"),
    ("blank", "confirmed interpretations"),
    ("duplicate", "duplicate JSON key"),
])
def test_invalid_interpretation_inventory(approval_fixture, capsys, mutation, message):
    record, path = approval_fixture
    interpretations = record["confirmed_interpretations"]
    if mutation == "missing":
        interpretations.pop("cross-01")
    elif mutation == "unexpected":
        interpretations["unknown-01"] = interpretations.pop("cross-01")
    elif mutation == "blank":
        interpretations["cross-01"] = " "
    encoded = json.dumps(record)
    if mutation == "duplicate":
        encoded = encoded.replace('"cross-01":', '"cross-01": "duplicate", "cross-01":', 1)
    path.write_text(encoded, encoding="utf-8")
    destination = path.with_name("output.json")
    assert export.main(["--workbook", "unread.xlsx", "--output", str(destination)]) == 2
    assert message in capsys.readouterr().err
    assert not destination.exists() and not list(path.parent.glob(".aura-reviewed-*"))


@pytest.mark.parametrize("phase", ["write", "flush", "close"])
def test_failed_staging_never_publishes_and_removes_temp_files(tmp_path, monkeypatch, phase):
    destination = tmp_path / "final.json"
    original_open = Path.open

    class FailingStream:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.stream.close()
            if phase == "close":
                raise OSError("injected close failure")

        def write(self, payload):
            if phase == "write":
                self.stream.write(payload[:64])
                raise OSError("injected write failure")
            return self.stream.write(payload)

        def flush(self):
            if phase == "flush":
                raise OSError("injected flush failure")
            self.stream.flush()

    def open_staging(path, mode, *args, **kwargs):
        assert mode == "xb" and path.parent.parent == tmp_path and path != destination
        return FailingStream(original_open(path, mode, *args, **kwargs))

    with monkeypatch.context() as patches:
        patches.setattr(Path, "open", open_staging)
        with pytest.raises(OSError, match=f"injected {phase} failure"):
            export.publish_export(destination, b"x" * 100)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("failure", ["link", "existing", "collision"])
def test_no_overwrite_publication_and_staging_cleanup(tmp_path, monkeypatch, failure):
    destination = tmp_path / "final.json"
    operation = "rename" if export.os.name == "nt" else "link"
    original_publish = getattr(export.os, operation)
    payload = b'{"complete": true}\n'
    if failure == "existing":
        destination.write_bytes(b"existing bytes")

    def publish(staged, requested):
        assert staged.read_bytes() == payload and requested == destination
        if failure == "link":
            raise OSError("injected publication failure")
        if failure == "collision":
            destination.write_bytes(b"concurrent bytes")
        return original_publish(staged, requested)

    monkeypatch.setattr(export.os, operation, publish)
    expected = OSError if failure == "link" else FileExistsError
    with pytest.raises(expected):
        export.publish_export(destination, payload)
    if failure == "link":
        assert not list(tmp_path.iterdir())
    else:
        assert list(tmp_path.iterdir()) == [destination]
        assert destination.read_bytes() == (b"existing bytes" if failure == "existing" else b"concurrent bytes")


def test_publication_writes_exact_serialized_dataset_and_cleans_staging(reviewed_artifacts, tmp_path):
    dataset, _, _, _ = reviewed_artifacts
    destination = tmp_path / "final.json"
    export.publish_export(destination, export.export_bytes(dataset))
    assert destination.read_bytes() == (
        export.CORPUS_DIR / export.DATASET_NAME
    ).read_text(encoding="ascii").encode("ascii")
    assert list(tmp_path.iterdir()) == [destination]
