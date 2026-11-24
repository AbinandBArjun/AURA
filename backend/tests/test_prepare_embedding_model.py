import hashlib
import json
from unittest.mock import Mock

import pytest

import prepare_embedding_model as model


ASSETS = {
    "1_Pooling/config.json": b'{"pooling": "mean"}',
    "model.safetensors": b"synthetic weights for hashing",
    "tokenizer.json": b'{"vocab": ["test"]}',
}


@pytest.fixture(autouse=True)
def synthetic_manifest(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({
            "repo_id": "example/model",
            "revision": "a" * 40,
            "files": {name: hashlib.sha256(content).hexdigest() for name, content in ASSETS.items()},
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(model, "MANIFEST_PATH", path)
    return path


def write_assets(directory, assets):
    for name, content in assets.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def test_valid_model_directory(tmp_path):
    target = tmp_path / "model"
    write_assets(target, ASSETS)

    assert model.validate_model(target) == target.resolve()


def test_model_is_missing_required_file(tmp_path):
    target = tmp_path / "model"
    write_assets(target, ASSETS)
    (target / "1_Pooling/config.json").unlink()

    with pytest.raises(ValueError, match=r"Missing or invalid model file: .*config.json"):
        model.validate_model(target)


@pytest.mark.parametrize(
    ("name", "replacement"),
    [
        ("model.safetensors", b"\x00" * len(ASSETS["model.safetensors"])),
        ("1_Pooling/config.json", b'{"pooling": "max"}'),
        ("tokenizer.json", b'{"vocab": ["changed"]}'),
        ("model.safetensors", ASSETS["model.safetensors"][:4]),
    ],
    ids=["zero-weights", "altered-config", "altered-tokenizer", "truncated-weights"],
)
def test_assets_with_incorrect_hash_fail(tmp_path, name, replacement):
    target = tmp_path / "model"
    write_assets(target, ASSETS)
    (target / name).write_bytes(replacement)

    with pytest.raises(ValueError, match="SHA-256 mismatch") as error:
        model.validate_model(target)

    assert str(target / name) in str(error.value)


def test_model_rejects_unexpected_files(tmp_path):
    target = tmp_path / "model"
    write_assets(target, {**ASSETS, "modeling_custom.py": b"pass"})

    with pytest.raises(ValueError, match=r"Unexpected file .*modeling_custom.py"):
        model.validate_model(target)


def test_manifest_rejects_paths_outside_model(tmp_path, synthetic_manifest):
    contents = json.loads(synthetic_manifest.read_text(encoding="utf-8"))
    contents["files"]["../outside.txt"] = "a" * 64
    synthetic_manifest.write_text(json.dumps(contents), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsafe model manifest path"):
        model.validate_model(tmp_path)


def test_incomplete_download_is_not_published(tmp_path, monkeypatch):
    target = tmp_path / "model"

    def incomplete_download(*, local_dir, **options):
        write_assets(local_dir, {"tokenizer.json": ASSETS["tokenizer.json"]})

    monkeypatch.setattr(model, "snapshot_download", incomplete_download)

    with pytest.raises(ValueError, match="Missing or invalid model directory"):
        model.prepare_model(target)

    assert not target.exists()
    assert not list(tmp_path.glob(".aura-model-*"))


def test_interrupted_download_leaves_other_directories_alone(tmp_path, monkeypatch):
    target = tmp_path / "model"
    sibling = tmp_path / "other-model"
    write_assets(sibling, {"keep.txt": b"unchanged"})

    def interrupted_download(*, local_dir, **options):
        write_assets(local_dir, {"tokenizer.json": ASSETS["tokenizer.json"]})
        raise OSError("Hub unavailable")

    monkeypatch.setattr(model, "snapshot_download", interrupted_download)

    with pytest.raises(ValueError, match="Could not download .*Hub unavailable"):
        model.prepare_model(target)

    assert not target.exists()
    assert not list(tmp_path.glob(".aura-model-*"))
    assert (sibling / "keep.txt").read_bytes() == b"unchanged"


def test_incomplete_existing_model_fails_offline(tmp_path, monkeypatch, capsys):
    target = tmp_path / "model"
    write_assets(target, {"tokenizer.json": ASSETS["tokenizer.json"]})
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    download = Mock(side_effect=OSError("Hub unavailable"))
    monkeypatch.setattr(model, "snapshot_download", download)

    with pytest.raises(SystemExit) as failure:
        model.main([str(target)])

    assert failure.value.code == 2
    captured = capsys.readouterr()
    assert "Existing model is invalid" in captured.err
    assert "Embedding model ready" not in captured.out
    download.assert_not_called()


def test_valid_cache_does_not_access_hub(tmp_path, monkeypatch):
    target = tmp_path / "model"
    write_assets(target, ASSETS)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    download = Mock(side_effect=AssertionError("Unexpected download"))
    monkeypatch.setattr(model, "snapshot_download", download)

    model.prepare_model(target)

    download.assert_not_called()


def test_invalid_cache_requires_explicit_recovery(tmp_path, monkeypatch):
    target = tmp_path / "model"
    invalid = b"unexpected weights"
    write_assets(target, {**ASSETS, "model.safetensors": invalid})
    download = Mock(side_effect=OSError("Recovery unavailable"))
    monkeypatch.setattr(model, "snapshot_download", download)

    with pytest.raises(ValueError, match="delete the matching GitHub Actions cache"):
        model.prepare_model(target)

    download.assert_not_called()
    assert (target / "model.safetensors").read_bytes() == invalid


def test_preparation_publishes_only_validated_assets_and_is_idempotent(tmp_path, monkeypatch):
    target = tmp_path / "model"

    def download_assets(*, local_dir, **options):
        assert not target.exists()
        assert options == {
            "repo_id": "example/model",
            "revision": "a" * 40,
            "allow_patterns": list(ASSETS),
            "token": False,
        }
        write_assets(local_dir, {**ASSETS, ".cache/huggingface/metadata": b"download metadata"})

    download = Mock(side_effect=download_assets)
    monkeypatch.setattr(model, "snapshot_download", download)

    model.prepare_model(target)
    model.prepare_model(target)

    download.assert_called_once()
    assert model.validate_model(target) == target.resolve()
    assert not (target / ".cache").exists()
    assert not list(tmp_path.glob(".aura-model-*"))
