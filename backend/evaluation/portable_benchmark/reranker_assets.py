"""Prepare the immutable cross-encoder snapshot used only by portable evaluation."""

import argparse
from pathlib import Path
import re
from runpy import run_path
import sys
from tempfile import TemporaryDirectory

from huggingface_hub import snapshot_download
from sentence_transformers import CrossEncoder

from app.retrieval.cross_encoder_reranker import CrossEncoderReranker
from evaluation.portable_benchmark.corpus import read_json
from evaluation.portable_benchmark.seed_assets import TEST_MODEL_PREPARATION


MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MODEL_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
MANIFEST_PATH = Path(__file__).with_name("reranker_manifest.json")
REQUIRED_FILES = {
    "config.json", "model.safetensors", "special_tokens_map.json",
    "tokenizer.json", "tokenizer_config.json", "vocab.txt",
}


def load_manifest():
    manifest = read_json(MANIFEST_PATH)
    if (
        not isinstance(manifest, dict) or set(manifest) != {"schema_version", "repo_id", "revision", "files"}
        or type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1
        or manifest["repo_id"] != MODEL_ID or manifest["revision"] != MODEL_REVISION
    ):
        raise ValueError("Incompatible cross-encoder manifest or model revision")
    files = manifest["files"]
    if not isinstance(files, dict) or set(files) != REQUIRED_FILES or any(
        not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
        for digest in files.values()
    ):
        raise ValueError("Invalid cross-encoder inference file manifest")
    return manifest


def verified_reranker(directory):
    manifest = load_manifest()
    validate_files = run_path(str(TEST_MODEL_PREPARATION))["validate_files"]
    validate_files(directory, manifest["files"])
    return directory.resolve(strict=True), manifest


def prepare_reranker(directory):
    if directory.exists() or directory.is_symlink() or directory.is_junction():
        verified_reranker(directory)
        return
    manifest = load_manifest()
    directory = directory.absolute()
    if directory.parent.resolve(strict=True) != directory.parent or not directory.parent.is_dir():
        raise ValueError("Use an existing trusted local parent directory for model preparation")
    with TemporaryDirectory(prefix=".aura-reranker-", dir=directory.parent) as temporary:
        workspace = Path(temporary)
        staged = workspace / "model"
        snapshot_download(
            repo_id=MODEL_ID, revision=MODEL_REVISION, token=False,
            allow_patterns=sorted(manifest["files"]), local_dir=staged,
        )
        metadata = staged / ".cache"
        if metadata.exists() or metadata.is_symlink() or metadata.is_junction():
            metadata.rename(workspace / "download-metadata")
        verified_reranker(staged)
        # Reserve the directory exclusively, then publish only verified assets.
        # If publishing is interrupted, future validation rejects the partial directory.
        directory.mkdir()
        for name in sorted(manifest["files"]):
            (staged / name).rename(directory / name)
        verified_reranker(directory)


def local_reranker(path):
    model = CrossEncoder(
        str(path), device="cpu", max_length=512, local_files_only=True, trust_remote_code=False,
        token=False, model_kwargs={"use_safetensors": True},
    )
    return CrossEncoderReranker(model_name=MODEL_ID, model=model)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        prepare_reranker(args.model_dir)
    except (OSError, ValueError) as error:
        print(f"Reranker preparation failed: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Reranker download failed ({type(error).__name__}). Check the public pinned snapshot.", file=sys.stderr)
        return 1
    print(f"Verified cross-encoder assets for {MODEL_ID}@{MODEL_REVISION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
