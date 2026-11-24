import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from tempfile import TemporaryDirectory

from huggingface_hub import snapshot_download


MANIFEST_PATH = Path(__file__).with_name("embedding_model_manifest.json")


def load_manifest():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if not re.fullmatch(r"[0-9a-f]{40}", manifest["revision"]):
        raise ValueError("The model manifest must pin a full upstream commit SHA")

    files = manifest["files"]
    if not files:
        raise ValueError("The model manifest has no required files")
    for name, checksum in files.items():
        path = PurePosixPath(name)
        if (
            not path.parts
            or path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != name
            or "\\" in name
            or ":" in name
            or path.parts[0] == ".cache"
        ):
            raise ValueError(f"Unsafe model manifest path: {name!r}")
        if not re.fullmatch(r"[0-9a-f]{64}", checksum):
            raise ValueError(f"Invalid SHA-256 in model manifest for {name!r}")
    return manifest


def is_link(path):
    return path.is_symlink() or getattr(path, "is_junction", lambda: False)()


def require_directory(path):
    if is_link(path) or not path.is_dir():
        raise ValueError(f"Missing or invalid model directory: {path}")


def validate_model(directory):
    files = load_manifest()["files"]
    validate_files(directory, files)
    return directory.resolve(strict=True)


def validate_files(directory, files):
    folders = {directory}
    for name in files:
        folders.update(directory / parent for parent in PurePosixPath(name).parents)

    allowed = set(files)
    allowed.update(folder.relative_to(directory).as_posix() for folder in folders)
    for folder in sorted(folders, key=lambda path: len(path.parts)):
        require_directory(folder)
        for entry in folder.iterdir():
            if entry.relative_to(directory).as_posix() not in allowed:
                raise ValueError(f"Unexpected file or directory in prepared model: {entry}")

    for name, expected in files.items():
        path = directory / name
        if is_link(path) or not path.is_file():
            raise ValueError(f"Missing or invalid model file: {path}")
        with path.open("rb") as file:
            actual = hashlib.file_digest(file, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {path} (expected {expected}, got {actual})")


def prepare_model(directory):
    manifest = load_manifest()
    files = manifest["files"]

    if directory.exists() or is_link(directory):
        try:
            validate_files(directory, files)
        except (OSError, ValueError) as error:
            raise ValueError(
                f"Existing model is invalid: {error}. Use a new model directory locally "
                "or delete the matching GitHub Actions cache and rerun."
            ) from error
        return

    directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".aura-model-", dir=directory.parent) as temporary:
        workspace = Path(temporary)
        staged = workspace / "model"
        repo_id, revision = manifest["repo_id"], manifest["revision"]
        print(f"Downloading {repo_id}@{revision}", flush=True)
        try:
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                local_dir=staged,
                allow_patterns=list(files),
                token=False,
            )
        except Exception as error:
            raise ValueError(f"Could not download {repo_id}@{revision}: {error}") from error

        require_directory(staged)
        metadata = staged / ".cache"
        if metadata.exists() or is_link(metadata):
            metadata.rename(workspace / "hub-metadata")
        validate_files(staged, files)
        if directory.exists() or is_link(directory):
            raise ValueError(f"Model directory appeared during preparation: {directory}")
        staged.rename(directory)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Prepare or validate the pinned MiniLM model for integration tests."
    )
    parser.add_argument("model_directory", type=Path)
    args = parser.parse_args(argv)

    try:
        prepare_model(args.model_directory)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Embedding model ready at {args.model_directory}")


if __name__ == "__main__":
    main()
