import json
from pathlib import Path

from app.db.session import SessionLocal
from app.models import Document, DocumentChunk

DATASET_PATH = (
    Path(__file__).parent
    / "datasets"
    / "retrieval_benchmark.json"
)


def load_dataset():
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset.get("queries"), list):
        raise ValueError("Dataset must contain a queries list.")

    return dataset


def validate_dataset():
    dataset = load_dataset()
    queries = dataset["queries"]

    workspace_id = dataset.get("workspace_id")
    if not isinstance(workspace_id, int):
        raise ValueError("Dataset must define an integer workspace_id.")

    query_ids = [item["query_id"] for item in queries]

    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Duplicate query IDs detected.")

    expected_ids = {
        int(chunk_id)
        for item in queries
        for chunk_id in item["relevant_chunks"]
    }

    with SessionLocal() as db:
        document = (
            db.query(Document)
            .filter(Document.workspace_id == workspace_id)
            .first()
        )

        if document is None:
            raise ValueError(
                f"No documents found in workspace {workspace_id}."
            )

        chunks = (
            db.query(DocumentChunk)
            .join(Document)
            .filter(
                Document.workspace_id == workspace_id,
                DocumentChunk.id.in_(expected_ids),
            )
            .all()
        )

        chunk_map = {chunk.id: chunk for chunk in chunks}

    missing_ids = expected_ids - set(chunk_map)

    if missing_ids:
        raise ValueError(
            f"Missing or out-of-workspace chunks: {sorted(missing_ids)}"
        )

    non_retrievable = [
        chunk_id
        for chunk_id, chunk in chunk_map.items()
        if not chunk.retrievable
    ]

    if non_retrievable:
        raise ValueError(
            f"Non-retrievable chunks labelled: {sorted(non_retrievable)}"
        )

    for item in queries:
        query_id = item.get("query_id", "<unknown>")

        if not item.get("query", "").strip():
            raise ValueError(f"Empty query: {query_id}")

        if not item.get("relevant_chunks"):
            raise ValueError(f"No relevance labels: {query_id}")

        for chunk_id, grade in item["relevant_chunks"].items():
            if not chunk_id.isdigit():
                raise ValueError(
                    f"Invalid chunk ID in {query_id}: {chunk_id}"
                )

            if not isinstance(grade, int) or grade not in (0, 1, 2, 3):
                raise ValueError(
                    f"Invalid relevance grade in {query_id}: {grade}"
                )

    print(f"Dataset: {dataset['dataset_name']}")
    print(f"Version: {dataset['dataset_version']}")
    print(f"Workspace: {workspace_id}")
    print(f"Queries: {len(queries)}")
    print(f"Unique labelled chunks: {len(expected_ids)}")
    print("All labelled chunks belong to the workspace and are retrievable.")
    print("Dataset validation successful.")


if __name__ == "__main__":
    validate_dataset()
