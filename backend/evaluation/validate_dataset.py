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

    return dataset


def validate_dataset_structure(dataset):
    if not isinstance(dataset, dict):
        raise ValueError("Dataset must be a JSON object.")

    for field in ("dataset_name", "dataset_version"):
        value = dataset.get(field)

        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Dataset must define a non-empty {field}."
            )

    workspace_id = dataset.get("workspace_id")

    if (
        not isinstance(workspace_id, int)
        or isinstance(workspace_id, bool)
        or workspace_id <= 0
    ):
        raise ValueError(
            "Dataset must define a positive integer workspace_id."
        )

    queries = dataset.get("queries")

    if not isinstance(queries, list) or not queries:
        raise ValueError(
            "Dataset must contain a non-empty queries list."
        )

    query_ids = []
    expected_ids = set()

    for index, item in enumerate(queries):
        if not isinstance(item, dict):
            raise ValueError(
                f"Query at index {index} must be an object."
            )

        query_id = item.get("query_id")

        if not isinstance(query_id, str) or not query_id.strip():
            raise ValueError(
                f"Query at index {index} must have a non-empty string query_id."
            )

        query_ids.append(query_id)

        query = item.get("query")

        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Empty or invalid query: {query_id}")

        relevant_chunks = item.get("relevant_chunks")

        if not isinstance(relevant_chunks, dict) or not relevant_chunks:
            raise ValueError(
                f"No valid relevance mapping: {query_id}"
            )

        has_relevant_chunk = False

        for chunk_id, grade in relevant_chunks.items():
            if (
                not isinstance(chunk_id, str)
                or not chunk_id.isdecimal()
                or int(chunk_id) <= 0
            ):
                raise ValueError(
                    f"Invalid chunk ID in {query_id}: {chunk_id}"
                )

            if (
                not isinstance(grade, int)
                or isinstance(grade, bool)
                or grade not in (0, 1, 2, 3)
            ):
                raise ValueError(
                    f"Invalid relevance grade in {query_id}: {grade}"
                )

            if grade > 0:
                has_relevant_chunk = True

            expected_ids.add(int(chunk_id))

        if not has_relevant_chunk:
            raise ValueError(
                f"Query has no positively relevant chunks: {query_id}"
            )

    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Duplicate query IDs detected.")

    return workspace_id, queries, expected_ids


def validate_dataset():
    dataset = load_dataset()
    workspace_id, queries, expected_ids = (
        validate_dataset_structure(dataset)
    )

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

    print(f"Dataset: {dataset['dataset_name']}")
    print(f"Version: {dataset['dataset_version']}")
    print(f"Workspace: {workspace_id}")
    print(f"Queries: {len(queries)}")
    print(f"Unique labelled chunks: {len(expected_ids)}")
    print("All labelled chunks belong to the workspace and are retrievable.")
    print("Dataset validation successful.")


if __name__ == "__main__":
    validate_dataset()
