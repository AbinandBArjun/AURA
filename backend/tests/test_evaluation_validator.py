import copy

import pytest

from evaluation.validate_dataset import validate_dataset_structure


@pytest.fixture
def valid_dataset():
    return {
        "dataset_name": "Test Retrieval Dataset",
        "dataset_version": "1.0",
        "workspace_id": 1,
        "queries": [
            {
                "query_id": "query_001",
                "query": "What is retrieval augmented generation?",
                "relevant_chunks": {
                    "101": 3,
                    "102": 2,
                    "103": 0,
                },
            }
        ],
    }


def test_validate_dataset_structure_returns_expected_values(valid_dataset):
    workspace_id, queries, expected_ids = (
        validate_dataset_structure(valid_dataset)
    )

    assert workspace_id == 1
    assert len(queries) == 1
    assert expected_ids == {101, 102, 103}


@pytest.mark.parametrize("workspace_id", [True, False, 0, -1, "1", None])
def test_validate_dataset_structure_rejects_invalid_workspace_id(
    valid_dataset,
    workspace_id,
):
    dataset = copy.deepcopy(valid_dataset)
    dataset["workspace_id"] = workspace_id

    with pytest.raises(ValueError, match="workspace_id"):
        validate_dataset_structure(dataset)


@pytest.mark.parametrize("chunk_id", ["abc", "0", "-1", "1.5", ""])
def test_validate_dataset_structure_rejects_invalid_chunk_ids(
    valid_dataset,
    chunk_id,
):
    dataset = copy.deepcopy(valid_dataset)
    dataset["queries"][0]["relevant_chunks"] = {chunk_id: 3}

    with pytest.raises(ValueError, match="Invalid chunk ID"):
        validate_dataset_structure(dataset)


@pytest.mark.parametrize("grade", [True, False, -1, 4, 1.5, "3", None])
def test_validate_dataset_structure_rejects_invalid_relevance_grades(
    valid_dataset,
    grade,
):
    dataset = copy.deepcopy(valid_dataset)
    dataset["queries"][0]["relevant_chunks"] = {"101": grade}

    with pytest.raises(ValueError, match="Invalid relevance grade"):
        validate_dataset_structure(dataset)


def test_validate_dataset_structure_rejects_duplicate_query_ids(
    valid_dataset,
):
    dataset = copy.deepcopy(valid_dataset)
    dataset["queries"].append(copy.deepcopy(dataset["queries"][0]))

    with pytest.raises(ValueError, match="Duplicate query IDs"):
        validate_dataset_structure(dataset)


@pytest.mark.parametrize("query", [None, "", "   ", 123])
def test_validate_dataset_structure_rejects_invalid_query_text(
    valid_dataset,
    query,
):
    dataset = copy.deepcopy(valid_dataset)
    dataset["queries"][0]["query"] = query

    with pytest.raises(ValueError, match="Empty or invalid query"):
        validate_dataset_structure(dataset)


@pytest.mark.parametrize("query_id", [None, "", "   ", 123])
def test_validate_dataset_structure_rejects_invalid_query_ids(
    valid_dataset,
    query_id,
):
    dataset = copy.deepcopy(valid_dataset)
    dataset["queries"][0]["query_id"] = query_id

    with pytest.raises(ValueError, match="query_id"):
        validate_dataset_structure(dataset)


def test_validate_dataset_structure_rejects_non_object_query(
    valid_dataset,
):
    dataset = copy.deepcopy(valid_dataset)
    dataset["queries"] = ["invalid"]

    with pytest.raises(ValueError, match="must be an object"):
        validate_dataset_structure(dataset)


def test_validate_dataset_structure_rejects_empty_queries(
    valid_dataset,
):
    dataset = copy.deepcopy(valid_dataset)
    dataset["queries"] = []

    with pytest.raises(ValueError, match="non-empty queries list"):
        validate_dataset_structure(dataset)


def test_validate_dataset_structure_rejects_empty_relevance_mapping(
    valid_dataset,
):
    dataset = copy.deepcopy(valid_dataset)
    dataset["queries"][0]["relevant_chunks"] = {}

    with pytest.raises(ValueError, match="relevance mapping"):
        validate_dataset_structure(dataset)


def test_validate_dataset_structure_requires_positive_relevance(
    valid_dataset,
):
    dataset = copy.deepcopy(valid_dataset)
    dataset["queries"][0]["relevant_chunks"] = {
        "101": 0,
        "102": 0,
    }

    with pytest.raises(ValueError, match="no positively relevant chunks"):
        validate_dataset_structure(dataset)


@pytest.mark.parametrize("dataset", [None, [], "invalid"])
def test_validate_dataset_structure_rejects_non_object_dataset(dataset):
    with pytest.raises(ValueError, match="JSON object"):
        validate_dataset_structure(dataset)


def test_validate_dataset_structure_rejects_missing_metadata(valid_dataset):
    dataset = copy.deepcopy(valid_dataset)
    del dataset["dataset_name"]

    with pytest.raises(ValueError, match="dataset_name"):
        validate_dataset_structure(dataset)
