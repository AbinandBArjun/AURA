"""Write an owned collection once and verify all of its points before publishing readiness."""

from hashlib import sha256
from math import isclose, isfinite
from struct import pack

from qdrant_client import models

from evaluation.portable_benchmark.seed_infra import COLLECTION, SeedError


DIMENSIONS = 384
PAYLOAD_INDEXES = {
    "workspace_id": models.PayloadSchemaType.INTEGER,
    "document_id": models.PayloadSchemaType.INTEGER,
    "retrievable": models.PayloadSchemaType.BOOL,
}


def vector_digest(vector):
    if (
        not isinstance(vector, list) or len(vector) != DIMENSIONS
        or any(type(value) not in (int, float) or not isfinite(value) for value in vector)
        or not isclose(sum(value * value for value in vector), 1.0, abs_tol=1e-4)
    ):
        raise SeedError("Found an invalid or unnormalized 384-dimensional embedding")
    return sha256(pack(f"<{DIMENSIONS}f", *vector)).hexdigest()


def point_digests(client, expected, originals=None):
    info = client.get_collection(COLLECTION)
    vectors = info.config.params.vectors
    if not isinstance(vectors, models.VectorParams) or vectors.size != DIMENSIONS or vectors.distance != models.Distance.COSINE:
        raise SeedError("The benchmark collection must use 384-dimensional cosine vectors")
    if {key: value.data_type for key, value in info.payload_schema.items()} != PAYLOAD_INDEXES:
        raise SeedError("The benchmark collection has unexpected payload indexes")
    points, offset = client.scroll(COLLECTION, limit=len(expected) + 1, with_vectors=True, with_payload=True)
    results = {}
    for point in points:
        key = str(point.id)
        if key in results or key not in expected:
            raise SeedError("Found a duplicate or unexpected point in the benchmark collection")
        payload = point.payload
        if payload is None or payload.keys() != expected[key].keys() or any(
            type(payload[field]) is not type(value) or payload[field] != value
            for field, value in expected[key].items()
        ):
            raise SeedError("Found a stale benchmark point payload that does not match PostgreSQL")
        results[key] = vector_digest(point.vector)
        if originals is not None and any(
            not isclose(current, original, rel_tol=1e-5, abs_tol=1e-6)
            for current, original in zip(point.vector, originals[key])
        ):
            raise SeedError("An indexed point differs from its generated MiniLM embedding")
    if offset is not None or results.keys() != expected.keys():
        raise SeedError("Missing or unexpected points in the benchmark collection")
    return results


def create_vectors(client, expected, embedding_service):
    vectors = embedding_service.embed_texts([payload["text"] for payload in expected.values()])
    if len(vectors) != len(expected):
        raise SeedError("MiniLM returned an unexpected number of vectors")
    originals = dict(zip(expected, vectors))
    points = []
    for key, vector in originals.items():
        vector_digest(vector)
        points.append(models.PointStruct(id=key, vector=vector, payload=expected[key]))

    if not client.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(size=DIMENSIONS, distance=models.Distance.COSINE),
    ):
        raise SeedError("Qdrant did not create a new benchmark collection")
    for name, kind in PAYLOAD_INDEXES.items():
        client.create_payload_index(COLLECTION, name, field_schema=kind, wait=True)
    client.upsert(COLLECTION, points, wait=True)
    return point_digests(client, expected, originals)
