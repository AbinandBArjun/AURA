"""Logical-ID adapters for the existing production search orchestration."""

from contextlib import closing
from math import isfinite

from qdrant_client import models
from sqlalchemy.orm import Session

from app.retrieval.bm25_index_cache import BM25IndexCache
from app.retrieval.bm25_retriever import BM25Retriever
from app.services.search_service import SearchService
from evaluation.portable_benchmark.seed import seed_transaction
from evaluation.portable_benchmark.seed_infra import COLLECTION, SeedError
from evaluation.portable_benchmark.seed_vectors import vector_digest


class LogicalMapping:
    """Join the verified readiness mapping to the canonical passage payloads."""

    def __init__(self, record, documents):
        catalog = {
            passage.passage_id: (document.key, passage.chunk)
            for document in documents for passage in document.passages
        }
        if len(catalog) != sum(len(document.passages) for document in documents):
            raise SeedError("Duplicate passage in the corpus catalog")
        self.workspace_id = record.workspace_id
        self.chunks = {}
        self.points = {}
        logical = set()
        document_keys = set()
        document_ids = set()
        for document in record.documents:
            if document.document_key in document_keys or document.document_id in document_ids:
                raise SeedError("Duplicate document mapping")
            document_keys.add(document.document_key)
            document_ids.add(document.document_id)
            for item in document.passages:
                if item.passage_id not in catalog:
                    raise SeedError("Unknown passage in the readiness mapping")
                key, chunk = catalog[item.passage_id]
                if (
                    key != document.document_key or item.passage_id in logical
                    or item.chunk_id in self.chunks
                    or chunk.retrievable != (item.vector_id is not None)
                ):
                    raise SeedError("Duplicate or inconsistent readiness mapping")
                logical.add(item.passage_id)
                payload = {
                    "passage_id": item.passage_id, "document_key": key, "workspace_id": self.workspace_id,
                    "document_id": document.document_id, "chunk_id": item.chunk_id, **vars(chunk),
                }
                self.chunks[item.chunk_id] = payload
                if item.vector_id is not None:
                    if item.vector_id in self.points:
                        raise SeedError("Duplicate Qdrant point mapping")
                    self.points[item.vector_id] = payload
        if set(catalog) != logical or not self.points:
            raise SeedError("Incomplete readiness mapping")

    @staticmethod
    def entry(row, expected):
        if any(type(row.get(key)) is not type(value) or row[key] != value for key, value in expected.items()):
            raise SeedError("Retrieved passage does not match its verified mapping")
        score = row.get("score")
        if type(score) is not float or not isfinite(score):
            raise SeedError("Retriever returned an invalid score")
        return {**row, "chunk_id": expected["passage_id"]}

    def dense(self, points):
        seen = set()
        entries = []
        for point in points:
            point_id = str(point.id)
            if point_id not in self.points or point_id in seen:
                raise SeedError("Unknown or duplicate Qdrant result identity")
            seen.add(point_id)
            payload = point.payload or {}
            expected = self.points[point_id]
            if payload.keys() != expected.keys():
                raise SeedError("Unexpected Qdrant payload fields")
            entries.append(self.entry({**payload, "score": point.score}, expected))
        if seen != self.points.keys():
            raise SeedError("Dense retrieval did not return every verified passage")
        return entries

    def sparse(self, rows):
        seen = set()
        fields = ("chunk_id", "document_id", "text", "page_start", "page_end", "section")
        entries = []
        for row in rows:
            chunk_id = row.get("chunk_id")
            if type(chunk_id) is not int or chunk_id in seen or chunk_id not in self.chunks:
                raise SeedError("Unknown or duplicate SQL result identity")
            seen.add(chunk_id)
            payload = self.chunks[chunk_id]
            if not payload["retrievable"]:
                raise SeedError("BM25 returned a non-retrievable passage")
            expected = {key: payload[key] for key in fields} | {"passage_id": payload["passage_id"]}
            entries.append(self.entry({"passage_id": payload["passage_id"], **row}, expected))
        return entries


def logical_cutoff(rows, limit, score="score"):
    return sorted(rows, key=lambda row: (-row[score], row["chunk_id"]))[:limit]


class DenseAdapter:
    def __init__(self, client, embedding, mapping):
        self.client = client
        self.embedding = embedding
        self.mapping = mapping

    def search(self, query, workspace_id, limit):
        if workspace_id != self.mapping.workspace_id:
            raise SeedError("Unexpected benchmark workspace")
        vector = self.embedding.embed_text(query)
        vector_digest(vector)
        result = self.client.query_points(
            collection_name=COLLECTION,
            query=vector,
            query_filter=models.Filter(must=[
                models.FieldCondition(key="workspace_id", match=models.MatchValue(value=workspace_id)),
                models.FieldCondition(key="retrievable", match=models.MatchValue(value=True)),
            ]),
            search_params=models.SearchParams(exact=True),
            limit=len(self.mapping.points),
            with_payload=True,
        )
        return logical_cutoff(self.mapping.dense(result.points), limit)


class SparseAdapter:
    def __init__(self, mapping):
        self.mapping = mapping
        self.retriever = BM25Retriever(index_cache=BM25IndexCache())

    def search(self, db, workspace_id, query, limit):
        if workspace_id != self.mapping.workspace_id:
            raise SeedError("Unexpected benchmark workspace")
        rows = self.retriever.search(db, workspace_id, query, limit=len(self.mapping.points))
        return logical_cutoff(self.mapping.sparse(rows), limit)


class RerankAdapter:
    def __init__(self, reranker):
        self.reranker = reranker

    def rerank(self, query, results, limit):
        if not results:
            return []
        rows = self.reranker.rerank(query, results, limit=len(results))
        return logical_cutoff(rows, limit, "rerank_score")


def search_service(client, embedding, reranker, mapping):
    # Production RRF consumes chunk_id as its deduplication and tie-break key.
    # These adapters supply logical IDs while leaving SQL document IDs intact.
    return SearchService(
        qdrant_service=DenseAdapter(client, embedding, mapping),
        bm25_retriever=SparseAdapter(mapping),
        reranker=RerankAdapter(reranker),
    )


def run_against_seed(targets, corpus, embedding_manifest, embedding_factory, reranker_factory, evaluate):
    """Run and revalidate under a read-only SQL snapshot and the seed lock."""
    engine = targets.engine()
    try:
        with closing(targets.qdrant()) as client, engine.begin() as connection:
            connection.exec_driver_sql("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            record = seed_transaction(connection, client, targets, corpus, embedding_manifest, require_ready=True)
            mapping = LogicalMapping(record, corpus.documents)
            service = search_service(client, embedding_factory(), reranker_factory(), mapping)
            with Session(bind=connection) as session:
                result = evaluate(service, session, record.workspace_id, mapping)
            final = seed_transaction(connection, client, targets, corpus, embedding_manifest, require_ready=True)
            if final != record:
                raise SeedError("Benchmark readiness changed during evaluation")
        return result
    finally:
        engine.dispose()
