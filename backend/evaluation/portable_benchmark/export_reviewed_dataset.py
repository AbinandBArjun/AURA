import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from xml.etree import ElementTree
from zipfile import ZipFile

from evaluation.portable_benchmark.corpus import (
    CORPUS_DIR, json_digest, load_corpus, load_dataset, passage_catalog, protocol_fingerprint, read_json,
)
from evaluation.portable_benchmark.schema import PortableDataset


DATASET_NAME = "dataset.reviewed.json"
RECORD_NAME = "review_approval.json"
WORKBOOK_SHA256 = "89843226a3353cf87a11e7eaa48c3643781243609abc160a27470adebb277d60"
CORPUS_SHA256 = "e4f0f2fba8a9c2b598a260269891cde0b54d9566b0c0a68beb74443a7d700300"
DRAFT_SHA256 = "cc6d83fe50eb3a785b74295651f0c7ee48d384bc70a573a02a64478128bbd756"
GRADE_COUNTS = {0: 1287, 1: 70, 2: 52, 3: 31}
APPROVAL_STATEMENT = (
    "I formally approve all 1,440 reviewed relevance judgments and all 41 flagged resolutions for AURA's "
    "portable retrieval benchmark. Human reviewer ID: aura-project-owner. Approval date: 2026-11-27. "
    "Assessment method: AI-assisted, human-verified. Proceed with Checkpoint 17.4.5B dataset export and validation "
    "using Codex."
)
INTERPRETATION_IDS = {
    "cross-01", "cross-02", "cross-03", "cross-04", "ai-03", "ai-04", "ai-07",
    "cy-01", "cy-04", "econ-05", "sw-02", "sw-05",
}
NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
# These sheet numbers and the cell encoding are specific to the hash-pinned workbook.
SHEETS = {"Guide": 1, "Assessments": 2, "Queries": 3, "Passages": 4, "AI Flags": 5}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def workbook_rows(path):
    data = path.read_bytes()
    require(sha256(data).hexdigest() == WORKBOOK_SHA256, "Source workbook SHA-256 mismatch")
    with ZipFile(BytesIO(data)) as archive:
        tables = {}
        for name, number in SHEETS.items():
            sheet = ElementTree.fromstring(archive.read(f"xl/worksheets/sheet{number}.xml"))
            rows = []
            for row in sheet.findall("x:sheetData/x:row", NS):
                cells = {}
                for cell in row.findall("x:c", NS):
                    value = cell.findtext("x:v", namespaces=NS)
                    if value is not None:
                        kind = cell.get("t", "n")
                        require(kind in ("str", "n") and cell.find("x:f", NS) is None,
                                "Unexpected workbook cell encoding")
                        column = cell.get("r").rstrip("0123456789")
                        cells[column] = value if kind == "str" else int(value)
                rows.append(tuple(cells.get(column) for column in "ABCDEFGHIJKLMNO"))
            tables[name] = rows[1:]
    return tables


def canonical_inputs(root=CORPUS_DIR):
    manifest, sources = load_corpus(root)
    catalog = passage_catalog(manifest, sources)
    draft = load_dataset(root, manifest, catalog)
    eligible = [passage for passage in catalog if passage.chunk.retrievable]
    require(len(sources) == 8 and len(eligible) == 40 and len(draft.queries) == 36,
            "Canonical corpus or query inventory size mismatch")
    require(manifest.corpus_fingerprint == CORPUS_SHA256
            and json_digest(draft.model_dump()) == DRAFT_SHA256,
            "Canonical corpus or draft dataset fingerprint mismatch")
    committed = read_json(root / "passage_ids.json")
    require(committed["corpus_fingerprint"] == CORPUS_SHA256
            and committed["passage_ids"] == [passage.passage_id for passage in eligible],
            "Canonical passage IDs differ from the frozen catalog")
    return manifest, eligible, draft


def reconcile(tables, manifest, catalog, draft, assessor_id):
    guide = {row[0]: row[1] for row in tables["Guide"]}
    require(guide["Corpus fingerprint"] == CORPUS_SHA256 and guide["Draft dataset SHA-256"] == DRAFT_SHA256,
            "Workbook corpus or draft fingerprint mismatch")
    queries = {query.query_id: (query.category, query.query) for query in draft.queries}
    query_rows = tables["Queries"]
    require(len(query_rows) == len(queries)
            and {row[0]: row[1:3] for row in query_rows} == queries,
            "Workbook query IDs, texts or categories differ from the draft")

    documents = {document.key: document for document in manifest.documents}
    passages = {}
    protocol = protocol_fingerprint(manifest)
    for passage in catalog:
        _, key, ordinal, embedded_protocol, _ = passage.passage_id.split(":")
        require(embedded_protocol == protocol, "Canonical passage protocol mismatch")
        chunk, document = passage.chunk, documents[key]
        passages[f"{key}/{ordinal}"] = (
            key, document.domain, chunk.section, f"{chunk.page_start}\u2013{chunk.page_end}",
            passage.passage_id, chunk.text, document.title,
        )
    passage_rows = tables["Passages"]
    require(len(passage_rows) == len(passages)
            and {row[0]: row[1:8] for row in passage_rows} == passages,
            "Workbook passage text, IDs or document metadata differ from the canonical catalog")

    grades = {}
    for row in tables["Assessments"]:
        query, category, ref, key, passage_id, grade, assessor, rationale = row[:8]
        require(query in queries and category == queries[query][0]
                and ref in passages and (key, passage_id) == (passages[ref][0], passages[ref][4]),
                "Workbook assessment query or passage reference mismatch")
        pair = (query, passage_id)
        require(pair not in grades and type(grade) is int and grade in GRADE_COUNTS,
                "Workbook has a duplicate pair or invalid grade")
        require(assessor == assessor_id and isinstance(rationale, str) and bool(rationale.strip())
                and all(value is None for value in row[8:]),
                "Workbook assessor, AI rationale or blank human review fields differ from the approved draft")
        grades[pair] = grade
    require(len(grades) == len(queries) * len(passages) and Counter(grades.values()) == GRADE_COUNTS,
            "Workbook assessment coverage or grade distribution mismatch")

    flags = []
    seen = set()
    for query, ref, proposed, alternative, reason, *_ in tables["AI Flags"]:
        pair = (query, passages[ref][4]) if ref in passages else None
        require(pair in grades and pair not in seen
                and type(proposed) is int and proposed == grades[pair]
                and type(alternative) is int and alternative in GRADE_COUNTS and alternative != proposed
                and isinstance(reason, str) and bool(reason.strip()),
                "Workbook flag disagrees with the assessment or contains an invalid alternative")
        seen.add(pair)
        flags.append({
            "query_id": query, "passage_ref": ref, "passage_id": pair[1],
            "ai_proposed_grade": proposed, "ai_alternative_grade": alternative,
            "ai_flag_reason": reason, "reviewed_grade": proposed, "resolution": "retained-ai-proposed-grade",
        })
    require(len(flags) == 41, "Workbook flagged alternative count mismatch")
    return grades, flags


def provenance(record):
    return {
        "method": record["assessment_method"],
        "assessor_id": record["ai_assessor_id"],
        "review_status": "reviewed",
        "reviewer_id": record["reviewer_id"],
        "ai_provenance": record["ai_provenance"],
        "human_review": {
            "reviewer_kind": "human", "decision": record["decision"],
            "approved_on": record["approved_on"], "approval_reference": RECORD_NAME,
        },
    }


def build_dataset(catalog, draft, grades, record):
    expected = {(query.query_id, passage.passage_id) for query in draft.queries for passage in catalog}
    require(grades.keys() == expected, "Judgment pairs do not exactly cover the canonical inventories")
    fields = provenance(record)
    return PortableDataset.model_validate({
        "schema_version": 2,
        "corpus_fingerprint": draft.corpus_fingerprint,
        "review_status": "reviewed",
        "queries": [
            {
                **query.model_dump(),
                "judgments": [
                    {"passage_id": passage.passage_id, "grade": grades[query.query_id, passage.passage_id], **fields}
                    for passage in catalog
                ],
            }
            for query in draft.queries
        ],
    })


def nonempty_text(value):
    return isinstance(value, str) and bool(value.strip())


def validate_approval_record(record):
    require(isinstance(record, dict) and type(record.get("schema_version")) is int
            and record["schema_version"] == 1, "Expected approval record schema v1")
    expected = {
        "approval_statement": APPROVAL_STATEMENT,
        "reviewer_id": "aura-project-owner",
        "ai_assessor_id": "openai-codex-ai-assisted-draft",
        "approved_on": "2026-11-27",
        "decision": "approved",
        "assessment_method": "ai-assisted",
        "ai_provenance": {"provider": "OpenAI", "tool": "Codex"},
        "source_workbook": {"name": "aura-relevance-assessment-draft.xlsx", "sha256": WORKBOOK_SHA256},
        "corpus_fingerprint": CORPUS_SHA256,
    }
    for key, value in expected.items():
        require(record.get(key) == value, f"Approval record has missing or incorrect {key}")
    interpretations = record.get("confirmed_interpretations")
    require(isinstance(interpretations, dict) and interpretations.keys() == INTERPRETATION_IDS
            and all(nonempty_text(value) for value in interpretations.values()),
            "Approval record must contain all 12 confirmed interpretations with nonempty text")
    for key in ("approval_source", "review_procedure"):
        require(nonempty_text(record.get(key)), f"Approval record requires nonempty {key}")
    limitations = record.get("limitations")
    require(isinstance(limitations, list) and limitations and all(nonempty_text(value) for value in limitations),
            "Approval record requires nonempty limitations")
    require(isinstance(record.get("flagged_resolutions"), list),
            "Approval record requires flagged resolutions")


def validate_artifacts(dataset, catalog, draft, record):
    validate_approval_record(record)
    require(dataset.schema_version == 2 and dataset.review_status == "reviewed",
            "Expected a reviewed schema v2 dataset")
    grades = {
        (query.query_id, judgment.passage_id): judgment.grade
        for query in dataset.queries for judgment in query.judgments
    }
    require(dataset == build_dataset(catalog, draft, grades, record),
            "Reviewed query order, passage order or judgment provenance mismatch")
    require(Counter(grades.values()) == GRADE_COUNTS, "Reviewed grade distribution mismatch")
    counts = {
        "documents": len({passage.passage_id.split(":")[1] for passage in catalog}),
        "passages": len(catalog), "queries": len(dataset.queries), "judgments": len(grades),
        "grades": {str(grade): count for grade, count in GRADE_COUNTS.items()},
        "positive_judgments": sum(grade > 0 for grade in grades.values()),
        "grades_2_or_3": sum(grade >= 2 for grade in grades.values()),
        "flags_resolved": len(record["flagged_resolutions"]),
        "grade_changes": 0, "pending_judgments": 0, "unresolved_disagreements": 0,
        "query_categories": dict(Counter(query.category for query in dataset.queries)),
    }
    require(json_digest(record.get("counts")) == json_digest(counts), "Approval record counts mismatch")
    require(record["corpus_fingerprint"] == dataset.corpus_fingerprint
            and record.get("dataset_fingerprint") == json_digest(dataset.model_dump()),
            "Approval record fingerprint mismatch")
    seen = set()
    for flag in record["flagged_resolutions"]:
        require(isinstance(flag, dict) and all(nonempty_text(flag.get(key)) for key in (
            "query_id", "passage_id", "passage_ref", "ai_flag_reason",
        )), "Approval record contains an invalid flagged resolution")
        pair = (flag["query_id"], flag["passage_id"])
        require(pair in grades and pair not in seen, "Approval record contains an inconsistent flagged resolution")
        _, key, ordinal, _, _ = pair[1].split(":")
        require(flag["passage_ref"] == f"{key}/{ordinal}"
                and type(flag.get("reviewed_grade")) is int and type(flag.get("ai_proposed_grade")) is int
                and grades[pair] == flag["reviewed_grade"] == flag["ai_proposed_grade"]
                and type(flag.get("ai_alternative_grade")) is int and flag["ai_alternative_grade"] in GRADE_COUNTS
                and flag["ai_alternative_grade"] != grades[pair]
                and flag.get("resolution") == "retained-ai-proposed-grade",
                "Approval record contains an inconsistent flagged resolution")
        seen.add(pair)
    require(len(seen) == 41, "Approval record must contain all 41 flagged resolutions")


def export_bytes(dataset):
    return (json.dumps(dataset.model_dump(exclude_none=True), indent=2, ensure_ascii=True) + "\n").encode("ascii")


def publish_export(destination, payload):
    # Unlike runner reports, this export has no read-back validation step or
    # native handle to retain. Stage in a private directory on the same volume.
    destination = destination.absolute()
    with TemporaryDirectory(prefix=".aura-reviewed-", dir=destination.parent) as workspace:
        staged = Path(workspace) / "dataset.json"
        with staged.open("xb") as stream:
            if stream.write(payload) != len(payload):
                raise OSError("Incomplete reviewed dataset write")
            stream.flush()
        # Windows rename fails if the target exists. POSIX rename can replace
        # it, so use an exclusive hard link there.
        if os.name == "nt":
            staged.rename(destination)
        else:
            os.link(staged, destination)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validate or reproduce the human-reviewed portable dataset")
    parser.add_argument("--workbook", type=Path, help="Original hash-pinned assessment workbook")
    parser.add_argument("--output", type=Path, help="Write a fresh export to this path")
    args = parser.parse_args(argv)
    if args.output is not None and args.workbook is None:
        parser.error("--output requires --workbook")
    try:
        manifest, catalog, draft = canonical_inputs()
        record = read_json(CORPUS_DIR / RECORD_NAME)
        included = CORPUS_DIR / DATASET_NAME
        dataset = load_dataset(CORPUS_DIR, manifest, catalog, dataset_path=included)
        validate_artifacts(dataset, catalog, draft, record)
        if args.workbook is not None:
            grades, flags = reconcile(workbook_rows(args.workbook), manifest, catalog, draft, record["ai_assessor_id"])
            require(flags == record["flagged_resolutions"], "Workbook alternatives differ from the approval record")
            payload = export_bytes(build_dataset(catalog, draft, grades, record))
            require(payload == included.read_text(encoding="ascii").encode("ascii"),
                    "Workbook export differs from the included reviewed dataset")
            if args.output is not None:
                publish_export(args.output, payload)
        print(f"Validated {len(dataset.queries)} queries and 1,440 reviewed judgments")
        print(f"Dataset fingerprint: {record['dataset_fingerprint']}")
    except (OSError, ValueError, KeyError) as error:
        print(f"Reviewed dataset validation failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
