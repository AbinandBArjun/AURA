# AURA portable synthetic research corpus

This corpus is a separate benchmark foundation. The historical v0.2 labels and
results refer to database IDs and a workspace whose exact corpus is unavailable
in the repository. Portable judgments will refer to logical passage IDs. Scores
from the two corpora require separate baselines.

## Corpus and provenance

The eight source texts are original, fictional research material written for
AURA. They contain no imported documents, real observations, or third-party
passages. Every source marks its observations and numerical examples as
synthetic. The manifest records `no-grant-recorded` for redistribution because
this checkpoint does not establish a separate corpus license.

| Domain | Documents |
|---|---|
| Artificial intelligence | Conditional Evidence Review in Larkspur Service Desks, Delayed Labels in a Fictional Inspection Model |
| Cybersecurity | Access Review and Recovery Ownership at Elm Harbor, Evidence from a Fictional Archive Restoration Exercise |
| Economics | Household Response to a Fictional Peak Tariff Credit, Switching Frictions in a Fictional Procurement Network |
| Software engineering | Replay Boundaries in a Fictional Work Order Pipeline, Cache Boundaries for Fictional Project Search |

The documents share vocabulary such as recovery, ownership, indexing, and
fictitious measurement windows. Each uses those terms for a different
operational question. The texts contrast superficially similar measurements,
include comparisons and qualifications, and include plausible passages that
share terms with a query while addressing a different problem. The 36 draft
queries include eight per domain and four cross-domain queries. They largely
describe the information need in different words from the sources.

The corpus is deliberately small and stylized. It can exercise retrieval
failure modes, but future scores on these texts will have limited external
validity for real research collections, scans, or document layouts.

## Layout

- `sources/` contains the eight canonical UTF-8 text files.
- `manifest.json` records corpus identity, source provenance, checksums, and the
  extraction and chunking protocols.
- `dataset.json` contains draft queries and empty judgment lists.
- `schema.py` defines the strict, versioned manifest and dataset schemas.
  `CorpusManifest.model_json_schema()` and `PortableDataset.model_json_schema()`
  expose their structural JSON Schemas. Runtime validation adds uniqueness,
  review, and corpus-reference checks.
- `corpus.py` validates the bundle, projects a passage catalog, and can generate
  text-based PDFs.

From `backend` with the existing dependencies installed:

```bash
python -m evaluation.portable_benchmark.corpus
python -m evaluation.portable_benchmark.corpus --pdf-dir /tmp/aura-corpus-pdfs
```

The PDF path must be a new directory under an existing, trusted local parent.
Use a path without symlinks, Windows junctions, or `..` components. Output files
are created exclusively. On Ubuntu the generator opens directories without
following links and opens each file relative to the verified output directory
descriptor. On Windows it holds directory handles without delete sharing for
the output and its ancestors, rejects reparse points, and creates each PDF with
the native exclusive-create operation. The CLI checks the directory identity
during generation and reports failure if an output cannot be written. It
leaves partial output in place for inspection. Choose a new directory to retry.

These safeguards assume a local filesystem that honors exclusive creation,
directory identities, and the platform's directory or sharing semantics.
Use a parent controlled by the invoking user. Directory creation and opening
are separate operations. The program can detect observed replacements, but
cannot exclude every race with another process that can modify the parent
during initialization. It does not provide protection against a process that
can modify completed files after the generator closes them.

PDFs are generated artifacts. The
generator uses US Letter pages, 10-point Courier, 13-point line spacing,
a 42-point left margin, and a first baseline 48 points from the top. It omits
per-run PDF identifiers. The
tests check byte repeatability under the same PyMuPDF runtime and confirm that
AURA's PDF extractor returns the intended page lines. Rendering-library
changes may change PDF bytes. PDF headers and binary layout are excluded from
corpus identity. Validation uses local files and the existing tokenizer. A
cold tokenizer cache may need the public vocabulary download already used by
backend CI.

## Identity and schemas

**Manifest schema v1.** The top-level schema, name, and corpus version describe
the bundle. Each document has a unique slug, title, domain, relative
`sources/<slug>.txt` path, provenance, redistribution status, and source
SHA-256. Validation rejects duplicate identifiers or paths, lexical path
traversal, links that redirect source paths, missing files, and checksum
mismatches. Source hashes use printable ASCII UTF-8 text with CRLF and CR
normalized to LF. Other whitespace, including the final newline, contributes
to the checksum.

The corpus fingerprint is SHA-256 of compact JSON with sorted object keys and
documents sorted by logical key. The fingerprint field itself is omitted.
All other manifest fields, including source checksums and the protocol
configuration, contribute. The calculation contains no absolute paths,
timestamps, PDF binary bytes, platform metadata, or database IDs.

**Passage protocol p1.** The canonical text is wrapped to 88 columns and
paginated at 50 lines. The projection uses AURA's section classifier, joins
consecutive lines in the same section on a page with spaces, and applies
AURA's 500-token chunker with 75-token overlap and `cl100k_base` from
`tiktoken` 0.14.0. The line joining is specific to this benchmark. This
checkpoint leaves the production ingestion path intact.

A logical ID has the form
`p1:<document-key>:<zero-based-chunk-index>:<protocol-sha256>:<text-sha256>`.
The protocol digest uses the canonical JSON encoding of the extraction,
chunking, and ID-scheme fields. The text digest uses UTF-8 after Unicode NFC
normalization and collapse of whitespace runs to single spaces. A change to
the defined extraction or chunking behavior requires a new protocol version.
Database document IDs, chunk IDs, and Qdrant point IDs are runtime mappings
for a later checkpoint.

**Dataset schema v1.** The dataset supplies a matching corpus fingerprint,
a draft or reviewed status, and unique query IDs with a category and text.
Each judgment supplies a catalog passage ID, a grade from 0 to 3, human
provenance through an assessor ID, and review status. A reviewed judgment
requires a different reviewer ID. Zero is an explicit non-relevance decision.
An empty judgment list leaves the query unjudged. References to a different
corpus, unknown passages, and excluded passages fail validation.

## Review and subsequent checkpoints

For future judgments, have an assessor record 0 for irrelevant, 1 for
marginally relevant, 2 for relevant, or 3 for highly relevant. A separate
reviewer checks the passage in context and adjudicates disagreements,
including plausible distractors. The reviewed dataset requires at least one
positive judgment per query and every recorded judgment to be reviewed.
This checkpoint includes no final judgments or scores.

A later checkpoint can seed a disposable database and vector collection,
capture the logical-to-runtime ID map, and verify that seeded passages
match this protocol. Run and review a portable baseline only after corpus,
judgment, and seeding checks pass. Keep those artifacts separate from v0.2.
