# AURA portable synthetic research corpus

This corpus is a separate benchmark foundation. The historical v0.2 labels and
results refer to database IDs and a workspace whose exact corpus is unavailable
in the repository. Portable judgments refer to logical passage IDs. Scores
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
- [`dataset.reviewed.json`](dataset.reviewed.json) contains the exhaustive schema v2 judgments.
- [`review_approval.json`](review_approval.json) records the owner approval, source workbook hash,
  confirmed query interpretations, and 41 resolved alternatives.
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
in a seed readiness record.

**Dataset schemas v1 and v2.** The dataset supplies a matching corpus
fingerprint, a draft or reviewed status, and unique query IDs with a category
and text. Each judgment supplies a catalog passage ID, a grade from 0 to 3,
an assessment method, an assessor ID, and review status. A reviewed judgment
requires a different reviewer ID. Zero is an explicit non-relevance decision.
An empty judgment list leaves the query unjudged. References to a different
corpus, unknown passages, and excluded passages fail validation.

Version 1 supports the original `method: "human"` judgment format. Version 2
also supports `method: "ai-assisted"`. For that method, `assessor_id` identifies
the AI assessor and `ai_provenance` records its `provider` and `tool`. Include
`model` and `model_revision` only when known. A pending judgment leaves both
`reviewer_id` and `human_review` unset. After an actual human verifies the
judgment and approves the grade, record their identity in `reviewer_id` and
add `human_review` with `reviewer_kind: "human"`, `decision: "approved"`,
`approved_on` (a calendar date in YYYY-MM-DD format), and
`approval_reference` (a reference to the retained approval record). The
dataset rejects any reviewer ID also declared as an AI assessor in the dataset.
Keep the underlying review record and use a reference suitable for publishing.
Validation checks the structure and consistency of these declarations.
Authenticating the reviewer and verifying that they performed and approved the
review remain the responsibility of the dataset owner.

Version 1 datasets retain their canonical fingerprints. Version 2 includes
the AI and human review declarations in the dataset fingerprint. The report
schema and evaluation protocol keep their existing versions because their
format and scoring rules are unchanged.

## Human-verified relevance judgments

Codex proposed a grade for every pair of the 36 committed queries and 40
eligible passages from eight synthetic documents. The project owner verified
individual passages and groups of zero-grade judgments, resolved all 41
flagged alternatives, and approved the complete dataset on 2026-11-27 with
zero grade changes. The approval record includes the owner's exact statement
as supplied in the project handoff. Flag rationales in that record come from
the AI draft. The human decision for each flagged alternative was to retain
the proposed grade. This workflow used one human reviewer verifying an AI
draft, so it provides no independent inter-annotator agreement measurement.

| Grade | Meaning | Judgment count |
| --- | --- | ---: |
| 0 | No useful evidence | 1,287 |
| 1 | Useful indirect evidence or supporting context | 70 |
| 2 | Direct evidence addressing a substantial part of the query | 52 |
| 3 | Sufficient direct evidence addressing all essential parts | 31 |
| **Total** | | **1,440** |

These are annotation counts. There are 153 positive judgments and 83 with
grades 2 or 3. The reviewed export preserves all 36 query IDs, texts,
categories, and ordering. The original schema v1 draft remains available for
ranking-only runs. This checkpoint produced no retrieval scores or baseline.

Run the validation command from `backend`. **Workbook-free validation** checks
the frozen corpus and draft and checks the reviewed dataset and approval record
for structural and internal consistency. It cannot independently establish
that every grade matches the workbook or detect every coordinated change to
the two included files.

```console
python -m evaluation.portable_benchmark.export_reviewed_dataset
```

**Workbook-backed validation** verifies the original workbook's pinned SHA-256
before parsing, reconciles every judgment and grade against that source, and
checks the flagged alternatives against the workbook. This establishes
correspondence with the hash-pinned AI draft approved by the project owner.
Both modes require the exact approval statement and the complete interpretation
inventory. Authenticating the reviewer or verifying that review occurred
requires evidence outside the validator. Independent human annotation was
outside this workflow.

```console
python -m evaluation.portable_benchmark.export_reviewed_dataset --workbook /path/to/aura-relevance-assessment-draft.xlsx --output /path/to/fresh-reviewed.json
```

Omit `--output` to compare the deterministic export without writing. Keep the
workbook outside the repository. Use an existing, trusted local output
directory. The exporter writes, flushes, and closes a private staging file
before publishing it by same-volume rename on Windows or hard link on Linux.
Publication requires filesystem support for the corresponding operation and
fails if the destination exists, including when another process creates it
during publication.
Handled failures remove staging. An abrupt process termination can leave a
`.aura-reviewed-*` staging directory. Publication does not guarantee persistence
through a power loss.

The corpus is small and synthetic, so results on it cannot establish
real-world retrieval generalization. Passage-level metrics measure how
individual evidence passages rank. They do not establish whether the
retrieved set jointly covers all essential clauses of a cross-domain question.

## Seed a disposable benchmark environment

Run from `backend` with Python 3.12, the backend dependencies, and a local
Linux Docker Engine or Docker Desktop in Linux-container mode. Prepare the
pinned MiniLM weights once. Cold preparation needs public Hugging Face
access. Seeding validates every required model file and loads it offline
on CPU with remote model code disabled. A cold `tiktoken` cache can still
need its public vocabulary download. No Ollama or paid inference is used.

```bash
python tests/prepare_embedding_model.py /absolute/path/to/aura-benchmark-minilm

state="$(mktemp -d)/ownership.json"
python -m evaluation.portable_benchmark.seed_lifecycle start \
  --docker-context default --state "$state" --postgres-port 15432 --qdrant-port 16333

project="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["record"]["project"])' "$state")"
python -m evaluation.portable_benchmark.seed \
  --docker-context default --project "$project" --postgres-port 15432 --qdrant-port 16333 \
  --model-dir /absolute/path/to/aura-benchmark-minilm
# Repeat the same seed command to verify the existing state.
python -m evaluation.portable_benchmark.seed_lifecycle stop --state "$state"
```

For Windows PowerShell, select `desktop-linux` if that is the local Docker
context and use the following commands with a prepared model directory:

```powershell
$state = Join-Path $env:TEMP ('aura-bench-' + (New-Guid).ToString('N') + '.json')
python -m evaluation.portable_benchmark.seed_lifecycle start --docker-context desktop-linux --state $state --postgres-port 15432 --qdrant-port 16333
if ($LASTEXITCODE -ne 0) { throw 'Benchmark startup failed' }
$stack = (Get-Content -LiteralPath $state -Raw | ConvertFrom-Json).record
python -m evaluation.portable_benchmark.seed --docker-context $stack.context --project $stack.project --postgres-port $stack.postgres_port --qdrant-port $stack.qdrant_port --model-dir 'C:\path\to\prepared-minilm'
# After verification:
python -m evaluation.portable_benchmark.seed_lifecycle stop --state $state
```

### Lifecycle ownership and isolation

| Resource | Benchmark setting |
| --- | --- |
| PostgreSQL | 16.16 Bookworm, pinned by digest. Database `aura_portable_benchmark`, role `aura_benchmark`, loopback host port 15432, and `tmpfs` at `/var/lib/postgresql/data`. |
| Qdrant | 1.20.0, pinned by digest. Collection `aura_portable_passages_v1`, loopback host port 16333, and `tmpfs` at `/qdrant/storage`. |
| Generated PDFs | A temporary directory owned by the seeder. SQL `storage_key` values identify benchmark documents without publishing PDF files to the application's document browser. |

The lifecycle command chooses a unique project and checks the namespace
and project labels for existing containers, networks, and volumes before
creation. It also checks the requested loopback ports. Docker reserves the
project through a stopped, uniquely named ownership container. The
namespace can be supplied explicitly with `--project` for testing, with
the same conflict checks. The lifecycle makes a labeled bridge and gives
it to Compose as an external network. Compose creates with
`--no-recreate`, and the lifecycle starts only the service IDs it
inspected. The explicit environment file avoids loading the application's
`.env`. The command ignores inherited Compose overrides and generates
fresh, separate credentials for each execution.

Keep the ownership record in a trusted local directory outside the
repository. It records the execution label, project, context, ports, and
resource IDs without credentials. A file lock serializes commands on the
same record. The Docker name reservation handles two cooperating starts
that choose the same project using different files. Shutdown checks the
record, the execution labels, the recorded IDs, and network attachments.
It removes resources by exact Docker ID and leaves the ownership record
marked stopped. It does not use `docker compose down` or delete volumes.
The record checksum detects accidental edits and does not authenticate a
file against a user who can rewrite it and control Docker.

### Seed identity and readiness

The seeder accepts no database or vector-store URL. It verifies the local
Docker context, benchmark configuration, images, network, loopback ports,
`tmpfs` mounts, execution label, and ownership reservation. PostgreSQL
migrations and data writes use one connection and transaction. The
connection must report the same database, role, and cluster system ID as
the inspected PostgreSQL container. A PostgreSQL transaction advisory
lock excludes concurrent seed attempts for that database.

Qdrant requests are sent through `docker exec` using the full inspected
container ID. The exec checks the container process namespace and start
token before opening its internal loopback socket for each request. A
new container reusing the published port and API key does not receive
those requests. A restart under the same container ID changes the process
identity. The seeder repeats infrastructure checks before storing or
returning readiness. Test canary teardown also uses the original Qdrant
transport and verifies the PostgreSQL cluster before issuing SQL cleanup.

The Qdrant library can warn that the API key is used over HTTP. The seeder's
requests use HTTP on the inspected container's loopback interface, and
the Compose host port is bound to `127.0.0.1`. This setup is for disposable
local verification. The HTTP service, temporary storage, and local
administrative credentials are unsuitable for remote exposure or for
production use.

The seeder verifies the frozen 40-passage catalog again after text PDF
extraction and benchmark-specific chunking. It indexes retrievable chunks
as normalized 384-dimensional cosine vectors and verifies every stored
point ID, payload, and vector. A successful run prints a versioned JSON
readiness record and stores the same record in PostgreSQL's
`aura_portable_seed` table. It contains corpus and protocol fingerprints,
migration and MiniLM revisions, verified model manifest digest, resource
identities, and the logical-to-runtime mappings. Vector digests hash the
read-back float32 values. Repeat seeding verifies SQL rows and the full
Qdrant collection against that record and returns identical readiness for
the intact stack. Seeding does not create relevance judgments or benchmark scores.

### Recovery and limitations

If startup is interrupted, run `seed_lifecycle stop` with the original
ownership file. When Docker completed only part of creation, the command
uses the reserved execution label to inspect and record owned resources
before removing them. After interrupted teardown, rerun the same stop
command. A missing, altered, or inconsistent record or an unexpected
resource makes cleanup stop. Preserve those resources for inspection.
Choose a fresh state path and project after investigation. Do not edit
the record to make it match a replacement.

SQL and Qdrant do not share a transaction. If indexing succeeded but SQL
did not commit, the seeder rejects the orphaned collection on the next
run. If SQL committed but stdout was lost, repeat seeding verifies and
returns the recorded readiness. Detected replacement never triggers
automatic deletion. Stopping a container discards its temporary data.

The checks assume a trusted Docker daemon, host, and ownership-file
directory. Non-cooperating Docker operations can race between inspection
and creation, so startup refuses ambiguity and leaves resources for
inspection. A service can also stop after final verification. The seeder
cannot make container lifetime, Qdrant writes, and SQL commit atomic.
Docker administrators can change container internals and are outside this
boundary. CPU libraries, PyMuPDF, and Qdrant versions can affect PDF bytes
or float representations. Logical passage identity does not depend on
bitwise equality of embeddings across environments.

### Opt-in live verification

Default core CI collects the service-independent regressions without
PostgreSQL. For the actual Docker tests, use the lifecycle command twice
with distinct ownership paths and ports `15432/16333` and
`15433/16343`. Keep both stacks fresh. Set the following environment
variables:

- `AURA_BENCHMARK_TEST_CONTEXT` to the local Docker context.
- `AURA_BENCHMARK_TEST_PROJECT`, `AURA_BENCHMARK_TEST_PG_PORT`, and
  `AURA_BENCHMARK_TEST_QDRANT_PORT` to the first ownership record's values.
- The same three names with `_SECOND` appended to the second record's values.
- `AURA_BENCHMARK_TEST_MODEL_DIR` to the verified MiniLM directory.

```bash
python -m pytest -q --strict-markers -m portable_benchmark_live \
  evaluation/portable_benchmark/integration_tests
```

The suite checks initial and repeat seeding, SQL-to-Qdrant consistency,
independent runtime mappings, canary preservation, project collisions,
and Qdrant replacement on the original port with the original key.
The two lifecycle regression tests create and remove their own
disposable resources. Use the two original ownership records to stop
the operator-provisioned stacks after verification.

## Run portable retrieval

Seed a disposable stack using the lifecycle commands above and keep it
running. The runner requires the explicit Docker context, project and
ports from that stack. It verifies the frozen corpus, the selected
dataset, the pinned models, and the readiness record against live
PostgreSQL rows and Qdrant points. Evaluation runs under a read-only
repeatable-read SQL transaction with the seeder advisory lock. Qdrant
queries use the seeder's container-bound transport. The runner verifies
the full seed and service identities again before publishing results.

### Prepare and use the offline models

From `backend`, prepare the embedding model as shown above and prepare
the separate reranker in a trusted directory outside the repository.

```bash
python -m evaluation.portable_benchmark.reranker_assets \
  --model-dir /absolute/path/to/aura-benchmark-reranker
```

The reranker is `cross-encoder/ms-marco-MiniLM-L-6-v2` at revision
`233902d25c440f23af6f7d6e94d2946bac0bee0a`. Its manifest lists six
inference files and their SHA-256 digests. The Safetensors digest was
checked against the upstream LFS digest and the five smaller files
against the upstream Git blob IDs before recording their SHA-256
digests. Preparation downloads only those files and validates them on
first use and reuse. Both models revalidate their assets before loading
from local paths on CPU with remote code disabled. Cold model preparation
and a cold `tiktoken` vocabulary cache can require public downloads.

Run against the 36 checked-in draft queries after seeding. Keep the
same `$project` and retained ownership file from the lifecycle example.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m evaluation.portable_benchmark.run_benchmark \
  --docker-context default --project "$project" --postgres-port 15432 --qdrant-port 16333 \
  --embedding-model-dir /absolute/path/to/aura-benchmark-minilm \
  --reranker-model-dir /absolute/path/to/aura-benchmark-reranker \
  --mode ranking-only --output /absolute/path/to/reports/aura-portable-ranks.json
```

In PowerShell, use `$stack.context`, `$stack.project`,
`$stack.postgres_port` and `$stack.qdrant_port` from the ownership
record. Set `$env:HF_HUB_OFFLINE = '1'` and
`$env:TRANSFORMERS_OFFLINE = '1'` before invoking Python. To shut down,
use `seed_lifecycle stop` with the retained ownership file.

### Retrieval and logical ordering

| Default profile | Candidate selection |
| --- | --- |
| `dense` | Qdrant cosine similarity |
| `hybrid` | Dense and production BM25 rankings combined by reciprocal rank fusion, pool 20 |
| `hybrid_rerank_5` | Fused pool of 5, scored by the cross-encoder |
| `hybrid_rerank_10` | Fused pool of 10, scored by the cross-encoder |
| `hybrid_rerank_20` | Fused pool of 20, scored by the cross-encoder |

`--top-k` defaults to 5. `--hybrid-pool` defaults to the greater of
20 and four times top-k, capped at the retrievable corpus size.
`--rerank-pools 5 10 20` sets three distinct increasing pools. Each
pool must be between top-k and the retrievable corpus size.

The runner reuses the production search orchestration, BM25, fusion
and cross-encoder. For evaluation it requests exact dense scores for
all verified points and collects all BM25 matches before their cutoffs.
It sorts equal scores by logical passage ID, uses that ID for fusion,
and breaks equal reranker scores the same way. Each returned runtime
chunk or point must match its readiness mapping and expected payload.
Exhaustive dense retrieval is appropriate for this 40-passage corpus.
Different runtime IDs do not affect tie ordering. CPU libraries and
near-ties can still change scores or ranks across platforms.

### Reviewed datasets and metrics

Ranking-only reports contain logical rankings, scores and diagnostic
latency. Quality fields are absent. The included schema v2 dataset meets the
judgment requirements for a future quality run. To use it, pass
`--mode quality --dataset evaluation/portable_benchmark/dataset.reviewed.json` and
`--dataset-sha256 <canonical-dataset-sha256>` plus a new output path.
Use `run_benchmark.dataset_fingerprint` on the validated `PortableDataset` to
derive the digest. It covers compact, sorted-key JSON with non-ASCII
characters escaped, including judgment metadata and review state.

Grades 1–3 count as relevant for Precision@K, Recall@K and MRR@K.
Precision divides by K, Recall divides by the number of positively
judged passages for the query, and MRR uses the first positive hit
within K. nDCG@K uses grades 0–3 with gain `2**grade - 1` and ideal
ordering from the judged grades. Unjudged retrieved passages contribute
zero gain. Queries without a positive judgment are rejected. Results
include equal-weight per-query averages overall and within categories.

### Reports, recovery, and limitations

Report schema 1 records the corpus and dataset fingerprints and review
status, passage and evaluation protocols, retrieval configuration,
both model revisions and their manifest digests, software versions and
platform, query and category IDs, and ranked logical passage IDs.
Latency is diagnostic and includes retrieval overhead such as Docker
transport and cache warmup.

Validation requires independent expectations. From `backend`, use
`run_benchmark.load_report_context("ranking-only")` for the default
configuration and pass that context with the report bytes to
`ranking_report.validate_report(data, context)` or
`BenchmarkReport.model_validate_json(data, context=context)`. Both reject
duplicate keys and nonfinite numbers before schema validation. Generic
Pydantic JSON adapters cannot validate a portable report directly.
Supply custom parameters or a reviewed dataset and its canonical SHA-256
to the context loader from the original run configuration. Do not derive them from the report
under review. The loader revalidates the corpus, frozen passage catalog,
dataset, and pinned model manifests. Default reports require all 180
query/profile combinations and exact query categories. Every ranking
must use known logical passage IDs, finite scores, and the declared
score and tie order. Quality validation recomputes each query's metrics
from reviewed judgments and derives both macro averages and category
counts from those recomputed scores. Metric fields may differ by at most
`1e-12` absolute, and counts and category membership must agree exactly.
Ranking-only reports have no quality fields.

Choose a new `.json` filename in an existing local directory you control,
outside the historical datasets and results directories. The runner
writes, flushes, reads back, and validates the same file handle used for
publication. On Windows, native handles pin the directory and its
ancestors against rename and deletion. The staged file is opened
exclusively, and the final rename and any failure cleanup operate by
handle. On Linux, directory descriptors disallow links during traversal.
The runner stages an unnamed inode and links it into the verified
directory without overwriting. The publisher owns the staging descriptor
and lends it to the stream, so a failed stream setup closes the
descriptor and removes any owned Windows staging file. Linux requires
a local filesystem with
`O_TMPFILE` support and accessible `/proc/self/fd`. Unsupported filesystems
fail without falling back to path-based publication.

On Linux, a directory rename after the final identity check can move
the published name with the original directory. It cannot redirect
publication to a replacement directory or substitute a staged path.
Do not grant unrelated processes write access to the report directory.
These checks do not defend against a privileged process or an actor
allowed to alter the output after publication. An interruption before
publication leaves no final report. Abrupt termination can leave a
randomly named `.aura-report-*.tmp` on Windows. Remove such a file only
after confirming the runner that owns it has stopped.

Runner diagnostics identify the failed phase. Model validation can name
the model and an expected relative asset. The CLI suppresses raw nested
exceptions and library progress so error output does not include local
paths, connection details, or credentials. A collision or failed run
returns nonzero.

For failed asset verification, move the invalid model directory aside
and prepare a fresh destination. On incomplete readiness, an identity
change or inconsistent vectors, inspect the original ownership record
and follow the lifecycle recovery instructions above. SQL and Qdrant
do not share a snapshot. A concurrent out-of-band change that is fully
reverted between checks can evade the before-and-after comparisons.
The historical v0.2 regression checker uses a separate report format.
No portable quality baseline is approved in this checkpoint.

For the opt-in ranking test, also set
`AURA_BENCHMARK_TEST_RERANKER_DIR` to the prepared reranker directory.
The ranking test creates and cleans up two independent stacks through
the lifecycle command and compares their logical rankings.
