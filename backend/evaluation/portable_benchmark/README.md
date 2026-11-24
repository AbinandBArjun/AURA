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
in a seed readiness record.

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

Run and review a portable baseline only after corpus, judgment, and seeding checks pass. Keep those artifacts separate from v0.2.

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
the intact stack. There are no relevance judgments or benchmark scores.

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
