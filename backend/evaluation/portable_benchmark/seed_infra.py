"""Resolve and prove ownership of the evaluation services before connecting for writes."""

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Self

import httpx
from pydantic import model_validator
from qdrant_client import QdrantClient
from sqlalchemy import URL, create_engine
from sqlalchemy.pool import NullPool

from evaluation.portable_benchmark.schema import StrictModel
from evaluation.portable_benchmark.seed_transport import (
    INSTANCE_PATTERN, ContainerQdrantTransport, ContainerTransportError, read_instance,
)


COMPOSE_FILE = Path(__file__).with_name("compose.yml").resolve()
DATABASE = "aura_portable_benchmark"
ROLE = "aura_benchmark"
COLLECTION = "aura_portable_passages_v1"
POSTGRES_IMAGE = "postgres:16.16-bookworm@sha256:cd0e9518bda5f2ea667ff84d22795d089720c883f288e215d97ce57437e8ef37"
QDRANT_IMAGE = "qdrant/qdrant:v1.20.0@sha256:9bfc89f9914fc14876a88c05270685921a55b8877b254a98bcc1cc2f0a424cd2"
SERVICES = {
    "postgres": (POSTGRES_IMAGE, "5432/tcp", "/var/lib/postgresql/data"),
    "qdrant": (QDRANT_IMAGE, "6333/tcp", "/qdrant/storage"),
}
PROJECT = re.compile(r"aura-benchmark-[a-z0-9][a-z0-9-]{0,39}")
CONTAINER_ID = re.compile(r"[0-9a-f]{64}")
EXECUTION_ID = re.compile(r"[0-9a-f]{32}")
EXECUTION_LABEL = "org.aura.portable-execution"
ANCHOR_LABEL = "org.aura.portable-anchor"
SQL_IDENTITY = (
    "SELECT current_database(), session_user, system_identifier::text "
    "FROM pg_catalog.pg_control_system()"
)


class SeedError(ValueError):
    """An actionable failure that is safe to print."""


class ResourceIdentity(StrictModel):
    docker_context: str
    project: str
    postgres_container: str
    qdrant_container: str
    postgres_port: int
    qdrant_port: int
    postgres_system_id: str
    execution_id: str
    qdrant_instance: str
    database: Literal["aura_portable_benchmark"]
    collection: Literal["aura_portable_passages_v1"]

    @model_validator(mode="after")
    def valid_identity(self) -> Self:
        validate_request(self.docker_context, self.project, self.postgres_port, self.qdrant_port)
        if (
            not CONTAINER_ID.fullmatch(self.postgres_container)
            or not CONTAINER_ID.fullmatch(self.qdrant_container)
            or self.postgres_container == self.qdrant_container
            or not re.fullmatch(r"[0-9]{1,20}", self.postgres_system_id)
            or not EXECUTION_ID.fullmatch(self.execution_id)
            or not re.fullmatch(INSTANCE_PATTERN, self.qdrant_instance)
        ):
            raise ValueError("Readiness has invalid benchmark resource identifiers")
        return self


@dataclass(frozen=True)
class Targets:
    identity: ResourceIdentity
    postgres_password: str = field(repr=False)
    qdrant_key: str = field(repr=False)

    def engine(self):
        address = URL.create(
            "postgresql+psycopg", username=ROLE, password=self.postgres_password,
            host="127.0.0.1", port=self.identity.postgres_port, database=DATABASE,
        )
        return create_engine(
            address, poolclass=NullPool,
            connect_args={
                "hostaddr": "127.0.0.1", "connect_timeout": 5, "sslmode": "disable",
                "options": "-c search_path=public",
            },
        )

    def qdrant_transport(self):
        instance = self.identity
        return ContainerQdrantTransport(instance.docker_context, instance.qdrant_container, instance.qdrant_instance)

    def qdrant(self):
        return QdrantClient(
            host="127.0.0.1", port=6333, https=False,
            api_key=self.qdrant_key, timeout=15, check_compatibility=False,
            trust_env=False, follow_redirects=False, transport=self.qdrant_transport(),
        )

    def verify_postgres(self, connection):
        row = tuple(connection.exec_driver_sql(SQL_IDENTITY).one())
        if row != (DATABASE, ROLE, self.identity.postgres_system_id):
            raise SeedError("PostgreSQL endpoint did not match the inspected benchmark container")

    def verify_qdrant(self):
        url = "http://127.0.0.1:6333/collections"
        with httpx.Client(
            transport=self.qdrant_transport(), timeout=5, trust_env=False, follow_redirects=False,
        ) as client:
            rejected = client.get(url)
            if rejected.status_code not in (401, 403):
                raise SeedError("Qdrant endpoint did not require authentication")
            accepted = client.get(url, headers={"api-key": self.qdrant_key})
        if accepted.status_code != 200:
            raise SeedError("Qdrant endpoint did not enforce the inspected benchmark API key")
        body = accepted.json()
        result = body.get("result") if isinstance(body, dict) else None
        if not isinstance(result, dict) or not isinstance(result.get("collections"), list):
            raise SeedError("Qdrant did not return its expected collection listing")

    def verify_connections(self, connection):
        self.verify_postgres(connection)
        self.verify_qdrant()

    def verify_current_identity(self, connection):
        current = resolve_targets(
            self.identity.docker_context, self.identity.project,
            self.identity.postgres_port, self.identity.qdrant_port,
        )
        if current != self:
            raise SeedError("Benchmark infrastructure identity changed. Readiness was not published")
        self.verify_connections(connection)


def docker(context, *arguments, env=None, timeout=20):
    try:
        return subprocess.run(
            ["docker", "--context", context, *arguments], check=True,
            capture_output=True, text=True, encoding="utf-8", timeout=timeout, env=env,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        # Inspect output and the subprocess exception can contain credentials.
        raise SeedError(f"Docker {arguments[0]} failed. Check the local engine and benchmark services.") from error


def validate_request(context, project, postgres_port, qdrant_port):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,79}", context):
        raise SeedError("Specify an explicit local Docker context")
    if not PROJECT.fullmatch(project):
        raise SeedError("Choose a dedicated project name such as aura-benchmark-local")
    ports = (postgres_port, qdrant_port)
    if any(type(port) is not int or not 1024 <= port <= 65535 for port in ports):
        raise SeedError("Specify valid benchmark host ports")
    if postgres_port == qdrant_port or set(ports) & {5432, 5433, 6333, 6334}:
        raise SeedError("Benchmark ports must be distinct from each other and from application service ports")


def validate_container(container, project, service, host_port, *, before_start=False):
    image, port, storage = SERVICES[service]
    config = container["Config"]
    labels = config["Labels"]
    files = labels.get("com.docker.compose.project.config_files", "")
    host = container["HostConfig"]
    binding = {port: [{"HostIp": "127.0.0.1", "HostPort": str(host_port)}]}

    if (
        not CONTAINER_ID.fullmatch(container["Id"])
        or labels.get("com.docker.compose.project") != project
        or labels.get("com.docker.compose.service") != service
        or labels.get("com.docker.compose.oneoff", "").lower() != "false"
        or labels.get("org.aura.portable-benchmark") != "1"
        or Path(files).resolve() != COMPOSE_FILE
        or config["Image"] != image
        or (not before_start and not container["State"]["Running"])
        or host.get("Privileged")
        or host.get("PidMode")
        or host.get("IpcMode") == "host"
        or host.get("Binds")
        or host.get("VolumesFrom")
        or host.get("NetworkMode") != f"{project}_default"
        or (
            not before_start
            and set(container["NetworkSettings"]["Networks"]) != {f"{project}_default"}
        )
        or set(host.get("Tmpfs") or {}) != {storage}
        or any(mount["Type"] != "tmpfs" or mount["Destination"] != storage for mount in container["Mounts"])
        or host.get("PortBindings") != binding
        or (
            not before_start
            and {key: value for key, value in container["NetworkSettings"]["Ports"].items() if value} != binding
        )
    ):
        raise SeedError(f"Refusing unverified {service} container. Use the dedicated Compose file and loopback ports")
    values = config["Env"]
    settings = dict(value.split("=", 1) for value in values)
    if len(values) != len(settings):
        raise SeedError(f"Ambiguous {service} container configuration")
    if service == "postgres" and (
        settings.get("POSTGRES_DB") != DATABASE or settings.get("POSTGRES_USER") != ROLE
        or settings.get("PGDATA") != f"{storage}/pgdata"
        or any(key in settings for key in ("POSTGRES_HOST_AUTH_METHOD", "POSTGRES_PASSWORD_FILE"))
    ):
        raise SeedError("Unexpected PostgreSQL database or authentication settings")
    if service == "qdrant" and settings.get("QDRANT_ALLOW_RECOVERY_MODE", "false") != "false":
        raise SeedError("Qdrant recovery mode can replace the service process within a container")
    key = settings.get("POSTGRES_PASSWORD" if service == "postgres" else "QDRANT__SERVICE__API_KEY", "")
    if not re.fullmatch(r"[0-9a-f]{64}", key):
        raise SeedError(f"{service} requires a per-stack secret generated with secrets.token_hex(32)")
    return key


def local_context(context):
    try:
        endpoint, = json.loads(docker(context, "context", "inspect", context))
        host = endpoint["Endpoints"]["docker"]["Host"]
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, SeedError):
            raise
        raise SeedError("Malformed Docker context inspection output") from error
    if not re.fullmatch(r"(unix:///[^\r\n]*docker\.sock|npipe:////\./pipe/docker[A-Za-z0-9_-]*)", host):
        raise SeedError("The Docker context does not address a supported local daemon")


def validate_anchor(container, project, execution):
    config = container["Config"]
    host = container["HostConfig"]
    labels = config["Labels"]
    if (
        container["Name"] != f"/{project}-ownership"
        or not CONTAINER_ID.fullmatch(container["Id"])
        or labels.get(EXECUTION_LABEL) != execution
        or labels.get(ANCHOR_LABEL) != "1"
        or config["Image"] != QDRANT_IMAGE
        or config["Entrypoint"] != ["/bin/true"]
        or container["State"]["Status"] != "created"
        or host["NetworkMode"] != "none"
        or host.get("Binds") or host.get("PortBindings")
        or any(mount["Type"] != "tmpfs" or mount["Destination"] != "/qdrant/storage" for mount in container["Mounts"])
    ):
        raise SeedError("Benchmark ownership reservation does not match the execution")
    return container["Id"]


def resolve_targets(context, project, postgres_port, qdrant_port):
    validate_request(context, project, postgres_port, qdrant_port)
    local_context(context)
    try:
        ids = docker(
            context, "ps", "-aq", "--no-trunc", "--filter", f"label=com.docker.compose.project={project}",
        ).splitlines()
        if len(ids) != 2 or len(set(ids)) != 2 or not all(CONTAINER_ID.fullmatch(item) for item in ids):
            raise SeedError("Expected exactly two containers in the named benchmark project")
        containers = {
            entry["Config"]["Labels"]["com.docker.compose.service"]: entry
            for entry in json.loads(docker(context, "inspect", *ids))
        }
        if set(containers) != set(SERVICES) or {entry["Id"] for entry in containers.values()} != set(ids):
            raise SeedError("Unexpected or missing benchmark service containers")

        network, = json.loads(docker(context, "network", "inspect", f"{project}_default"))
        if (
            network["Driver"] != "bridge" or network["Scope"] != "local"
            or network["Labels"].get("com.docker.compose.project") != project
            or network["Labels"].get("com.docker.compose.network") != "default"
            or set(network["Containers"]) != set(ids)
        ):
            raise SeedError("Benchmark network is not an exclusive local Compose bridge")

        pg_container, qdrant_container = containers["postgres"], containers["qdrant"]
        password = validate_container(pg_container, project, "postgres", postgres_port)
        key = validate_container(qdrant_container, project, "qdrant", qdrant_port)
        execution = network["Labels"].get(EXECUTION_LABEL, "")
        if not EXECUTION_ID.fullmatch(execution) or any(
            container["Config"]["Labels"].get(EXECUTION_LABEL) != execution for container in containers.values()
        ):
            raise SeedError("Benchmark resources do not belong to a single lifecycle execution")
        anchor, = json.loads(docker(context, "container", "inspect", f"{project}-ownership"))
        validate_anchor(anchor, project, execution)
        if password == key:
            raise SeedError("Use distinct per-stack PostgreSQL and Qdrant secrets")
        system_id = docker(
            context, "exec", pg_container["Id"], "psql", "-XAt", "-U", ROLE, "-d", DATABASE,
            "-c", "SELECT system_identifier::text FROM pg_catalog.pg_control_system()",
        ).strip()
        if not re.fullmatch(r"[0-9]{1,20}", system_id):
            raise SeedError("Could not verify the benchmark PostgreSQL cluster identity")
        qdrant_instance = read_instance(context, qdrant_container["Id"])
    except SeedError:
        raise
    except ContainerTransportError as error:
        raise SeedError(str(error)) from error
    except (KeyError, TypeError, AttributeError, ValueError) as error:
        raise SeedError("Malformed Docker inspection output. No benchmark target was accepted.") from error
    return Targets(
        ResourceIdentity(
            docker_context=context, project=project,
            postgres_container=pg_container["Id"], qdrant_container=qdrant_container["Id"],
            postgres_port=postgres_port, qdrant_port=qdrant_port, postgres_system_id=system_id,
            database=DATABASE, collection=COLLECTION, execution_id=execution, qdrant_instance=qdrant_instance,
        ),
        password, key,
    )
