"""Provision and remove only resources owned by a disposable benchmark execution."""

import argparse
from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import secrets
import socket
import sys
from tempfile import NamedTemporaryFile
from time import monotonic, sleep
from typing import Literal

from evaluation.portable_benchmark.corpus import json_digest, read_json
from evaluation.portable_benchmark.schema import StrictModel, VersionOne
from evaluation.portable_benchmark.seed_infra import (
    ANCHOR_LABEL, COMPOSE_FILE, CONTAINER_ID, EXECUTION_ID, EXECUTION_LABEL, QDRANT_IMAGE,
    SERVICES, SeedError, docker, local_context, resolve_targets, validate_anchor,
    validate_container, validate_request,
)
from evaluation.portable_benchmark.seed_transport import ContainerTransportError


PROJECT_LABEL = "com.docker.compose.project"


class Ownership(StrictModel):
    schema_version: VersionOne
    execution_id: str
    context: str
    project: str
    postgres_port: int
    qdrant_port: int
    phase: Literal["starting", "ready", "stopped"]
    anchor: str | None
    network: str | None
    containers: dict[str, str]

    @property
    def network_name(self):
        return f"{self.project}_default"

    def check(self):
        validate_request(self.context, self.project, self.postgres_port, self.qdrant_port)
        ids = [item for item in (self.anchor, self.network, *self.containers.values()) if item is not None]
        if (
            not EXECUTION_ID.fullmatch(self.execution_id) or not self.containers.keys() <= SERVICES.keys()
            or len(set(ids)) != len(ids) or any(not CONTAINER_ID.fullmatch(item) for item in ids)
            or (self.phase == "ready" and (not self.anchor or not self.network or self.containers.keys() != SERVICES.keys()))
            or (self.containers and (not self.anchor or not self.network))
            or (self.network and not self.anchor)
        ):
            raise SeedError("Invalid benchmark ownership record")
        return self


def checked_path(path):
    absolute = path.absolute()
    destination = absolute.parent.resolve(strict=True) / absolute.name
    guard = destination.with_name(destination.name + ".lock")
    if absolute != destination or any(
        candidate.is_symlink() or getattr(candidate, "is_junction", lambda: False)()
        for candidate in (destination, guard)
    ):
        raise SeedError("Use a regular ownership path under a trusted local directory")
    return destination, guard


@contextmanager
def state_lock(path):
    destination, guard = checked_path(path)
    with guard.open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"\0")
            lock.flush()
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise SeedError("Another lifecycle command is using this ownership file") from error
        yield destination


def save_state(path, state, *, initial=False):
    record = state.check().model_dump()
    payload = json.dumps({"record": record, "sha256": json_digest(record)}, indent=2) + "\n"
    if initial:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return
    with NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload.encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_state(path):
    try:
        envelope = read_json(path)
        if set(envelope) != {"record", "sha256"} or json_digest(envelope["record"]) != envelope["sha256"]:
            raise ValueError("checksum mismatch")
        return Ownership.model_validate(envelope["record"]).check()
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise SeedError("Missing, malformed, or altered benchmark ownership record") from error


def project_resources(state):
    """Discover by namespace and labels so ambiguous leftovers fail closed."""
    containers = set()
    networks = set()
    volumes = set()
    selectors = (f"{PROJECT_LABEL}={state.project}", f"{EXECUTION_LABEL}={state.execution_id}")
    for kind, found, fmt, flags in (
        ("container", containers, "{{.ID}}\t{{.Names}}", ("--all", "--no-trunc")),
        ("network", networks, "{{.ID}}\t{{.Name}}", ("--no-trunc",)),
        ("volume", volumes, "{{.Name}}\t{{.Name}}", ()),
    ):
        for line in docker(state.context, kind, "ls", *flags, "--format", fmt).splitlines():
            identifier, name = line.split("\t", 1)
            if name == state.project or name.startswith((f"{state.project}-", f"{state.project}_")):
                found.add(identifier)
        for label in selectors:
            found.update(docker(state.context, kind, "ls", *flags, "-q", "--filter", f"label={label}").splitlines())
    if not all(CONTAINER_ID.fullmatch(item) for item in containers | networks):
        raise SeedError("Docker returned invalid project resource IDs")
    return containers, networks, volumes


def inspect_ids(context, kind, ids):
    if not ids:
        return []
    records = json.loads(docker(context, kind, "inspect", *sorted(ids)))
    if len(records) != len(ids) or {item["Id"] for item in records} != set(ids):
        raise SeedError("Docker resource identities changed during inspection")
    return records


def owned_resources(state):
    container_ids, network_ids, volumes = project_resources(state)
    if volumes or len(network_ids) > 1:
        raise SeedError("Unexpected benchmark project volumes or networks. No resources were removed")
    found = {}
    anchor = None
    for item in inspect_ids(state.context, "container", container_ids):
        labels = item["Config"]["Labels"] or {}
        if labels.get(ANCHOR_LABEL) == "1":
            if anchor is not None:
                raise SeedError("Multiple benchmark ownership reservations")
            anchor = validate_anchor(item, state.project, state.execution_id)
            continue
        service = labels.get("com.docker.compose.service")
        if (
            service not in SERVICES or service in found
            or labels.get(EXECUTION_LABEL) != state.execution_id
            or labels.get(PROJECT_LABEL) != state.project
        ):
            raise SeedError("Unexpected or unowned benchmark project container")
        port = state.postgres_port if service == "postgres" else state.qdrant_port
        validate_container(item, state.project, service, port, before_start=True)
        found[service] = item["Id"]

    network = None
    for item in inspect_ids(state.context, "network", network_ids):
        labels = item["Labels"] or {}
        if (
            item["Name"] != state.network_name or item["Driver"] != "bridge" or item["Scope"] != "local"
            or labels.get(EXECUTION_LABEL) != state.execution_id or labels.get(PROJECT_LABEL) != state.project
            or labels.get("com.docker.compose.network") != "default"
            or not set(item["Containers"] or {}) <= set(found.values())
        ):
            raise SeedError("Benchmark network has changed ownership or has unexpected attachments")
        network = item["Id"]

    if (network or found) and not anchor:
        raise SeedError("Benchmark ownership reservation is missing")
    if (
        (anchor and state.anchor and anchor != state.anchor)
        or (network and state.network and network != state.network)
        or any(state.containers.get(service, identifier) != identifier for service, identifier in found.items())
        or (
            state.phase == "ready"
            and ((network and not state.network) or not found.keys() <= state.containers.keys())
        )
    ):
        raise SeedError("Benchmark resources were replaced. No resources were removed")
    return anchor, network, found


def verify_ports(postgres_port, qdrant_port):
    try:
        with ExitStack() as sockets:
            for port in (postgres_port, qdrant_port):
                listener = sockets.enter_context(socket.socket())
                if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                    listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                listener.bind(("127.0.0.1", port))
    except OSError as error:
        raise SeedError(f"Benchmark loopback port {port} is unavailable") from error


def compose_environment(state):
    # Ignore inherited Compose options and application benchmark variables.
    result = {
        key: value for key, value in os.environ.items()
        if not key.upper().startswith(("COMPOSE_", "AURA_BENCHMARK_"))
    }
    result.update(
        AURA_BENCHMARK_EXECUTION_ID=state.execution_id, AURA_BENCHMARK_NETWORK=state.network_name,
        AURA_BENCHMARK_POSTGRES_PORT=str(state.postgres_port), AURA_BENCHMARK_QDRANT_PORT=str(state.qdrant_port),
        AURA_BENCHMARK_POSTGRES_PASSWORD=secrets.token_hex(32), AURA_BENCHMARK_QDRANT_KEY=secrets.token_hex(32),
    )
    return result


def wait_for_services(state):
    deadline = monotonic() + 60
    last_error = None
    while monotonic() < deadline:
        try:
            targets = resolve_targets(state.context, state.project, state.postgres_port, state.qdrant_port)
            if (
                targets.identity.postgres_container != state.containers["postgres"]
                or targets.identity.qdrant_container != state.containers["qdrant"]
                or targets.identity.execution_id != state.execution_id
            ):
                raise SeedError("Benchmark container identity changed during startup")
            targets.verify_qdrant()
            return
        except (SeedError, ContainerTransportError) as error:
            last_error = error
            sleep(1)
    raise SeedError(f"Benchmark services did not become ready. {last_error}")


def start(path, context, project, postgres_port, qdrant_port):
    execution = secrets.token_hex(16)
    project = project or f"aura-benchmark-{execution}"
    state = Ownership(
        schema_version=1, execution_id=execution, context=context, project=project,
        postgres_port=postgres_port, qdrant_port=qdrant_port, phase="starting",
        anchor=None, network=None, containers={},
    ).check()
    with state_lock(path) as destination:
        if destination.exists():
            raise SeedError("Use a new ownership file for a new benchmark execution")
        local_context(context)
        if any(project_resources(state)):
            raise SeedError("Existing resources conflict with the benchmark project")
        verify_ports(postgres_port, qdrant_port)
        save_state(destination, state, initial=True)

        # Docker reserves container names atomically, including across state files.
        state.anchor = docker(
            context, "container", "create", "--name", f"{project}-ownership", "--network", "none",
            "--entrypoint", "/bin/true", "--tmpfs", "/qdrant/storage",
            "--label", f"{ANCHOR_LABEL}=1", "--label", f"{EXECUTION_LABEL}={execution}",
            QDRANT_IMAGE, timeout=300,
        ).strip()
        save_state(destination, state)
        if owned_resources(state) != (state.anchor, None, {}):
            raise SeedError("Conflicting resources appeared during benchmark reservation")

        state.network = docker(
            context, "network", "create", "--driver", "bridge", "--label", f"{PROJECT_LABEL}={project}",
            "--label", "com.docker.compose.network=default", "--label", f"{EXECUTION_LABEL}={execution}",
            state.network_name,
        ).strip()
        save_state(destination, state)
        if owned_resources(state) != (state.anchor, state.network, {}):
            raise SeedError("Conflicting resources appeared before Compose creation")

        # Compose cannot recreate containers or manage the external network.
        # Starting explicit inspected IDs avoids starting a same-project container.
        docker(
            context, "compose", "--env-file", str(COMPOSE_FILE.with_name("compose.env")),
            "-f", str(COMPOSE_FILE), "-p", project, "create", "--no-recreate",
            env=compose_environment(state), timeout=300,
        )
        anchor, network, containers = owned_resources(state)
        if anchor != state.anchor or network != state.network or containers.keys() != SERVICES.keys():
            raise SeedError("Compose did not create exactly the two owned benchmark services")
        state.containers = containers
        save_state(destination, state)
        docker(context, "container", "start", *containers.values(), timeout=45)
        wait_for_services(state)
        state.phase = "ready"
        save_state(destination, state)
        return state


def stop(path):
    with state_lock(path) as destination:
        state = load_state(destination)
        if state.phase == "stopped":
            raise SeedError("This ownership record is already stopped")
        local_context(state.context)
        anchor, network, containers = owned_resources(state)
        # A partial create may have finished just before its IDs were persisted.
        # Write them before removing anything so retries refuse replacements.
        if anchor:
            state.anchor = anchor
        if network:
            state.network = network
        state.containers.update(containers)
        save_state(destination, state)
        for identifier in containers.values():
            item, = inspect_ids(state.context, "container", {identifier})
            if item["State"]["Running"]:
                docker(state.context, "container", "stop", "--time", "10", identifier)
            docker(state.context, "container", "rm", identifier)
        if network:
            docker(state.context, "network", "rm", network)
        if anchor:
            docker(state.context, "container", "rm", anchor)
        if any(project_resources(state)):
            raise SeedError("Benchmark resources still exist. Ownership record was retained")
        state.phase = "stopped"
        save_state(destination, state)
        return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    launch = commands.add_parser("start")
    launch.add_argument("--state", required=True, type=Path)
    launch.add_argument("--docker-context", required=True)
    launch.add_argument("--project", help="Optional unused benchmark project name")
    launch.add_argument("--postgres-port", required=True, type=int)
    launch.add_argument("--qdrant-port", required=True, type=int)
    teardown = commands.add_parser("stop")
    teardown.add_argument("--state", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "start":
            state = start(args.state, args.docker_context, args.project, args.postgres_port, args.qdrant_port)
        else:
            state = stop(args.state)
    except SeedError as error:
        print(f"Benchmark lifecycle failed: {error}", file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Benchmark lifecycle failed ({type(error).__name__}). Check Docker and the ownership file.", file=sys.stderr)
        return 1
    print(state.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
