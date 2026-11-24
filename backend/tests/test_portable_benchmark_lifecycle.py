import json
from unittest.mock import Mock

import pytest

from evaluation.portable_benchmark import seed_lifecycle as lifecycle
from evaluation.portable_benchmark.seed_infra import EXECUTION_LABEL, SeedError


PROJECT = "aura-benchmark-tests"
ANCHOR, NETWORK, POSTGRES, QDRANT, FOREIGN = (letter * 64 for letter in "abcde")
SERVICES = {"postgres": POSTGRES, "qdrant": QDRANT}


@pytest.fixture
def state():
    return lifecycle.Ownership(
        schema_version=1, execution_id="a" * 32, context="local", project=PROJECT,
        postgres_port=15432, qdrant_port=16333, phase="ready", anchor=ANCHOR,
        network=NETWORK, containers=SERVICES.copy(),
    )


@pytest.mark.parametrize("inventory", [
    ({FOREIGN}, set(), set()), (set(), {FOREIGN}, set()), (set(), set(), {"unrelated"}),
])
def test_start_refuses_conflicts_before_any_docker_creation(tmp_path, monkeypatch, inventory):
    daemon = Mock()
    monkeypatch.setattr(lifecycle, "docker", daemon)
    monkeypatch.setattr(lifecycle, "local_context", Mock())
    monkeypatch.setattr(lifecycle, "project_resources", Mock(return_value=inventory))
    target = tmp_path / "ownership.json"
    with pytest.raises(SeedError, match="Existing resources"):
        lifecycle.start(target, "local", PROJECT, 15432, 16333)
    daemon.assert_not_called()
    assert not target.exists()


def test_discovery_checks_names_and_owner_labels(monkeypatch, state):
    def listed(context, kind, action, *args):
        if "--format" in args:
            return {
                "container": f"{FOREIGN}\t{PROJECT}-outside\n",
                "network": f"{NETWORK}\t{PROJECT}_default\n",
                "volume": f"{PROJECT}_old\t{PROJECT}_old\n",
            }[kind]
        if kind == "container" and f"label={EXECUTION_LABEL}={state.execution_id}" in args:
            return ANCHOR + "\n"
        return ""

    monkeypatch.setattr(lifecycle, "docker", listed)
    assert lifecycle.project_resources(state) == ({FOREIGN, ANCHOR}, {NETWORK}, {f"{PROJECT}_old"})


def test_inherited_compose_configuration_cannot_redirect_start(monkeypatch, state):
    monkeypatch.setenv("COMPOSE_REMOVE_ORPHANS", "1")
    monkeypatch.setenv("COMPOSE_FILE", "application.yml")
    monkeypatch.setenv("AURA_BENCHMARK_POSTGRES_PORT", "5432")
    monkeypatch.setenv("AURA_BENCHMARK_POSTGRES_PASSWORD", "excluded")
    env = lifecycle.compose_environment(state)
    assert "COMPOSE_REMOVE_ORPHANS" not in env and "COMPOSE_FILE" not in env
    assert env["AURA_BENCHMARK_POSTGRES_PORT"] == "15432"
    assert len(env["AURA_BENCHMARK_POSTGRES_PASSWORD"]) == 64
    assert env["AURA_BENCHMARK_POSTGRES_PASSWORD"] != env["AURA_BENCHMARK_QDRANT_KEY"]
    assert env["AURA_BENCHMARK_NETWORK"] == state.network_name


def test_competing_lifecycle_stops_when_docker_name_is_reserved(tmp_path, monkeypatch):
    monkeypatch.setattr(lifecycle, "local_context", Mock())
    monkeypatch.setattr(lifecycle, "project_resources", Mock(return_value=(set(), set(), set())))
    monkeypatch.setattr(lifecycle, "verify_ports", Mock())
    daemon = Mock(side_effect=SeedError("Docker name already reserved"))
    monkeypatch.setattr(lifecycle, "docker", daemon)
    target = tmp_path / "other-owner.json"
    with pytest.raises(SeedError, match="already reserved"):
        lifecycle.start(target, "local", PROJECT, 15432, 16333)
    assert daemon.call_count == 1
    assert daemon.call_args.args[1:5] == ("container", "create", "--name", f"{PROJECT}-ownership")
    assert lifecycle.load_state(target).phase == "starting"


def test_same_state_path_has_an_exclusive_lifecycle_lock(tmp_path):
    target = tmp_path / "ownership.json"
    with lifecycle.state_lock(target):
        with pytest.raises(SeedError, match="Another lifecycle command"):
            with lifecycle.state_lock(target):
                pytest.fail("Acquired an already-held lifecycle lock")
    with lifecycle.state_lock(target) as reopened:
        assert reopened == target.resolve()


@pytest.mark.parametrize("defect", ["missing", "corrupt", "altered", "stopped"])
def test_teardown_refuses_invalid_or_stale_record(tmp_path, monkeypatch, state, defect):
    target = tmp_path / "ownership.json"
    if defect != "missing":
        if defect == "stopped":
            state.phase = "stopped"
        lifecycle.save_state(target, state, initial=True)
    if defect == "corrupt":
        target.write_text("{")
    elif defect == "altered":
        envelope = json.loads(target.read_text())
        envelope["record"]["postgres_port"] = 5432
        target.write_text(json.dumps(envelope))
    daemon = Mock()
    monkeypatch.setattr(lifecycle, "docker", daemon)
    with pytest.raises(SeedError, match="ownership record|already stopped"):
        lifecycle.stop(target)
    daemon.assert_not_called()


@pytest.fixture
def inspection(monkeypatch, state):
    labels = {lifecycle.PROJECT_LABEL: PROJECT, EXECUTION_LABEL: state.execution_id}
    records = {
        ANCHOR: {"Id": ANCHOR, "Config": {"Labels": {lifecycle.ANCHOR_LABEL: "1"}}},
        NETWORK: {
            "Id": NETWORK, "Name": state.network_name, "Driver": "bridge", "Scope": "local",
            "Labels": labels | {"com.docker.compose.network": "default"}, "Containers": {},
        },
    }
    for service, identifier in SERVICES.items():
        records[identifier] = {
            "Id": identifier, "State": {"Running": False},
            "Config": {"Labels": labels | {"com.docker.compose.service": service}},
        }
    monkeypatch.setattr(lifecycle, "validate_anchor", Mock(return_value=ANCHOR))
    monkeypatch.setattr(lifecycle, "validate_container", Mock())
    monkeypatch.setattr(lifecycle, "local_context", Mock())
    monkeypatch.setattr(lifecycle, "inspect_ids", lambda context, kind, ids: [records[item] for item in sorted(ids)])
    return records


@pytest.mark.parametrize("defect", ["container", "network", "volume", "attachment", "replacement"])
def test_teardown_preserves_unowned_or_replacement_resources(tmp_path, monkeypatch, state, inspection, defect):
    containers, networks, volumes = {ANCHOR, POSTGRES, QDRANT}, {NETWORK}, set()
    if defect == "container":
        containers.add(FOREIGN)
        inspection[FOREIGN] = {"Id": FOREIGN, "Config": {"Labels": {}}}
    elif defect == "network":
        networks.add(FOREIGN)
    elif defect == "volume":
        volumes.add("unrelated")
    elif defect == "attachment":
        inspection[NETWORK]["Containers"][FOREIGN] = {}
    else:
        containers.remove(QDRANT)
        containers.add(FOREIGN)
        inspection[FOREIGN] = inspection[QDRANT] | {"Id": FOREIGN}
    monkeypatch.setattr(lifecycle, "project_resources", Mock(return_value=(containers, networks, volumes)))
    daemon = Mock()
    monkeypatch.setattr(lifecycle, "docker", daemon)
    path = tmp_path / "ownership.json"
    lifecycle.save_state(path, state, initial=True)
    with pytest.raises(SeedError):
        lifecycle.stop(path)
    daemon.assert_not_called()
    assert lifecycle.load_state(path) == state


def test_partial_start_and_interrupted_teardown_resolve_only_recorded_ids(tmp_path, monkeypatch, state, inspection):
    state.phase = "starting"
    state.containers = {}
    path = tmp_path / "ownership.json"
    lifecycle.save_state(path, state, initial=True)
    containers, networks = {ANCHOR, POSTGRES}, {NETWORK}
    monkeypatch.setattr(lifecycle, "project_resources", lambda _: (set(containers), set(networks), set()))
    removed = []
    fail_network = True

    def remove(context, kind, action, identifier):
        if kind == "network" and fail_network:
            raise SeedError("interrupted network removal")
        assert action == "rm"
        removed.append(identifier)
        (containers if kind == "container" else networks).remove(identifier)

    monkeypatch.setattr(lifecycle, "docker", remove)
    with pytest.raises(SeedError, match="interrupted"):
        lifecycle.stop(path)
    assert lifecycle.load_state(path).containers == {"postgres": POSTGRES}
    assert removed == [POSTGRES] and ANCHOR in containers
    fail_network = False
    assert lifecycle.stop(path).phase == "stopped"
    assert removed == [POSTGRES, NETWORK, ANCHOR]


def test_normal_start_uses_no_recreate_and_starts_only_owned_ids(tmp_path, monkeypatch):
    daemon = Mock(side_effect=[ANCHOR, NETWORK, "", ""])
    monkeypatch.setattr(lifecycle, "docker", daemon)
    monkeypatch.setattr(lifecycle, "local_context", Mock())
    monkeypatch.setattr(lifecycle, "verify_ports", Mock())
    monkeypatch.setattr(lifecycle, "project_resources", Mock(return_value=(set(), set(), set())))
    monkeypatch.setattr(lifecycle, "owned_resources", Mock(side_effect=[
        (ANCHOR, None, {}), (ANCHOR, NETWORK, {}), (ANCHOR, NETWORK, SERVICES),
    ]))
    monkeypatch.setattr(lifecycle, "wait_for_services", Mock())
    path = tmp_path / "ownership.json"
    result = lifecycle.start(path, "local", PROJECT, 15432, 16333)
    assert result == lifecycle.load_state(path) and result.phase == "ready"
    assert daemon.call_args_list[2].args[-2:] == ("create", "--no-recreate")
    assert daemon.call_args_list[3].args[1:] == ("container", "start", POSTGRES, QDRANT)
    assert "PASSWORD" not in path.read_text() and "QDRANT_KEY" not in path.read_text()
