from sqlalchemy import select, func

from app.core.config import settings
from app.models import User, Workspace


def test_current_workspace_creates_user_and_workspace(
    client,
    db_session,
    monkeypatch,
):
    monkeypatch.setattr(settings, "environment", "development")

    response = client.get("/api/v1/workspaces/current")

    assert response.status_code == 200

    result = response.json()

    assert result["name"] == settings.development_workspace_name
    assert result["owner_id"] > 0
    assert result["id"] > 0

    user_count = db_session.scalar(
        select(func.count()).select_from(User)
    )
    workspace_count = db_session.scalar(
        select(func.count()).select_from(Workspace)
    )

    assert user_count == 1
    assert workspace_count == 1


def test_current_workspace_is_idempotent(
    client,
    db_session,
    monkeypatch,
):
    monkeypatch.setattr(settings, "environment", "development")

    first_response = client.get("/api/v1/workspaces/current")
    second_response = client.get("/api/v1/workspaces/current")

    assert first_response.status_code == 200
    assert second_response.status_code == 200

    assert first_response.json()["id"] == second_response.json()["id"]

    user_count = db_session.scalar(
        select(func.count()).select_from(User)
    )
    workspace_count = db_session.scalar(
        select(func.count()).select_from(Workspace)
    )

    assert user_count == 1
    assert workspace_count == 1


def test_current_workspace_rejects_non_development_environment(
    client,
    monkeypatch,
):
    monkeypatch.setattr(settings, "environment", "production")

    response = client.get("/api/v1/workspaces/current")

    assert response.status_code == 404
    assert response.json()["detail"] == (
        "Workspace bootstrap is only available in development."
    )
