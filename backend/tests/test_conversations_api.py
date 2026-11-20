from datetime import datetime, timezone
from types import SimpleNamespace

from app.api.routes import conversations as conversation_route


NOW = datetime.now(timezone.utc)


def make_conversation(
    conversation_id=1,
    workspace_id=1,
    title="New conversation",
    messages=None,
):
    return SimpleNamespace(
        id=conversation_id,
        workspace_id=workspace_id,
        title=title,
        created_at=NOW,
        updated_at=NOW,
        messages=messages or [],
    )


def make_message(
    message_id,
    conversation_id,
    role,
    content,
    citations=None,
):
    return SimpleNamespace(
        id=message_id,
        conversation_id=conversation_id,
        role=role,
        content=content,
        citations=citations or [],
        created_at=NOW,
    )


def test_create_conversation_uses_server_workspace(
    client,
    current_workspace,
    monkeypatch,
):
    captured = {}

    class FakeConversationService:
        def __init__(self, db, workspace_id):
            captured["workspace_id"] = workspace_id

        def create_conversation(self, title=None):
            captured["title"] = title

            return make_conversation(
                workspace_id=current_workspace.id,
                title=title or "New conversation",
            )

    monkeypatch.setattr(
        conversation_route,
        "ConversationService",
        FakeConversationService,
    )

    response = client.post(
        "/api/v1/conversations",
        json={"title": "Research discussion"},
    )

    assert response.status_code == 201
    assert response.json()["title"] == "Research discussion"
    assert captured["workspace_id"] == current_workspace.id
    assert captured["title"] == "Research discussion"


def test_list_conversations_uses_server_workspace(
    client,
    current_workspace,
    monkeypatch,
):
    captured = {}

    class FakeConversationService:
        def __init__(self, db, workspace_id):
            captured["workspace_id"] = workspace_id

        def list_conversations(self):
            return [
                make_conversation(
                    conversation_id=1,
                    workspace_id=current_workspace.id,
                    title="First discussion",
                ),
                make_conversation(
                    conversation_id=2,
                    workspace_id=current_workspace.id,
                    title="Second discussion",
                ),
            ]

    monkeypatch.setattr(
        conversation_route,
        "ConversationService",
        FakeConversationService,
    )

    response = client.get("/api/v1/conversations")

    assert response.status_code == 200
    assert len(response.json()) == 2
    assert captured["workspace_id"] == current_workspace.id


def test_get_conversation_returns_message_history(
    client,
    current_workspace,
    monkeypatch,
):
    user_message = make_message(
        1,
        1,
        "user",
        "Explain retrieval-augmented generation.",
    )

    assistant_message = make_message(
        2,
        1,
        "assistant",
        "RAG combines retrieval and generation.",
    )

    conversation = make_conversation(
        conversation_id=1,
        workspace_id=current_workspace.id,
        title="RAG discussion",
        messages=[user_message, assistant_message],
    )

    class FakeConversationService:
        def __init__(self, db, workspace_id):
            assert workspace_id == current_workspace.id

        def get_conversation(self, conversation_id):
            assert conversation_id == 1
            return conversation

    monkeypatch.setattr(
        conversation_route,
        "ConversationService",
        FakeConversationService,
    )

    response = client.get("/api/v1/conversations/1")

    assert response.status_code == 200
    data = response.json()

    assert data["title"] == "RAG discussion"
    assert len(data["messages"]) == 2
    assert data["messages"][0]["role"] == "user"
    assert data["messages"][1]["role"] == "assistant"


def test_get_conversation_returns_404_when_not_found(
    client,
    current_workspace,
    monkeypatch,
):
    class FakeConversationService:
        def __init__(self, db, workspace_id):
            pass

        def get_conversation(self, conversation_id):
            raise conversation_route.ConversationNotFoundError(
                "Conversation not found."
            )

    monkeypatch.setattr(
        conversation_route,
        "ConversationService",
        FakeConversationService,
    )

    response = client.get("/api/v1/conversations/999")

    assert response.status_code == 404
    assert response.json()["detail"] == "Conversation not found."


def test_send_message_returns_persisted_turn(
    client,
    current_workspace,
    monkeypatch,
):
    captured = {}

    user_message = make_message(
        1,
        1,
        "user",
        "What is RAG?",
    )

    assistant_message = make_message(
        2,
        1,
        "assistant",
        "RAG combines retrieval and generation.",
    )

    class FakeConversationService:
        def __init__(self, db, workspace_id):
            captured["workspace_id"] = workspace_id

        def send_message(self, conversation_id, content, limit):
            captured["conversation_id"] = conversation_id
            captured["content"] = content
            captured["limit"] = limit

            return {
                "conversation_id": conversation_id,
                "user_message": user_message,
                "assistant_message": assistant_message,
                "evidence_count": 3,
            }

    monkeypatch.setattr(
        conversation_route,
        "ConversationService",
        FakeConversationService,
    )

    response = client.post(
        "/api/v1/conversations/1/messages",
        json={
            "content": "What is RAG?",
            "limit": 7,
        },
    )

    assert response.status_code == 201
    data = response.json()

    assert data["conversation_id"] == 1
    assert data["user_message"]["content"] == "What is RAG?"
    assert data["assistant_message"]["role"] == "assistant"
    assert data["evidence_count"] == 3

    assert captured["workspace_id"] == current_workspace.id
    assert captured["conversation_id"] == 1
    assert captured["content"] == "What is RAG?"
    assert captured["limit"] == 7


def test_send_message_returns_404_for_other_workspace_conversation(
    client,
    current_workspace,
    monkeypatch,
):
    class FakeConversationService:
        def __init__(self, db, workspace_id):
            assert workspace_id == current_workspace.id

        def send_message(self, conversation_id, content, limit):
            raise conversation_route.ConversationNotFoundError(
                "Conversation not found."
            )

    monkeypatch.setattr(
        conversation_route,
        "ConversationService",
        FakeConversationService,
    )

    response = client.post(
        "/api/v1/conversations/999/messages",
        json={"content": "Tell me more."},
    )

    assert response.status_code == 404


def test_conversation_endpoints_reject_client_workspace_id(
    client,
    current_workspace,
):
    response = client.post(
        "/api/v1/conversations",
        json={
            "title": "Invalid conversation",
            "workspace_id": current_workspace.id,
        },
    )

    assert response.status_code == 422


def test_send_message_rejects_invalid_limit(
    client,
    current_workspace,
):
    response = client.post(
        "/api/v1/conversations/1/messages",
        json={
            "content": "Explain RAG.",
            "limit": 25,
        },
    )

    assert response.status_code == 422
