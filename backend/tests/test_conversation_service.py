from datetime import datetime, timedelta, timezone

import pytest

from app.models import Conversation, ConversationMessage
from app.services.conversation_service import (
    ConversationNotFoundError,
    ConversationService,
)


class FakeRAGService:
    def __init__(self, answer="Evidence-grounded answer.", evidence_count=2):
        self.answer_text = answer
        self.evidence_count = evidence_count
        self.questions = []
        self.workspace_ids = []

    def answer(self, workspace_id, question, limit):
        self.workspace_ids.append(workspace_id)
        self.questions.append((question, limit))

        return {
            "question": question,
            "answer": self.answer_text,
            "citations": [],
            "evidence_count": self.evidence_count,
            "claims": [],
        }


class FakeOllamaClient:
    def __init__(self, response="Resolved standalone question."):
        self.response = response
        self.prompts = []
        self.calls = 0

    def generate(self, prompt, system_prompt, temperature):
        self.calls += 1
        self.prompts.append(prompt)
        return self.response


def test_create_and_list_conversations(db_session, test_workspace):
    service = ConversationService(
        db=db_session,
        workspace_id=test_workspace.id,
        rag_service=FakeRAGService(),
        ollama_client=FakeOllamaClient(),
    )

    first = service.create_conversation("First discussion")
    second = service.create_conversation("Second discussion")

    conversations = service.list_conversations()

    assert first.id is not None
    assert second.id is not None
    assert len(conversations) == 2
    assert {item.title for item in conversations} == {
        "First discussion",
        "Second discussion",
    }
    assert all(
        item.workspace_id == test_workspace.id
        for item in conversations
    )


def test_get_conversation_returns_persisted_messages(
    db_session,
    test_workspace,
):
    service = ConversationService(
        db=db_session,
        workspace_id=test_workspace.id,
        rag_service=FakeRAGService(),
        ollama_client=FakeOllamaClient(),
    )

    conversation = service.create_conversation("History test")

    db_session.add_all([
        ConversationMessage(
            conversation_id=conversation.id,
            role="user",
            content="What is RAG?",
            citations=[],
        ),
        ConversationMessage(
            conversation_id=conversation.id,
            role="assistant",
            content="RAG combines retrieval and generation.",
            citations=[],
        ),
    ])
    db_session.commit()

    result = service.get_conversation(conversation.id)

    assert len(result.messages) == 2
    assert result.messages[0].role == "user"
    assert result.messages[1].role == "assistant"


def test_first_message_persists_user_and_assistant_turn(
    db_session,
    test_workspace,
):
    rag = FakeRAGService(
        answer="RAG combines retrieval and generation.",
        evidence_count=3,
    )
    ollama = FakeOllamaClient()

    service = ConversationService(
        db=db_session,
        workspace_id=test_workspace.id,
        rag_service=rag,
        ollama_client=ollama,
    )

    conversation = service.create_conversation()

    result = service.send_message(
        conversation_id=conversation.id,
        content="What is retrieval-augmented generation?",
        limit=7,
    )

    messages = service.get_conversation(conversation.id).messages

    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[0].content == (
        "What is retrieval-augmented generation?"
    )
    assert messages[1].role == "assistant"
    assert messages[1].content == (
        "RAG combines retrieval and generation."
    )

    assert conversation.title == (
        "What is retrieval-augmented generation?"
    )
    assert result["evidence_count"] == 3

    assert rag.questions == [
        ("What is retrieval-augmented generation?", 7)
    ]
    assert rag.workspace_ids == [test_workspace.id]

    # No history exists on the first turn, so question rewriting is skipped.
    assert ollama.calls == 0


def test_follow_up_question_uses_conversation_history(
    db_session,
    test_workspace,
):
    rag = FakeRAGService()
    ollama = FakeOllamaClient(
        response="Explain the advantages of retrieval-augmented generation."
    )

    service = ConversationService(
        db=db_session,
        workspace_id=test_workspace.id,
        rag_service=rag,
        ollama_client=ollama,
    )

    conversation = service.create_conversation()

    service.send_message(
        conversation_id=conversation.id,
        content="What is retrieval-augmented generation?",
    )

    service.send_message(
        conversation_id=conversation.id,
        content="What are its advantages?",
    )

    messages = service.get_conversation(conversation.id).messages

    assert ollama.calls == 1
    assert "What is retrieval-augmented generation?" in ollama.prompts[0]
    assert "What are its advantages?" in ollama.prompts[0]

    # The second RAG request receives the context-resolved question.
    assert rag.questions[1][0] == (
        "Explain the advantages of retrieval-augmented generation."
    )

    # The original user wording remains unchanged in persisted history.
    assert messages[2].content == "What are its advantages?"
    assert messages[2].role == "user"


def test_conversation_isolation_between_workspaces(
    db_session,
    test_workspace,
):
    conversation = Conversation(
        workspace_id=test_workspace.id,
        title="Private discussion",
    )
    db_session.add(conversation)
    db_session.commit()
    db_session.refresh(conversation)

    other_workspace_id = test_workspace.id + 10000

    service = ConversationService(
        db=db_session,
        workspace_id=other_workspace_id,
        rag_service=FakeRAGService(),
        ollama_client=FakeOllamaClient(),
    )

    with pytest.raises(ConversationNotFoundError):
        service.get_conversation(conversation.id)

    with pytest.raises(ConversationNotFoundError):
        service.send_message(
            conversation_id=conversation.id,
            content="Try to access this conversation.",
        )


def test_rag_failure_does_not_persist_partial_turn(
    db_session,
    test_workspace,
):
    class FailingRAGService:
        def answer(self, workspace_id, question, limit):
            raise RuntimeError("Simulated RAG failure.")

    service = ConversationService(
        db=db_session,
        workspace_id=test_workspace.id,
        rag_service=FailingRAGService(),
        ollama_client=FakeOllamaClient(),
    )

    conversation = service.create_conversation()

    with pytest.raises(RuntimeError, match="Simulated RAG failure"):
        service.send_message(
            conversation_id=conversation.id,
            content="A question that cannot be answered.",
        )

    refreshed = service.get_conversation(conversation.id)

    assert refreshed.messages == []
    assert refreshed.title == "New conversation"


def test_recent_history_returns_latest_eight_in_chronological_order(
    db_session,
    test_workspace,
):
    service = ConversationService(
        db=db_session,
        workspace_id=test_workspace.id,
        rag_service=FakeRAGService(),
        ollama_client=FakeOllamaClient(),
    )

    conversation = service.create_conversation("History limit test")

    base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)

    messages = [
        ConversationMessage(
            conversation_id=conversation.id,
            role="user" if index % 2 == 0 else "assistant",
            content=f"message-{index}",
            citations=[],
            created_at=base_time + timedelta(seconds=index),
        )
        for index in range(12)
    ]

    db_session.add_all(messages)
    db_session.commit()

    recent_history = service._get_recent_history(conversation.id)

    assert len(recent_history) == 8
    assert [message.content for message in recent_history] == [
        f"message-{index}"
        for index in range(4, 12)
    ]

