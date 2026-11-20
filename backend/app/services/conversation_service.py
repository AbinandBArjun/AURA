from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.llm.ollama_client import OllamaClient, OllamaError
from app.models import Conversation, ConversationMessage
from app.services.rag_service import RAGService


class ConversationNotFoundError(Exception):
    """Raised when a conversation does not belong to the active workspace."""


class ConversationService:

    HISTORY_LIMIT = 8

    def __init__(
        self,
        db: Session,
        workspace_id: int,
        rag_service: RAGService | None = None,
        ollama_client: OllamaClient | None = None,
    ):
        self.db = db
        self.workspace_id = workspace_id
        self.rag_service = rag_service or RAGService()
        self.ollama_client = ollama_client or OllamaClient()

    def create_conversation(
        self,
        title: str | None = None,
    ) -> Conversation:
        conversation = Conversation(
            workspace_id=self.workspace_id,
            title=title or "New conversation",
        )

        self.db.add(conversation)
        self.db.commit()
        self.db.refresh(conversation)

        return conversation

    def list_conversations(self) -> list[Conversation]:
        statement = (
            select(Conversation)
            .where(Conversation.workspace_id == self.workspace_id)
            .order_by(
                Conversation.updated_at.desc(),
                Conversation.id.desc(),
            )
        )

        return list(self.db.scalars(statement).all())

    def _get_conversation_record(
        self,
        conversation_id: int,
    ) -> Conversation:
        statement = select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.workspace_id == self.workspace_id,
        )

        conversation = self.db.scalar(statement)

        if conversation is None:
            raise ConversationNotFoundError(
                "Conversation not found."
            )

        return conversation

    def _get_recent_history(
        self,
        conversation_id: int,
    ) -> list[ConversationMessage]:
        statement = (
            select(ConversationMessage)
            .where(
                ConversationMessage.conversation_id == conversation_id
            )
            .order_by(
                ConversationMessage.created_at.desc(),
                ConversationMessage.id.desc(),
            )
            .limit(self.HISTORY_LIMIT)
        )

        recent_messages = list(self.db.scalars(statement).all())

        return list(reversed(recent_messages))

    def get_conversation(
        self,
        conversation_id: int,
    ) -> Conversation:
        statement = (
            select(Conversation)
            .options(selectinload(Conversation.messages))
            .where(
                Conversation.id == conversation_id,
                Conversation.workspace_id == self.workspace_id,
            )
        )

        conversation = self.db.scalar(statement)

        if conversation is None:
            raise ConversationNotFoundError(
                "Conversation not found."
            )

        return conversation

    def _resolve_question(
        self,
        history: list[ConversationMessage],
        question: str,
    ) -> str:
        recent_history = history[-self.HISTORY_LIMIT:]

        if not recent_history:
            return question

        history_text = "\n".join(
            f"{message.role.upper()}: {message.content}"
            for message in recent_history
        )

        prompt = f"""
Conversation history:
{history_text}

New user question:
{question}

Rewrite the new user question as a standalone question that can
be understood without the conversation history.

Instructions:
- Resolve references using the conversation history.
- Preserve the user's original intent.
- Do not answer the question.
- Do not introduce facts or assumptions.
- If the question is already standalone, keep its meaning unchanged.
- Return only the rewritten question.
"""

        try:
            resolved = self.ollama_client.generate(
                prompt=prompt,
                system_prompt=(
                    "You are a precise conversational question resolver. "
                    "Your task is only to rewrite questions using context."
                ),
                temperature=0.0,
            ).strip()

            if resolved and len(resolved) <= 2000:
                return resolved

        except OllamaError:
            pass

        return question

    def send_message(
        self,
        conversation_id: int,
        content: str,
        limit: int = 5,
    ) -> dict:
        conversation = self._get_conversation_record(
            conversation_id
        )

        recent_history = self._get_recent_history(
            conversation_id
        )

        resolved_question = self._resolve_question(
            history=recent_history,
            question=content,
        )

        result = self.rag_service.answer(
            workspace_id=self.workspace_id,
            question=resolved_question,
            limit=limit,
        )

        citations = [
            citation.model_dump()
            if hasattr(citation, "model_dump")
            else citation
            for citation in result["citations"]
        ]

        user_message = ConversationMessage(
            conversation_id=conversation.id,
            role="user",
            content=content,
            citations=[],
        )

        assistant_message = ConversationMessage(
            conversation_id=conversation.id,
            role="assistant",
            content=result["answer"],
            citations=citations,
        )

        if conversation.title == "New conversation":
            conversation.title = content[:200]

        conversation.updated_at = datetime.now(timezone.utc)

        self.db.add_all([user_message, assistant_message])

        try:
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

        self.db.refresh(user_message)
        self.db.refresh(assistant_message)

        return {
            "conversation_id": conversation.id,
            "user_message": user_message,
            "assistant_message": assistant_message,
            "evidence_count": result["evidence_count"],
        }
