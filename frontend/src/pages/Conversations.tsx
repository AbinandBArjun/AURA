import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import {
  AlertCircle,
  FileText,
  LoaderCircle,
  MessageSquare,
  MessageSquarePlus,
  Send,
  Sparkles,
} from "lucide-react";

import {
  createConversation,
  getConversation,
  getConversations,
  sendConversationMessage,
} from "../services/api";

import type {
  ConversationDetailResponse,
  ConversationResponse,
} from "../types/api";

function getErrorMessage(error: unknown): string {
  return error instanceof Error
    ? error.message
    : "An unexpected error occurred.";
}

function formatDate(value: string): string {
  return new Date(value).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
}

export default function Conversations() {
  const [conversations, setConversations] = useState<
    ConversationResponse[]
  >([]);

  const [activeConversation, setActiveConversation] =
    useState<ConversationDetailResponse | null>(null);

  const [draft, setDraft] = useState("");
  const [loadingList, setLoadingList] = useState(true);
  const [loadingConversation, setLoadingConversation] =
    useState(false);
  const [creating, setCreating] = useState(false);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function loadInitialConversations() {
      try {
        const items = await getConversations();

        if (cancelled) return;

        setConversations(items);

        if (items.length > 0) {
          const detail = await getConversation(items[0].id);

          if (!cancelled) {
            setActiveConversation(detail);
          }
        }
      } catch (err) {
        if (!cancelled) {
          setError(getErrorMessage(err));
        }
      } finally {
        if (!cancelled) {
          setLoadingList(false);
        }
      }
    }

    void loadInitialConversations();

    return () => {
      cancelled = true;
    };
  }, []);

  async function handleCreateConversation() {
    if (creating || sending) return;

    setCreating(true);
    setError("");

    try {
      const conversation = await createConversation();

      setConversations((current) => [
        conversation,
        ...current,
      ]);

      setActiveConversation({
        ...conversation,
        messages: [],
      });

      setDraft("");
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setCreating(false);
    }
  }

  async function handleSelectConversation(
    conversationId: number
  ) {
    if (
      activeConversation?.id === conversationId ||
      loadingConversation ||
      sending
    ) {
      return;
    }

    setLoadingConversation(true);
    setError("");

    try {
      const detail = await getConversation(conversationId);
      setActiveConversation(detail);
      setDraft("");
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setLoadingConversation(false);
    }
  }

  async function handleSendMessage(
    event: FormEvent<HTMLFormElement>
  ) {
    event.preventDefault();

    const content = draft.trim();

    if (!content || !activeConversation || sending) return;

    const conversationId = activeConversation.id;

    setSending(true);
    setError("");
    setDraft("");

    try {
      const response = await sendConversationMessage(
        conversationId,
        content,
        5
      );

      setActiveConversation((current) => {
        if (!current || current.id !== response.conversation_id) {
          return current;
        }

        return {
          ...current,
          messages: [
            ...current.messages,
            response.user_message,
            response.assistant_message,
          ],
        };
      });

      // Refresh the sidebar independently. A refresh failure must not
      // make a successfully submitted message appear to have failed.
      try {
        const updatedConversations = await getConversations();
        setConversations(updatedConversations);

        const updatedActive = updatedConversations.find(
          (conversation) => conversation.id === conversationId
        );

        if (updatedActive) {
          setActiveConversation((current) => {
            if (!current || current.id !== conversationId) {
              return current;
            }

            return {
              ...current,
              title: updatedActive.title,
              updated_at: updatedActive.updated_at,
            };
          });
        }
      } catch (refreshError) {
        setError(
          `Your message was sent, but the conversation list could not be refreshed: ${getErrorMessage(refreshError)}`
        );
      }
    } catch (err) {
      setDraft(content);
      setError(getErrorMessage(err));
    } finally {
      setSending(false);
    }
  }

  const canSend =
    Boolean(draft.trim()) &&
    Boolean(activeConversation) &&
    !sending &&
    !loadingConversation;

  return (
    <section className="conversations-page">
      <div className="search-page-header">
        <div>
          <p className="eyebrow">MULTI-TURN RESEARCH</p>
          <h1>Conversations</h1>
          <p className="welcome-description">
            Explore your research through persistent, evidence-grounded
            AI conversations.
          </p>
        </div>
      </div>

      {error && (
        <div className="document-alert error-alert" role="alert">
          <AlertCircle size={17} />
          <span>{error}</span>
        </div>
      )}

      <div className="conversation-workspace">
        <aside className="conversation-sidebar">
          <div className="conversation-sidebar-header">
            <div>
              <p className="eyebrow">YOUR HISTORY</p>
              <h2>Conversations</h2>
            </div>

            <button
              type="button"
              className="conversation-new-button"
              onClick={handleCreateConversation}
              disabled={creating || sending}
              aria-label="Create new conversation"
              title="New conversation"
            >
              {creating ? (
                <LoaderCircle size={18} className="spin" />
              ) : (
                <MessageSquarePlus size={18} />
              )}
            </button>
          </div>

          <button
            type="button"
            className="conversation-create-action"
            onClick={handleCreateConversation}
            disabled={creating || sending}
          >
            <span>
              <MessageSquarePlus size={16} />
              New conversation
            </span>
            <span>+</span>
          </button>

          <div className="conversation-list">
            {loadingList ? (
              <div className="conversation-list-state">
                <LoaderCircle size={19} className="spin" />
                <span>Loading history...</span>
              </div>
            ) : conversations.length === 0 ? (
              <div className="conversation-list-state">
                <MessageSquare size={20} />
                <span>No conversations yet.</span>
              </div>
            ) : (
              conversations.map((conversation) => (
                <button
                  type="button"
                  key={conversation.id}
                  className={`conversation-list-item ${
                    activeConversation?.id === conversation.id
                      ? "conversation-list-item-active"
                      : ""
                  }`}
                  onClick={() =>
                    void handleSelectConversation(conversation.id)
                  }
                  disabled={loadingConversation || sending}
                >
                  <MessageSquare size={16} />

                  <span className="conversation-list-item-content">
                    <strong>{conversation.title}</strong>
                    <small>{formatDate(conversation.updated_at)}</small>
                  </span>
                </button>
              ))
            )}
          </div>

          <div className="conversation-sidebar-footer">
            <Sparkles size={15} />
            <span>Answers grounded in your workspace evidence</span>
          </div>
        </aside>

        <section className="conversation-chat-panel">
          {activeConversation ? (
            <>
              <header className="conversation-chat-header">
                <div className="conversation-chat-heading-icon">
                  <Sparkles size={19} />
                </div>

                <div>
                  <h2>{activeConversation.title}</h2>
                  <p>
                    Evidence-grounded research assistant
                  </p>
                </div>

                <span className="conversation-status">
                  <span />
                  Workspace AI
                </span>
              </header>

              <div className="conversation-messages">
                {loadingConversation ? (
                  <div className="conversation-empty-state">
                    <LoaderCircle size={27} className="spin" />
                    <h3>Loading conversation</h3>
                    <p>Retrieving your saved message history...</p>
                  </div>
                ) : activeConversation.messages.length === 0 ? (
                  <div className="conversation-empty-state">
                    <div className="conversation-empty-icon">
                      <Sparkles size={27} />
                    </div>
                    <h3>Start your research conversation</h3>
                    <p>
                      Ask a question about your indexed documents.
                      Follow-up questions can use the context of
                      earlier messages.
                    </p>
                  </div>
                ) : (
                  activeConversation.messages.map((message) => (
                    <article
                      key={message.id}
                      className={`conversation-message conversation-message-${message.role}`}
                    >
                      <div className="conversation-message-avatar">
                        {message.role === "assistant" ? (
                          <Sparkles size={17} />
                        ) : (
                          <span>Y</span>
                        )}
                      </div>

                      <div className="conversation-message-body">
                        <div className="conversation-message-meta">
                          <strong>
                            {message.role === "assistant"
                              ? "AURA Assistant"
                              : "You"}
                          </strong>
                          <time>
                            {new Date(
                              message.created_at
                            ).toLocaleTimeString([], {
                              hour: "2-digit",
                              minute: "2-digit",
                            })}
                          </time>
                        </div>

                        <p className="conversation-message-text">
                          {message.content}
                        </p>

                        {message.role === "assistant" && (
                          <div className="conversation-message-citations">
                            <div className="conversation-citation-heading">
                              <FileText size={14} />
                              <strong>
                                Evidence citations ({message.citations.length})
                              </strong>
                            </div>

                            {message.citations.length > 0 ? (
                              <div className="conversation-citation-list">
                                {message.citations.map((citation) => (
                                  <div
                                    className="conversation-citation"
                                    key={`${citation.source_id}-${citation.chunk_id}`}
                                  >
                                    <span className="conversation-citation-id">
                                      [{citation.source_id}]
                                    </span>

                                    <div>
                                      <strong>{citation.filename}</strong>
                                      <small>
                                        Pages {citation.page_start}
                                        {citation.page_end !==
                                        citation.page_start
                                          ? `-${citation.page_end}`
                                          : ""}
                                      </small>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            ) : (
                              <p className="conversation-no-citations">
                                No citations returned for this response.
                              </p>
                            )}
                          </div>
                        )}
                      </div>
                    </article>
                  ))
                )}

                {sending && (
                  <div className="conversation-thinking">
                    <LoaderCircle size={18} className="spin" />
                    <span>
                      Retrieving evidence and generating a verified answer...
                    </span>
                  </div>
                )}
              </div>

              <form
                className="conversation-composer"
                onSubmit={handleSendMessage}
              >
                <label htmlFor="conversation-message-input">
                  YOUR MESSAGE
                </label>

                <div className="conversation-composer-input">
                  <textarea
                    id="conversation-message-input"
                    value={draft}
                    onChange={(event) => setDraft(event.target.value)}
                    placeholder="Ask a research question or follow up..."
                    maxLength={2000}
                    rows={3}
                    disabled={sending || loadingConversation}
                  />

                  <button
                    type="submit"
                    className="primary-button"
                    disabled={!canSend}
                  >
                    {sending ? (
                      <LoaderCircle size={17} className="spin" />
                    ) : (
                      <Send size={16} />
                    )}
                    {sending ? "Thinking..." : "Send"}
                  </button>
                </div>

                <div className="conversation-composer-footer">
                  <span>
                    Responses use fresh document evidence and validated citations.
                  </span>
                  <span>{draft.length}/2000</span>
                </div>
              </form>
            </>
          ) : (
            <div className="conversation-empty-state conversation-no-selection">
              <div className="conversation-empty-icon">
                <MessageSquare size={27} />
              </div>
              <h3>No conversation selected</h3>
              <p>
                Create a conversation to start asking questions
                about your research documents.
              </p>
              <button
                type="button"
                className="primary-button"
                onClick={handleCreateConversation}
                disabled={creating || sending}
              >
                <MessageSquarePlus size={16} />
                Start a conversation
              </button>
            </div>
          )}
        </section>
      </div>
    </section>
  );
}
