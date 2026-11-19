import { useState } from "react";
import {
  AlertCircle,
  BookOpen,
  FileText,
  LoaderCircle,
  Search as SearchIcon,
  Sparkles,
} from "lucide-react";

import {
  generateAnswer,
  searchKnowledge,
} from "../services/api";

import type {
  RAGResponse,
  SearchResult,
} from "../types/api";

type Operation = "search" | "answer" | null;

function getErrorMessage(error: unknown): string {
  return error instanceof Error
    ? error.message
    : "An unexpected error occurred.";
}

function getVerificationLabel(status: string): string {
  switch (status) {
    case "supported":
      return "Supported";
    case "partially_supported":
      return "Partially supported";
    case "contradicted":
      return "Contradicted";
    default:
      return "Insufficient evidence";
  }
}

export default function Search() {
  const [query, setQuery] = useState("");
  const [limit, setLimit] = useState(5);

  const [results, setResults] = useState<SearchResult[]>([]);
  const [answer, setAnswer] = useState<RAGResponse | null>(null);

  const [searchedQuery, setSearchedQuery] = useState("");
  const [loading, setLoading] = useState<Operation>(null);
  const [error, setError] = useState("");
  const [hasSearched, setHasSearched] = useState(false);
  const [duration, setDuration] = useState<number | null>(null);

  async function handleSearch(
    event: React.FormEvent<HTMLFormElement>
  ) {
    event.preventDefault();

    const trimmedQuery = query.trim();

    if (!trimmedQuery || loading) return;

    setLoading("search");
    setError("");
    setAnswer(null);
    setResults([]);
    setHasSearched(false);

    const startTime = performance.now();

    try {
      const response = await searchKnowledge(trimmedQuery, limit);

      setResults(response.results);
      setSearchedQuery(response.query);
      setDuration(performance.now() - startTime);
      setHasSearched(true);
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setLoading(null);
    }
  }

  async function handleGenerateAnswer() {
    const trimmedQuery = query.trim();

    if (!trimmedQuery || loading) return;

    setLoading("answer");
    setError("");
    setAnswer(null);
    setResults([]);
    setHasSearched(false);

    const startTime = performance.now();

    try {
      const response = await generateAnswer(trimmedQuery, limit);

      setAnswer(response);
      setSearchedQuery(response.question);
      setDuration(performance.now() - startTime);
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setLoading(null);
    }
  }

  const isLoading = loading !== null;

  return (
    <section className="search-page">
      <div className="search-page-header">
        <div>
          <p className="eyebrow">RETRIEVAL INTELLIGENCE</p>
          <h1>Knowledge Search</h1>
          <p className="welcome-description">
            Search your research documents or generate
            evidence-grounded answers using AI.
          </p>
        </div>
      </div>

      <div className="knowledge-search-panel">
        <div className="search-panel-heading">
          <div className="search-panel-icon">
            <Sparkles size={20} />
          </div>

          <div>
            <h2>Ask your knowledge base</h2>
            <p>
              Retrieve relevant passages or generate an answer
              grounded in your indexed research.
            </p>
          </div>
        </div>

        <form onSubmit={handleSearch}>
          <label
            className="search-input-label"
            htmlFor="knowledge-query"
          >
            YOUR RESEARCH QUESTION
          </label>

          <div className="knowledge-search-input">
            <SearchIcon size={19} />

            <input
              id="knowledge-query"
              type="text"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="e.g. What are the key findings about retrieval augmented generation?"
              maxLength={2000}
              required
            />

            <button
              type="submit"
              className="primary-button"
              disabled={isLoading || !query.trim()}
            >
              {loading === "search" ? (
                <LoaderCircle size={16} className="spin" />
              ) : (
                <SearchIcon size={16} />
              )}

              {loading === "search" ? "Searching..." : "Search"}
            </button>

            <button
              type="button"
              className="generate-answer-button"
              onClick={handleGenerateAnswer}
              disabled={isLoading || !query.trim()}
            >
              {loading === "answer" ? (
                <LoaderCircle size={16} className="spin" />
              ) : (
                <Sparkles size={16} />
              )}

              {loading === "answer"
                ? "Generating..."
                : "Generate Answer"}
            </button>
          </div>

          <div className="search-controls">
            <label htmlFor="search-limit">
              Maximum evidence passages
            </label>

            <select
              id="search-limit"
              value={limit}
              onChange={(event) =>
                setLimit(Number(event.target.value))
              }
              disabled={isLoading}
            >
              <option value={5}>5 passages</option>
              <option value={10}>10 passages</option>
              <option value={15}>15 passages</option>
              <option value={20}>20 passages</option>
            </select>

            <span>Hybrid retrieval with reranking</span>
          </div>
        </form>
      </div>

      {error && (
        <div
          className="document-alert error-alert"
          role="alert"
        >
          <AlertCircle size={17} />
          <span>{error}</span>
        </div>
      )}

      {isLoading && (
        <div className="search-state-panel">
          <LoaderCircle size={28} className="spin" />

          <h3>
            {loading === "answer"
              ? "Generating an evidence-grounded answer"
              : "Searching your knowledge base"}
          </h3>

          <p>
            {loading === "answer"
              ? "Retrieving evidence, generating a response, and verifying claims..."
              : "Retrieving and ranking relevant research passages..."}
          </p>
        </div>
      )}

      {!isLoading && !hasSearched && !answer && !error && (
        <div className="search-state-panel search-intro">
          <div className="search-state-icon">
            <BookOpen size={27} />
          </div>

          <h3>Explore your research with AI</h3>

          <p>
            Search for relevant passages or generate an answer
            supported by document evidence and claim verification.
          </p>
        </div>
      )}

      {!isLoading && hasSearched && (
        <div className="search-results-section">
          <div className="search-results-heading">
            <div>
              <p className="eyebrow">RETRIEVAL RESULTS</p>
              <h2>Search results</h2>
              <p>
                Results for <strong>"{searchedQuery}"</strong>
              </p>
            </div>

            <div className="search-result-summary">
              <strong>{results.length}</strong>
              <span>
                {results.length === 1
                  ? "passage found"
                  : "passages found"}
              </span>
            </div>
          </div>

          {results.length === 0 ? (
            <div className="search-state-panel">
              <SearchIcon size={27} />
              <h3>No matching passages found</h3>
              <p>
                Try a different query or ensure that your documents
                have been processed and indexed.
              </p>
            </div>
          ) : (
            <div className="search-results-list">
              {results.map((result, index) => (
                <article
                  className="search-result-card"
                  key={`${result.document_id}-${result.chunk_id}`}
                >
                  <div className="search-result-top">
                    <span className="search-rank">
                      #{index + 1}
                    </span>

                    <span className="search-score">
                      Reranker: {result.score.toFixed(4)}
                    </span>
                  </div>

                  <div className="search-result-document">
                    <FileText size={17} />
                    <strong>{result.filename}</strong>

                    <span>
                      Pages {result.page_start}
                      {result.page_end !== result.page_start
                        ? `-${result.page_end}`
                        : ""}
                    </span>
                  </div>

                  <p className="search-result-text">
                    {result.text}
                  </p>

                  <div className="search-result-footer">
                    <span>Document ID: {result.document_id}</span>
                    <span>Chunk ID: {result.chunk_id}</span>
                  </div>
                </article>
              ))}
            </div>
          )}

          {duration !== null && (
            <p className="search-duration">
              Retrieved in {(duration / 1000).toFixed(2)} seconds
            </p>
          )}
        </div>
      )}

      {!isLoading && answer && (
        <div className="search-results-section">
          <div className="search-results-heading">
            <div>
              <p className="eyebrow">AI RESEARCH RESPONSE</p>
              <h2>Generated answer</h2>
              <p>
                Answer for <strong>"{searchedQuery}"</strong>
              </p>
            </div>

            <div className="search-result-summary">
              <strong>{answer.evidence_count}</strong>
              <span>evidence passages</span>
            </div>
          </div>

          <article className="generated-answer-card">
            <div className="generated-answer-heading">
              <Sparkles size={19} />
              <h3>Evidence-grounded response</h3>
            </div>

            <p className="generated-answer-text">
              {answer.answer}
            </p>

            <div className="generated-answer-meta">
              <span>
                {answer.citations.length} source citations
              </span>
              <span>
                {answer.claims.length} verified claims
              </span>
            </div>
          </article>

          <section className="rag-section">
            <div className="rag-section-heading">
              <div>
                <p className="eyebrow">SOURCE ATTRIBUTION</p>
                <h3>Evidence citations</h3>
              </div>

              <span>{answer.citations.length} sources</span>
            </div>

            {answer.citations.length === 0 ? (
              <div className="rag-empty-state">
                No source citations were returned.
              </div>
            ) : (
              <div className="rag-citations-list">
                {answer.citations.map((citation) => (
                  <article
                    className="rag-citation-card"
                    key={`${citation.document_id}-${citation.chunk_id}`}
                  >
                    <div className="rag-citation-id">
                      [{citation.source_id}]
                    </div>

                    <div className="rag-citation-content">
                      <strong>{citation.filename}</strong>

                      <p>
                        Pages {citation.page_start}
                        {citation.page_end !== citation.page_start
                          ? `-${citation.page_end}`
                          : ""}
                      </p>

                      <div className="rag-citation-meta">
                        <span>
                          Document #{citation.document_id}
                        </span>
                        <span>
                          Chunk #{citation.chunk_id}
                        </span>
                        <span>
                          Score: {citation.score.toFixed(4)}
                        </span>
                      </div>
                    </div>
                  </article>
                ))}
              </div>
            )}
          </section>

          <section className="rag-section">
            <div className="rag-section-heading">
              <div>
                <p className="eyebrow">SEMANTIC VALIDATION</p>
                <h3>Claim verification</h3>
              </div>

              <span>{answer.claims.length} claims</span>
            </div>

            {answer.claims.length === 0 ? (
              <div className="rag-empty-state">
                No individual claims were returned for verification.
              </div>
            ) : (
              <div className="rag-claims-list">
                {answer.claims.map((claim) => (
                  <article
                    className="rag-claim-card"
                    key={claim.claim_id}
                  >
                    <div className="rag-claim-top">
                      <span className="rag-claim-id">
                        {claim.claim_id}
                      </span>

                      <span
                        className={`rag-verification-status status-${claim.status}`}
                      >
                        {getVerificationLabel(claim.status)}
                      </span>
                    </div>

                    <p className="rag-claim-text">
                      {claim.claim_text}
                    </p>

                    <p className="rag-claim-rationale">
                      {claim.rationale}
                    </p>

                    {claim.evidence_ids.length > 0 && (
                      <div className="rag-claim-evidence">
                        <span>Evidence:</span>

                        {claim.evidence_ids.map((evidenceId) => (
                          <span
                            className="rag-evidence-tag"
                            key={evidenceId}
                          >
                            [{evidenceId}]
                          </span>
                        ))}
                      </div>
                    )}
                  </article>
                ))}
              </div>
            )}
          </section>

          {duration !== null && (
            <p className="search-duration">
              Generated and verified in {(duration / 1000).toFixed(2)} seconds
            </p>
          )}
        </div>
      )}
    </section>
  );
}
