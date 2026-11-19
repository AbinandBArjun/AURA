import { useState } from "react";
import {
  AlertCircle,
  BookOpen,
  FileText,
  LoaderCircle,
  Search as SearchIcon,
  Sparkles,
} from "lucide-react";

import { searchKnowledge } from "../services/api";
import type { SearchResult } from "../types/api";

function getErrorMessage(error: unknown): string {
  return error instanceof Error
    ? error.message
    : "An unexpected error occurred.";
}

export default function Search() {
  const [query, setQuery] = useState("");
  const [limit, setLimit] = useState(5);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [searchedQuery, setSearchedQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [hasSearched, setHasSearched] = useState(false);
  const [duration, setDuration] = useState<number | null>(null);

  async function handleSearch(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const trimmedQuery = query.trim();

    if (!trimmedQuery || loading) return;

    setLoading(true);
    setError("");
    setHasSearched(false);
    setResults([]);

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
      setLoading(false);
    }
  }

  return (
    <section className="search-page">
      <div className="search-page-header">
        <div>
          <p className="eyebrow">RETRIEVAL INTELLIGENCE</p>
          <h1>Knowledge Search</h1>
          <p className="welcome-description">
            Search across your indexed research documents and discover
            relevant evidence.
          </p>
        </div>
      </div>

      <div className="knowledge-search-panel">
        <div className="search-panel-heading">
          <div className="search-panel-icon">
            <Sparkles size={20} />
          </div>
          <div>
            <h2>Search your knowledge base</h2>
            <p>
              Ask a question in natural language to retrieve relevant passages.
            </p>
          </div>
        </div>

        <form onSubmit={handleSearch}>
          <label className="search-input-label" htmlFor="knowledge-query">
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
              disabled={loading || !query.trim()}
            >
              {loading ? (
                <LoaderCircle size={16} className="spin" />
              ) : (
                <SearchIcon size={16} />
              )}
              {loading ? "Searching..." : "Search"}
            </button>
          </div>

          <div className="search-controls">
            <label htmlFor="search-limit">Maximum results</label>
            <select
              id="search-limit"
              value={limit}
              onChange={(event) => setLimit(Number(event.target.value))}
              disabled={loading}
            >
              <option value={5}>5 results</option>
              <option value={10}>10 results</option>
              <option value={15}>15 results</option>
              <option value={20}>20 results</option>
            </select>

            <span>Hybrid retrieval with reranking</span>
          </div>
        </form>
      </div>

      {error && (
        <div className="document-alert error-alert" role="alert">
          <AlertCircle size={17} />
          <span>{error}</span>
        </div>
      )}

      {loading && (
        <div className="search-state-panel">
          <LoaderCircle size={28} className="spin" />
          <h3>Searching your knowledge base</h3>
          <p>Retrieving and ranking relevant research passages...</p>
        </div>
      )}

      {!loading && !hasSearched && !error && (
        <div className="search-state-panel search-intro">
          <div className="search-state-icon">
            <BookOpen size={27} />
          </div>
          <h3>Find evidence across your documents</h3>
          <p>
            Enter a research question above to retrieve relevant passages
            from your processed and indexed PDFs.
          </p>
        </div>
      )}

      {!loading && hasSearched && (
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
                {results.length === 1 ? "passage found" : "passages found"}
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
    </section>
  );
}
