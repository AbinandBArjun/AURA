import { useEffect, useState } from "react";

type Health = { status: string; service: string };

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch("http://localhost:8000/api/v1/health")
      .then((r) => {
        if (!r.ok) throw new Error("Backend returned an error");
        return r.json();
      })
      .then(setHealth)
      .catch((e: Error) => setError(e.message));
  }, []);

  return (
    <main className="shell">
      <header className="topbar"><span className="brand">AURA</span><span>Research Intelligence</span></header>
      <section className="hero">
        <p className="eyebrow">AUTONOMOUS RESEARCH PLATFORM</p>
        <h1>Turn information into<br />evidence-backed intelligence.</h1>
        <p className="intro">Upload research documents, investigate questions and produce reports with traceable sources.</p>
        <div className="status">
          <strong>Backend status</strong>
          {health ? <span className="ok">● {health.status}</span> : error ? <span className="bad">● {error}</span> : <span>Checking…</span>}
        </div>
      </section>
      <section className="grid">
        <article><h2>Document workspace</h2><p>Document ingestion and indexing will be implemented in the next milestone.</p></article>
        <article><h2>Research runs</h2><p>LangGraph planning, evidence retrieval and report generation will appear here.</p></article>
        <article><h2>Evaluation</h2><p>Retrieval quality, citation support, faithfulness, latency and cost will be measured.</p></article>
      </section>
    </main>
  );
}
