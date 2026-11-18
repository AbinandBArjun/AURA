import { useEffect, useState } from "react";
import {
  Activity,
  ArrowUpRight,
  BrainCircuit,
  FileText,
  FolderOpen,
  Search,
  Workflow,
} from "lucide-react";

import { ApiError, getHealth } from "../services/api";

type BackendStatus =
  | "checking"
  | "connected"
  | "disconnected";

export default function Dashboard() {
  const [backendStatus, setBackendStatus] =
    useState<BackendStatus>("checking");

  const [errorMessage, setErrorMessage] = useState("");

  useEffect(() => {
    let active = true;

    getHealth()
      .then(() => {
        if (active) {
          setBackendStatus("connected");
          setErrorMessage("");
        }
      })
      .catch((error: unknown) => {
        if (!active) return;

        setBackendStatus("disconnected");

        setErrorMessage(
          error instanceof ApiError
            ? error.message
            : "Unable to connect to backend."
        );
      });

    return () => {
      active = false;
    };
  }, []);

  const metrics = [
    {
      label: "Documents Indexed",
      value: "-",
      description: "Document management integration pending",
      icon: FolderOpen,
    },
    {
      label: "Research Runs",
      value: "-",
      description: "Research history integration pending",
      icon: BrainCircuit,
    },
    {
      label: "Knowledge Queries",
      value: "-",
      description: "Query analytics integration pending",
      icon: Search,
    },
    {
      label: "Reports Generated",
      value: "-",
      description: "Report history integration pending",
      icon: FileText,
    },
  ];

  const modules = [
    {
      title: "Document Workspace",
      description:
        "Upload, organize and index research documents for retrieval.",
      icon: FolderOpen,
      tag: "DOCUMENT MANAGEMENT",
    },
    {
      title: "Knowledge Search",
      description:
        "Search your knowledge base using hybrid retrieval and reranking.",
      icon: Search,
      tag: "HYBRID RAG",
    },
    {
      title: "Research Agents",
      description:
        "Run autonomous investigations with evidence-backed findings.",
      icon: Workflow,
      tag: "LANGGRAPH",
    },
    {
      title: "Research Reports",
      description:
        "Review generated reports, citations and claim verification.",
      icon: FileText,
      tag: "EVIDENCE INTELLIGENCE",
    },
  ];

  return (
    <div className="dashboard">
      <section className="welcome-section">
        <div>
          <p className="eyebrow">RESEARCH INTELLIGENCE PLATFORM</p>
          <h1>Good to see you.</h1>
          <p className="welcome-description">
            Your research workspace is ready. Turn documents
            and information into evidence-backed intelligence.
          </p>
        </div>

        <div className={`backend-status ${backendStatus}`}>
          <span className="status-indicator" />
          <div>
            <strong>
              {backendStatus === "checking"
                ? "Checking backend"
                : backendStatus === "connected"
                  ? "Backend connected"
                  : "Backend unavailable"}
            </strong>
            <span>
              {backendStatus === "connected"
                ? "AURA API is responding"
                : backendStatus === "checking"
                  ? "Connecting to AURA API..."
                  : errorMessage}
            </span>
          </div>
        </div>
      </section>

      <section className="metrics-grid">
        {metrics.map((metric) => {
          const Icon = metric.icon;

          return (
            <article className="metric-card" key={metric.label}>
              <div className="metric-top">
                <span>{metric.label}</span>
                <Icon size={19} />
              </div>

              <strong className="metric-value">
                {metric.value}
              </strong>

              <p>{metric.description}</p>
            </article>
          );
        })}
      </section>

      <section className="section-heading">
        <div>
          <p className="eyebrow">YOUR WORKSPACE</p>
          <h2>Intelligence modules</h2>
          <p>Explore the core capabilities of AURA.</p>
        </div>
      </section>

      <section className="modules-grid">
        {modules.map((module) => {
          const Icon = module.icon;

          return (
            <article className="module-card" key={module.title}>
              <div className="module-icon">
                <Icon size={21} />
              </div>

              <span className="module-tag">{module.tag}</span>

              <h3>{module.title}</h3>

              <p>{module.description}</p>

              <span className="module-link">
                Explore module
                <ArrowUpRight size={16} />
              </span>
            </article>
          );
        })}
      </section>

      <section className="platform-footer">
        <div>
          <Activity size={18} />
          <span>AURA Research Engine</span>
        </div>

        <span>Version 0.1.0 - Autonomous Research Platform</span>
      </section>
    </div>
  );
}
