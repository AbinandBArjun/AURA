import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertCircle,
  CheckCircle2,
  Clock3,
  FileText,
  LoaderCircle,
  RefreshCw,
  Upload,
  X,
} from "lucide-react";

import {
  ApiError,
  getCurrentWorkspace,
  getDocuments,
  processDocument,
  uploadDocument,
} from "../services/api";

import type {
  DocumentResponse,
  WorkspaceResponse,
} from "../types/api";

const MAX_FILE_SIZE = 20 * 1024 * 1024;

function formatFileSize(bytes: number): string {
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }

  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(value: string): string {
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

function getErrorMessage(error: unknown): string {
  return error instanceof ApiError || error instanceof Error
    ? error.message
    : "An unexpected error occurred.";
}

function StatusBadge({ status }: { status: string }) {
  const normalized = status.toLowerCase();

  const successful = normalized === "processed";
  const failed = ["failed", "index_failed"].includes(normalized);
  const attention = normalized === "needs_ocr";
  const active = ["indexing", "processing"].includes(normalized);

  const className = successful
    ? "document-status success"
    : failed
      ? "document-status failed"
      : attention
        ? "document-status attention"
        : active
          ? "document-status active"
          : "document-status pending";

  return (
    <span className={className}>
      {active && <LoaderCircle size={12} className="spin" />}
      {successful && <CheckCircle2 size={12} />}
      {failed && <AlertCircle size={12} />}
      {!successful && !failed && !active && <Clock3 size={12} />}
      {status.replace(/_/g, " ")}
    </span>
  );
}

export default function Documents() {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [workspace, setWorkspace] = useState<WorkspaceResponse | null>(null);
  const [documents, setDocuments] = useState<DocumentResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [processingId, setProcessingId] = useState<number | null>(null);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const loadDocuments = useCallback(async () => {
    setError("");

    try {
      const [workspaceResult, documentResults] = await Promise.all([
        getCurrentWorkspace(),
        getDocuments(),
      ]);

      setWorkspace(workspaceResult);
      setDocuments(documentResults);
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadDocuments();
  }, [loadDocuments]);

  async function handleFileSelection(
    event: React.ChangeEvent<HTMLInputElement>
  ) {
    const file = event.target.files?.[0];

    if (!file) return;

    setError("");
    setSuccess("");

    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setError("Please select a PDF file.");
      event.target.value = "";
      return;
    }

    if (file.size === 0) {
      setError("The selected file is empty.");
      event.target.value = "";
      return;
    }

    if (file.size > MAX_FILE_SIZE) {
      setError("The maximum upload size is 20 MB.");
      event.target.value = "";
      return;
    }

    setUploading(true);

    try {
      await uploadDocument(file);
      setSuccess(`${file.name} uploaded successfully.`);
      await loadDocuments();
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setUploading(false);
      event.target.value = "";
    }
  }

  async function handleProcess(document: DocumentResponse) {
    setError("");
    setSuccess("");
    setProcessingId(document.id);

    try {
      const result = await processDocument(document.id);

      setSuccess(
        `${result.filename}: ${result.message}`
      );

      await loadDocuments();
    } catch (err) {
      setError(getErrorMessage(err));
      await loadDocuments();
    } finally {
      setProcessingId(null);
    }
  }

  return (
    <section className="documents-page">
      <div className="documents-header">
        <div>
          <p className="eyebrow">KNOWLEDGE MANAGEMENT</p>
          <h1>Document Workspace</h1>
          <p className="welcome-description">
            Upload research papers and build your searchable knowledge base.
          </p>
          {workspace && (
            <div className="workspace-context">
              <span className="status-indicator connected" />
              {workspace.name}
            </div>
          )}
        </div>

        <button
          className="primary-button"
          onClick={() => fileInputRef.current?.click()}
          disabled={uploading || loading}
        >
          {uploading ? (
            <LoaderCircle size={16} className="spin" />
          ) : (
            <Upload size={16} />
          )}
          {uploading ? "Uploading..." : "Upload PDF"}
        </button>

        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,application/pdf"
          hidden
          onChange={handleFileSelection}
        />
      </div>

      {error && (
        <div className="document-alert error-alert" role="alert">
          <AlertCircle size={17} />
          <span>{error}</span>
          <button
            className="alert-dismiss"
            onClick={() => setError("")}
            aria-label="Dismiss error"
          >
            <X size={16} />
          </button>
        </div>
      )}

      {success && (
        <div className="document-alert success-alert" role="status">
          <CheckCircle2 size={17} />
          <span>{success}</span>
          <button
            className="alert-dismiss"
            onClick={() => setSuccess("")}
            aria-label="Dismiss message"
          >
            <X size={16} />
          </button>
        </div>
      )}

      <div className="documents-summary">
        <div className="documents-summary-icon">
          <FileText size={20} />
        </div>
        <div>
          <strong>{documents.length}</strong>
          <span>Documents in workspace</span>
        </div>

        <button
          className="secondary-button refresh-button"
          onClick={() => {
            setLoading(true);
            void loadDocuments();
          }}
          disabled={loading}
        >
          <RefreshCw size={15} />
          Refresh
        </button>
      </div>

      <div className="documents-panel">
        <div className="documents-panel-header">
          <div>
            <h2>All Documents</h2>
            <p>Manage uploads and document processing.</p>
          </div>
          <span className="document-count">
            {documents.length} files
          </span>
        </div>

        {loading ? (
          <div className="documents-empty">
            <LoaderCircle size={27} className="spin" />
            <p>Loading documents...</p>
          </div>
        ) : documents.length === 0 ? (
          <div className="documents-empty">
            <div className="empty-icon">
              <FileText size={27} />
            </div>
            <h3>No documents yet</h3>
            <p>
              Upload a PDF to start building your research knowledge base.
            </p>
            <button
              className="secondary-button"
              onClick={() => fileInputRef.current?.click()}
            >
              <Upload size={15} />
              Choose a PDF
            </button>
          </div>
        ) : (
          <div className="documents-table-wrapper">
            <table className="documents-table">
              <thead>
                <tr>
                  <th>DOCUMENT</th>
                  <th>SIZE</th>
                  <th>UPLOADED</th>
                  <th>STATUS</th>
                  <th>ACTION</th>
                </tr>
              </thead>
              <tbody>
                {documents.map((document) => (
                  <tr key={document.id}>
                    <td>
                      <div className="document-name">
                        <div className="document-file-icon">
                          <FileText size={17} />
                        </div>
                        <span title={document.filename}>
                          {document.filename}
                        </span>
                      </div>
                      {document.extraction_error && (
                        <p className="document-error-detail">
                          {document.extraction_error}
                        </p>
                      )}
                    </td>
                    <td>{formatFileSize(document.size_bytes)}</td>
                    <td>{formatDate(document.created_at)}</td>
                    <td>
                      <StatusBadge status={document.status} />
                    </td>
                    <td>
                      {["uploaded", "failed", "index_failed"].includes(
                        document.status
                      ) ? (
                        <button
                          className="process-button"
                          onClick={() => void handleProcess(document)}
                          disabled={
                            processingId !== null || uploading
                          }
                        >
                          {processingId === document.id ? (
                            <LoaderCircle
                              size={14}
                              className="spin"
                            />
                          ) : null}
                          {processingId === document.id
                            ? "Processing"
                            : "Process"}
                        </button>
                      ) : (
                        <span className="action-muted">
                          {document.status === "processed"
                            ? "Indexed"
                            : document.status === "needs_ocr"
                              ? "OCR needed"
                              : "In progress"}
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <p className="documents-footnote">
        PDF files only · Maximum file size 20 MB · Documents are stored
        within the active workspace.
      </p>
    </section>
  );
}
