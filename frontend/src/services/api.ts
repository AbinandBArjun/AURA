import type {
  DocumentProcessResponse,
  DocumentResponse,
  HealthResponse,
  SearchResponse,
  WorkspaceResponse,
} from "../types/api";

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ??
  "http://localhost:8000/api/v1"
).replace(/\/$/, "");

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function apiRequest<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  let response: Response;

  try {
    response = await fetch(`${API_BASE_URL}${endpoint}`, {
      ...options,
      headers: {
        Accept: "application/json",
        ...options.headers,
      },
    });
  } catch {
    throw new ApiError(
      "Unable to connect to the AURA backend.",
      0
    );
  }

  if (!response.ok) {
    const payload = await response
      .json()
      .catch(() => null) as { detail?: unknown } | null;

    const message =
      typeof payload?.detail === "string"
        ? payload.detail
        : `Backend request failed with status ${response.status}.`;

    throw new ApiError(message, response.status);
  }

  return response.json() as Promise<T>;
}

export function getHealth(): Promise<HealthResponse> {
  return apiRequest<HealthResponse>("/health");
}

export function getCurrentWorkspace(): Promise<WorkspaceResponse> {
  return apiRequest<WorkspaceResponse>("/workspaces/current");
}

export function getDocuments(): Promise<DocumentResponse[]> {
  return apiRequest<DocumentResponse[]>("/documents");
}

export function uploadDocument(file: File): Promise<DocumentResponse> {
  const formData = new FormData();
  formData.append("file", file);

  return apiRequest<DocumentResponse>("/documents/upload", {
    method: "POST",
    body: formData,
  });
}

export function processDocument(
  documentId: number
): Promise<DocumentProcessResponse> {
  return apiRequest<DocumentProcessResponse>(
    `/documents/${documentId}/process`,
    {
      method: "POST",
    }
  );
}


export function searchKnowledge(
  query: string,
  limit = 5
): Promise<SearchResponse> {
  return apiRequest<SearchResponse>("/search", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ query, limit }),
  });
}
