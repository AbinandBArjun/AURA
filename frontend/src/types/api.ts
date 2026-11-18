export interface HealthResponse {
  status: string;
  service: string;
}

export interface ApiErrorResponse {
  detail: string;
}

export interface WorkspaceResponse {
  id: number;
  name: string;
  owner_id: number;
  created_at: string;
}

export interface DocumentResponse {
  id: number;
  workspace_id: number;
  filename: string;
  size_bytes: number;
  status: string;
  extraction_error: string | null;
  checksum_sha256: string;
  created_at: string;
}

export interface DocumentProcessResponse {
  document_id: number;
  filename: string;
  status: string;
  page_count: number;
  total_characters: number;
  chunk_count: number;
  indexed_count: number;
  message: string;
}
