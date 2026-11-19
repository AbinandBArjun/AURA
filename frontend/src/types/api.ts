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


export interface SearchRequest {
  query: string;
  limit: number;
}

export interface SearchResult {
  document_id: number;
  filename: string;
  chunk_id: number;
  text: string;
  page_start: number;
  page_end: number;
  score: number;
}

export interface SearchResponse {
  query: string;
  results: SearchResult[];
  result_count: number;
}


export interface RAGRequest {
  question: string;
  limit: number;
}

export interface Citation {
  source_id: string;
  document_id: number;
  filename: string;
  chunk_id: number;
  page_start: number;
  page_end: number;
  score: number;
}

export type ClaimVerificationStatus =
  | "supported"
  | "partially_supported"
  | "contradicted"
  | "insufficient_evidence";

export interface ClaimVerification {
  claim_id: string;
  claim_text: string;
  status: ClaimVerificationStatus;
  rationale: string;
  evidence_ids: string[];
}

export interface RAGResponse {
  question: string;
  answer: string;
  citations: Citation[];
  evidence_count: number;
  claims: ClaimVerification[];
}
