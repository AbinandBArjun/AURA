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

export interface ConversationCreateRequest {
  title?: string | null;
}

export interface ConversationMessageCreateRequest {
  content: string;
  limit: number;
}

export interface ConversationMessageResponse {
  id: number;
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
  created_at: string;
}

export interface ConversationResponse {
  id: number;
  workspace_id: number;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface ConversationDetailResponse
  extends ConversationResponse {
  messages: ConversationMessageResponse[];
}

export interface ConversationTurnResponse {
  conversation_id: number;
  user_message: ConversationMessageResponse;
  assistant_message: ConversationMessageResponse;
  evidence_count: number;
}

