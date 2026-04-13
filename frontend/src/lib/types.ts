export interface QueryRequest {
  question: string;
  top_k?: number;
  prompt_variant?: 'qa' | 'summarize' | null;
}

export interface Source {
  document_title: string | null;
  source_path: string;
  chunk_summary: string;
  page_numbers: number[];
  similarity_score: number;
}

export interface LatencyBreakdown {
  retrieval_ms: number;
  context_ms: number;
  generation_ms: number;
  total_ms: number;
}

export interface TokenUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface QueryResponse {
  answer: string;
  sources: Source[];
  latency: LatencyBreakdown;
  token_usage: TokenUsage;
  prompt_version: string;
  query_log_id: string | null;
}

export interface DocumentItem {
  id: string;
  title: string | null;
  source_path: string;
  chunk_count: number;
  created_at: string;
}

export interface DocumentsResponse {
  documents: DocumentItem[];
  total: number;
}

export interface FeedbackRequest {
  query_log_id: string | null;
  feedback_type: 'thumbs_up' | 'thumbs_down' | 'correction';
  rating?: number;
  correction?: string;
  query_text?: string;
}

export interface FeedbackResponse {
  id: string;
  created_at: string;
}

export interface EvaluationSummary {
  total_feedback: number;
  positive_rate: number;
  avg_rating: number | null;
  counts_by_type: Record<string, number>;
  since: string | null;
}

export interface HealthResponse {
  status: string;
  database: string;
}
