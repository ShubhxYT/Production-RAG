import axios from 'axios';
import type {
  QueryRequest,
  QueryResponse,
  DocumentsResponse,
  FeedbackRequest,
  FeedbackResponse,
  EvaluationSummary,
  HealthResponse,
  UploadJobResponse,
  UploadStatusResponse,
} from './types';

const client = axios.create({ baseURL: '/api' });

export async function fetchHealth(): Promise<HealthResponse> {
  const { data } = await client.get<HealthResponse>('/health');
  return data;
}

export async function submitQuery(req: QueryRequest): Promise<QueryResponse> {
  const { data } = await client.post<QueryResponse>('/query', req);
  return data;
}

export async function fetchDocuments(): Promise<DocumentsResponse> {
  const { data } = await client.get<DocumentsResponse>('/documents');
  return data;
}

export async function submitFeedback(req: FeedbackRequest): Promise<FeedbackResponse> {
  const { data } = await client.post<FeedbackResponse>('/feedback', req);
  return data;
}

export async function fetchEvaluationSummary(since?: string): Promise<EvaluationSummary> {
  const params = since ? { since } : {};
  const { data } = await client.get<EvaluationSummary>('/evaluation/summary', { params });
  return data;
}

export async function uploadDocument(
  file: File,
  onProgress: (pct: number) => void,
): Promise<UploadJobResponse> {
  const formData = new FormData();
  formData.append('file', file);
  const { data } = await client.post<UploadJobResponse>('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress(event) {
      if (event.total) {
        onProgress(Math.round((event.loaded / event.total) * 20));
      }
    },
  });
  return data;
}

export async function getUploadStatus(jobId: string): Promise<UploadStatusResponse> {
  const { data } = await client.get<UploadStatusResponse>(`/upload/${jobId}/status`);
  return data;
}
