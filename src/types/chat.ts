export interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
  isStreaming?: boolean;
}

export interface ChatState {
  messages: Message[];
  isLoading: boolean;
  error: string | null;
}

export interface ApiResponse {
  message: string;
  error?: string;
}

export interface StreamResponse {
  content: string;
  done: boolean;
}

// Backend contracts
export interface RetrieverHit {
  page_id: string;
  score: number;
}

export interface QueryResponse {
  answer: string;
  hits: RetrieverHit[];
}