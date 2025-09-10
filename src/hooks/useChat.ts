'use client';

import { useState, useCallback, useEffect } from 'react';
import { Message, ChatState, RetrieverHit } from '@/types/chat';
import { generateId } from '@/lib/utils';

const STORAGE_KEY = 'chat-messages';

export function useChat() {
  const [state, setState] = useState<ChatState>({
    messages: [],
    isLoading: false,
    error: null,
  });

  // Load messages from localStorage on mount
  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const messages = JSON.parse(stored).map((msg: any) => ({
          ...msg,
          timestamp: new Date(msg.timestamp),
        }));
        setState(prev => ({ ...prev, messages }));
      }
    } catch (error) {
      console.error('Failed to load messages from storage:', error);
    }
  }, []);

  // Save messages to localStorage whenever messages change
  useEffect(() => {
    if (state.messages.length > 0) {
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(state.messages));
      } catch (error) {
        console.error('Failed to save messages to storage:', error);
      }
    }
  }, [state.messages]);

  const addMessage = useCallback((message: Omit<Message, 'id' | 'timestamp'>) => {
    const newMessage: Message = {
      ...message,
      id: generateId(),
      timestamp: new Date(),
    };

    setState(prev => ({
      ...prev,
      messages: [...prev.messages, newMessage],
      error: null,
    }));

    return newMessage;
  }, []);

  const updateMessage = useCallback((id: string, updates: Partial<Message>) => {
    setState(prev => ({
      ...prev,
      messages: prev.messages.map(msg =>
        msg.id === id ? { ...msg, ...updates } : msg
      ),
    }));
  }, []);

  const setLoading = useCallback((loading: boolean) => {
    setState(prev => ({ ...prev, isLoading: loading }));
  }, []);

  const setError = useCallback((error: string | null) => {
    setState(prev => ({ ...prev, error }));
  }, []);

  const clearMessages = useCallback(() => {
    setState(prev => ({ ...prev, messages: [] }));
    localStorage.removeItem(STORAGE_KEY);
  }, []);

  const sendMessage = useCallback(async (content: string) => {
    if (!content.trim()) return;

    // Add user message
    const userMessage = addMessage({
      content: content.trim(),
      role: 'user',
    });

    // Add assistant message placeholder
    const assistantMessage = addMessage({
      content: '',
      role: 'assistant',
      isStreaming: true,
    });

    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: content.trim(),
          messages: state.messages,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) {
        throw new Error('No response body');
      }

      let accumulatedContent = '';
      let hits: RetrieverHit[] | undefined;

      while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            
            if (data === '[DONE]') {
              updateMessage(assistantMessage.id, {
                isStreaming: false,
              });
              break;
            }

            try {
              const parsed = JSON.parse(data);
              if (Array.isArray(parsed.hits)) {
                hits = parsed.hits;
              }
              if (parsed.content) {
                accumulatedContent += parsed.content;
                updateMessage(assistantMessage.id, {
                  content: accumulatedContent,
                });
              }
            } catch (e) {
              // Skip invalid JSON
            }
          }
        }
      }
    } catch (error) {
      console.error('Chat error:', error);
      const errorMessage = error instanceof Error ? error.message : 'An unexpected error occurred';
      setError(errorMessage);
      
      // Update assistant message with error
      updateMessage(assistantMessage.id, {
        content: 'Sorry, I encountered an error while processing your request. Please try again.',
        isStreaming: false,
      });
    } finally {
      setLoading(false);
    }
  }, [state.messages, addMessage, updateMessage, setLoading, setError]);

  return {
    messages: state.messages,
    isLoading: state.isLoading,
    error: state.error,
    sendMessage,
    clearMessages,
  };
}