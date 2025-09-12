'use client';

import React from 'react';
import { MessageSquare, Sun, Moon, Trash2, Upload } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { useTheme } from '@/hooks/useTheme';
import { useIngest } from '@/hooks/useIngest';

interface ChatHeaderProps {
  onClearMessages: () => void;
  messageCount: number;
}

export function ChatHeader({ onClearMessages, messageCount }: ChatHeaderProps) {
  const { theme, toggleTheme } = useTheme();
  const { ingestPdf, progress } = useIngest({ jpegQuality: 0.9, maxPages: 20 });

  const fileInputRef = React.useRef<HTMLInputElement>(null);

  const handleUploadClick = () => fileInputRef.current?.click();

  const onFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (file.type !== 'application/pdf') {
      alert('Please select a PDF file.');
      e.target.value = '';
      return;
    }
    // Basic size validation (<= 25MB)
    const MAX_BYTES = 25 * 1024 * 1024;
    if (file.size > MAX_BYTES) {
      alert('PDF is too large (max 25MB).');
      e.target.value = '';
      return;
    }
    const base = file.name.replace(/\.[^/.]+$/, '');
    const docId = `${base}-${Date.now()}`;
    try {
      const res = await ingestPdf(file, docId);
      if (res && typeof res.pages_added === 'number') {
        alert(`Ingested ${res.pages_added} pages from ${file.name}`);
      }
    } catch (err) {
      console.error(err);
      const msg = err instanceof Error ? err.message : 'Failed to ingest PDF.';
      alert(msg);
    } finally {
      e.target.value = '';
    }
  };

  return (
    <div className="border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 px-4 py-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 bg-blue-500 rounded-lg flex items-center justify-center">
            <MessageSquare className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              Chat Assistant
            </h1>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              {messageCount > 0 ? `${messageCount} messages` : 'Start a conversation'}
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          {/* PDF Upload */}
          <input
            ref={fileInputRef}
            type="file"
            accept="application/pdf"
            className="hidden"
            onChange={onFileChange}
          />
          <Button
            variant="ghost"
            size="sm"
            onClick={handleUploadClick}
            className="w-9 h-9 p-0"
            title="Upload PDF"
          >
            <Upload className="w-4 h-4" />
          </Button>

          {/* Theme Toggle */}
          <Button
            variant="ghost"
            size="sm"
            onClick={toggleTheme}
            className="w-9 h-9 p-0"
            title={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
          >
            {theme === 'light' ? (
              <Moon className="w-4 h-4" />
            ) : (
              <Sun className="w-4 h-4" />
            )}
          </Button>

          {/* Clear Messages */}
          {messageCount > 0 && (
            <Button
              variant="ghost"
              size="sm"
              onClick={onClearMessages}
              className="w-9 h-9 p-0 text-red-500 hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20"
              title="Clear all messages"
            >
              <Trash2 className="w-4 h-4" />
            </Button>
          )}
        </div>
      </div>
      {progress.status !== 'idle' && progress.status !== 'done' && (
        <div className="mt-2 text-xs text-gray-500 dark:text-gray-400">
          <span className="font-medium">Ingestion:</span> {progress.status}
          {typeof progress.currentPage === 'number' && (
            <>
              {' '}({progress.completedPages}/{progress.totalPages})
            </>
          )}
          {progress.status === 'error' && progress.error && (
            <span className="text-red-500"> — {progress.error}</span>
          )}
        </div>
      )}
    </div>
  );
}