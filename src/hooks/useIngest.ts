'use client';

import { useCallback, useState } from 'react';
import { getDocument, GlobalWorkerOptions } from 'pdfjs-dist';
// Derive installed pdfjs-dist version to keep worker in sync
// JSON import requires resolveJsonModule: true (already set)
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore - package.json import typing
import pdfjsPkg from 'pdfjs-dist/package.json';

// Configure worker for pdfjs in browser.
// Always match the worker version to the installed pdfjs-dist to avoid version mismatch errors.
const INSTALLED_PDFJS_VERSION: string = (pdfjsPkg as any)?.version || '4.8.69';
const VERSIONED_CDN_WORKER = `https://unpkg.com/pdfjs-dist@${INSTALLED_PDFJS_VERSION}/build/pdf.worker.min.mjs`;
GlobalWorkerOptions.workerSrc = (process.env.NEXT_PUBLIC_PDFJS_WORKER as string) || VERSIONED_CDN_WORKER;

export interface IngestProgress {
  totalPages: number;
  completedPages: number;
  currentPage?: number;
  status: 'idle' | 'parsing' | 'rendering' | 'uploading' | 'done' | 'error';
  error?: string;
}

interface UseIngestOptions {
  maxPages?: number; // optional cap
  jpegQuality?: number; // 0-1
}

export function useIngest(options: UseIngestOptions = {}) {
  const [progress, setProgress] = useState<IngestProgress>({ totalPages: 0, completedPages: 0, status: 'idle' });

  const ingestPdf = useCallback(async (file: File, docId: string) => {
    try {
      if (!file || file.type !== 'application/pdf') {
        throw new Error('Please upload a valid PDF file.');
      }

      const arrayBuf = await file.arrayBuffer();
      setProgress({ totalPages: 0, completedPages: 0, status: 'parsing' });

      const pdf = await getDocument({ data: arrayBuf }).promise;
      const total = options.maxPages ? Math.min(pdf.numPages, options.maxPages) : pdf.numPages;
      setProgress({ totalPages: total, completedPages: 0, status: 'rendering' });

      const pages: { page_id: string; image_b64: string }[] = [];

      for (let i = 1; i <= total; i++) {
        setProgress(p => ({ ...p, currentPage: i }));
        const page = await pdf.getPage(i);
        const viewport = page.getViewport({ scale: 2 });
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        if (!ctx) throw new Error('Canvas not supported');
        canvas.width = viewport.width;
        canvas.height = viewport.height;
        await page.render({ canvasContext: ctx as any, viewport }).promise;
        const quality = typeof options.jpegQuality === 'number' ? options.jpegQuality : 0.9;
        const dataUrl = canvas.toDataURL('image/jpeg', quality);
        const image_b64 = dataUrl.replace(/^data:image\/jpeg;base64,/, '');
        pages.push({ page_id: `${docId}:${i}`, image_b64 });
        setProgress(p => ({ ...p, completedPages: i }));
      }

      setProgress(p => ({ ...p, status: 'uploading' }));
      const res = await fetch('/api/ingest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pages }),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Upload failed with status ${res.status}`);
      }
      setProgress(p => ({ ...p, status: 'done' }));
      return await res.json();
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Ingestion failed';
      setProgress(p => ({ ...p, status: 'error', error: msg }));
      throw err;
    }
  }, [options.maxPages, options.jpegQuality]);

  return { ingestPdf, progress };
}

