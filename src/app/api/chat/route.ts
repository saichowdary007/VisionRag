import { NextRequest, NextResponse } from 'next/server';

// Backend API Gateway (FastAPI) URL
const API_URL = (process.env.BACKEND_API_URL || 'http://localhost:8000').replace(/\/$/, '');

export async function POST(request: NextRequest) {
  try {
    const { message, top_k, max_images } = await request.json();

    if (!message || typeof message !== 'string') {
      return NextResponse.json(
        { error: 'Message is required and must be a string' },
        { status: 400 }
      );
    }

    // Call backend /query with extended timeout (no mock fallback)
    const timeoutMs = Number(process.env.QUERY_TIMEOUT_MS || process.env.BACKEND_TIMEOUT_MS || 120000);
    let data: any | null = null;
    try {
      const res = await fetch(`${API_URL}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: message }),
        cache: 'no-store',
        signal: AbortSignal.timeout(timeoutMs),
      });

      const text = await res.text();
      if (!res.ok) {
        // Stream a friendly error instead of returning non-200 to keep UI stable
        const msg = (() => {
          try { return JSON.parse(text)?.error || text; } catch { return text; }
        })() || `Backend error (${res.status})`;
        const encoder = new TextEncoder();
        const stream = new ReadableStream({
          start(controller) {
            controller.enqueue(encoder.encode(`data: ${JSON.stringify({ content: `Backend error: ${msg}` })}\n\n`));
            controller.enqueue(encoder.encode('data: [DONE]\n\n'));
            controller.close();
          }
        });
        return new Response(stream, {
          headers: {
            'Content-Type': 'text/plain; charset=utf-8',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
          },
        });
      }
      data = JSON.parse(text);
    } catch (backendErr) {
      console.error('Backend service error:', backendErr);
      const encoder = new TextEncoder();
      const detail = backendErr instanceof Error ? backendErr.message : String(backendErr);
      const stream = new ReadableStream({
        start(controller) {
          controller.enqueue(encoder.encode(`data: ${JSON.stringify({ content: `Backend unavailable: ${detail}` })}\n\n`));
          controller.enqueue(encoder.encode('data: [DONE]\n\n'));
          controller.close();
        }
      });
      return new Response(stream, {
        headers: {
          'Content-Type': 'text/plain; charset=utf-8',
          'Cache-Control': 'no-cache',
          'Connection': 'keep-alive',
        },
      });
    }

    const answer: string = data?.answer ?? '';
    const hits = Array.isArray(data?.sources) ? data.sources.map((s: any) => s.path) : [];

    // Stream the answer to the client as SSE-like chunks for compatibility with UI
    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      async start(controller) {
        try {
          controller.enqueue(encoder.encode(`data: ${JSON.stringify({ content: '', hits })}\n\n`));

          // Chunk by words to simulate streaming
          const words = String(answer).split(/(\s+)/);
          for (const chunk of words) {
            if (!chunk) continue;
            controller.enqueue(encoder.encode(`data: ${JSON.stringify({ content: chunk })}\n\n`));
            // Small delay improves UX; keep minimal on server
            // await new Promise(r => setTimeout(r, 0));
          }

          controller.enqueue(encoder.encode('data: [DONE]\n\n'));
          controller.close();
        } catch (err) {
          console.error('Streaming error:', err);
          controller.enqueue(
            encoder.encode(
              `data: ${JSON.stringify({ error: 'Failed to stream response' })}\n\n`
            )
          );
          controller.close();
        }
      },
    });

    return new Response(stream, {
      headers: {
        'Content-Type': 'text/plain; charset=utf-8',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      },
    });
  } catch (error) {
    console.error('API error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
