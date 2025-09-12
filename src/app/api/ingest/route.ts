import { NextRequest, NextResponse } from 'next/server';

// Retriever service URL (FastAPI retriever)
// When running Next.js locally (npm run dev), connect to Docker services on localhost
// When running in Docker, use service names
const RETRIEVER_URL = (process.env.RETRIEVER_URL || 'http://localhost:8080').replace(/\/$/, '');

// Accept JSON body: { pages: [{ page_id: string, image_b64: string }] }
// Proxies to retriever /ingest without mock fallback; surfaces real errors
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    if (!body || !Array.isArray(body.pages)) {
      return NextResponse.json(
        { error: 'Invalid payload: expected { pages: IngestItem[] }' },
        { status: 400 }
      );
    }

    try {
      // Call the retriever service with extended timeout (configurable)
      const timeoutMs = Number(process.env.INGEST_TIMEOUT_MS || process.env.RETRIEVER_TIMEOUT_MS || 60000);
      const res = await fetch(`${RETRIEVER_URL}/ingest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        cache: 'no-store',
        signal: AbortSignal.timeout(timeoutMs),
      });

      const text = await res.text();
      if (!res.ok) {
        // Forward retriever error response and status to client
        try {
          const data = JSON.parse(text);
          return NextResponse.json(data, { status: res.status });
        } catch {
          return NextResponse.json({ error: text || `Retriever error (${res.status})` }, { status: res.status });
        }
      }

      // Successful response passthrough
      try {
        return NextResponse.json(JSON.parse(text));
      } catch {
        return NextResponse.json({ status: 'ok', detail: text });
      }
    } catch (retrieverErr) {
      console.error('Retriever service error:', retrieverErr);
      return NextResponse.json(
        {
          error: 'Retriever service unavailable',
          details: retrieverErr instanceof Error ? retrieverErr.message : String(retrieverErr),
        },
        { status: 502 }
      );
    }

  } catch (err) {
    console.error('Ingest API error:', err);
    return NextResponse.json({
      error: 'Internal server error',
      details: err instanceof Error ? err.message : String(err)
    }, { status: 500 });
  }
}

