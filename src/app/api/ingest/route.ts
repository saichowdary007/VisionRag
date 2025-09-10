import { NextRequest, NextResponse } from 'next/server';

// Retriever service URL (FastAPI retriever)
const RETRIEVER_URL = (process.env.RETRIEVER_URL || 'http://localhost:8081').replace(/\/$/, '');

// Accept JSON body: { pages: [{ page_id: string, image_b64: string }] }
// Proxies to retriever /ingest
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    if (!body || !Array.isArray(body.pages)) {
      return NextResponse.json(
        { error: 'Invalid payload: expected { pages: IngestItem[] }' },
        { status: 400 }
      );
    }

    const res = await fetch(`${RETRIEVER_URL}/ingest`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      cache: 'no-store',
    });

    const text = await res.text();
    if (!res.ok) {
      return NextResponse.json(
        { error: `Retriever error ${res.status}: ${text || res.statusText}` },
        { status: res.status }
      );
    }

    try {
      return NextResponse.json(JSON.parse(text));
    } catch {
      return NextResponse.json({ status: 'ok', detail: text });
    }
  } catch (err) {
    console.error('Ingest proxy error:', err);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}


