import { NextResponse } from 'next/server';

const API_URL = (process.env.BACKEND_API_URL || 'http://localhost:8000').replace(/\/$/, '');

export async function GET() {
  try {
    const res = await fetch(`${API_URL}/healthz`, { cache: 'no-store' });
    const text = await res.text();
    try {
      const json = JSON.parse(text);
      return NextResponse.json(json, { status: res.status });
    } catch {
      if (!res.ok) return NextResponse.json({ error: text || 'Backend error' }, { status: res.status });
      return NextResponse.json({ status: 'ok', detail: text });
    }
  } catch (err) {
    return NextResponse.json(
      { error: 'Backend service unavailable', details: err instanceof Error ? err.message : String(err) },
      { status: 502 }
    );
  }
}

