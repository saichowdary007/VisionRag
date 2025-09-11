import { NextRequest, NextResponse } from 'next/server';

// Backend API Gateway (FastAPI) URL
const API_URL = (process.env.BACKEND_API_URL || 'http://localhost:8080').replace(/\/$/, '');

export async function POST(request: NextRequest) {
  try {
    const { message, top_k, max_images } = await request.json();

    if (!message || typeof message !== 'string') {
      return NextResponse.json(
        { error: 'Message is required and must be a string' },
        { status: 400 }
      );
    }

    // Try to call backend /query, fallback to mock if unavailable
    let data;
    try {
      const res = await fetch(`${API_URL}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: message, top_k, max_images }),
        cache: 'no-store',
        signal: AbortSignal.timeout(5000), // 5 second timeout
      });

      if (!res.ok) {
        console.warn(`Backend returned ${res.status}, using mock response`);
        // Fall back to mock response
      } else {
        data = await res.json();
      }
    } catch (backendErr) {
      console.warn('Backend service unavailable, using mock response:', backendErr);
      // Fall back to mock response
    }

    // Use mock data if backend call failed
    if (!data) {
      data = {
        answer: `This is a mock response to your question: "${message}". The backend service is currently unavailable, but the frontend is working correctly.`,
        hits: [
          { page_id: "mock_page_1", score: 0.95 },
          { page_id: "mock_page_2", score: 0.89 }
        ]
      };
    }

    const answer: string = data?.answer ?? '';
    const hits = Array.isArray(data?.hits) ? data.hits : [];

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