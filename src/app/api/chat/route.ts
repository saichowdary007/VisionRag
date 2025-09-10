import { NextRequest, NextResponse } from 'next/server';

// Mock AI responses for demonstration
const mockResponses = [
  "I'm a helpful AI assistant. How can I help you today?",
  "That's an interesting question! Let me think about that...",
  "I understand what you're asking. Here's my perspective on that topic:",
  "Great question! I'd be happy to help you with that.",
  "I see what you mean. Let me provide you with some information about that.",
  "That's a thoughtful inquiry. Here's what I think:",
  "I appreciate you asking! Let me share some insights on this:",
  "Excellent point! Here's how I would approach that:",
];

// Simulate streaming response
async function* generateMockStream(prompt: string) {
  const response = mockResponses[Math.floor(Math.random() * mockResponses.length)];
  const words = response.split(' ');
  
  for (let i = 0; i < words.length; i++) {
    const chunk = i === 0 ? words[i] : ' ' + words[i];
    yield chunk;
    // Add realistic delay between words
    await new Promise(resolve => setTimeout(resolve, 50 + Math.random() * 100));
  }
}

export async function POST(request: NextRequest) {
  try {
    const { message, messages } = await request.json();

    if (!message || typeof message !== 'string') {
      return NextResponse.json(
        { error: 'Message is required and must be a string' },
        { status: 400 }
      );
    }

    // Create a readable stream for Server-Sent Events
    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      async start(controller) {
        try {
          // Send initial response
          controller.enqueue(
            encoder.encode(`data: ${JSON.stringify({ content: '' })}\n\n`)
          );

          // Generate and stream the mock response
          for await (const chunk of generateMockStream(message)) {
            controller.enqueue(
              encoder.encode(`data: ${JSON.stringify({ content: chunk })}\n\n`)
            );
          }

          // Send completion signal
          controller.enqueue(encoder.encode('data: [DONE]\n\n'));
          controller.close();
        } catch (error) {
          console.error('Stream error:', error);
          controller.enqueue(
            encoder.encode(`data: ${JSON.stringify({ 
              error: 'An error occurred while generating the response' 
            })}\n\n`)
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

// Optional: Add OpenAI integration
/*
import OpenAI from 'openai';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

export async function POST(request: NextRequest) {
  try {
    const { message, messages } = await request.json();

    if (!message || typeof message !== 'string') {
      return NextResponse.json(
        { error: 'Message is required and must be a string' },
        { status: 400 }
      );
    }

    // Convert our message format to OpenAI format
    const openaiMessages = [
      { role: 'system', content: 'You are a helpful assistant.' },
      ...messages.map((msg: any) => ({
        role: msg.role,
        content: msg.content,
      })),
      { role: 'user', content: message },
    ];

    const completion = await openai.chat.completions.create({
      model: 'gpt-3.5-turbo',
      messages: openaiMessages,
      stream: true,
    });

    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      async start(controller) {
        try {
          for await (const chunk of completion) {
            const content = chunk.choices[0]?.delta?.content || '';
            if (content) {
              controller.enqueue(
                encoder.encode(`data: ${JSON.stringify({ content })}\n\n`)
              );
            }
          }
          controller.enqueue(encoder.encode('data: [DONE]\n\n'));
          controller.close();
        } catch (error) {
          console.error('OpenAI stream error:', error);
          controller.enqueue(
            encoder.encode(`data: ${JSON.stringify({ 
              error: 'An error occurred while generating the response' 
            })}\n\n`)
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
*/