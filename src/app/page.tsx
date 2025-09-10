import { ChatContainer } from '@/components/chat/ChatContainer';

// Avoid static pre-render to keep all client-only code
// from executing during build in server context.
export const dynamic = 'force-dynamic';

export default function Home() {
  return (
    <main className="h-screen">
      <ChatContainer />
    </main>
  );
}
