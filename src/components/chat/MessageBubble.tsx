import React from 'react';
import { Message } from '@/types/chat';
import { cn } from '@/lib/utils';
import { formatTime } from '@/lib/utils';
import { User, Bot } from 'lucide-react';
import { TypingIndicator } from '@/components/ui/TypingIndicator';

interface MessageBubbleProps {
  message: Message;
  showTimestamp?: boolean;
}

export function MessageBubble({ message, showTimestamp = true }: MessageBubbleProps) {
  const isUser = message.role === 'user';
  const isStreaming = message.isStreaming && message.content === '';

  return (
    <div className={cn('flex w-full mb-4', isUser ? 'justify-end' : 'justify-start')}>
      <div className={cn('flex max-w-[80%] md:max-w-[70%]', isUser ? 'flex-row-reverse' : 'flex-row')}>
        {/* Avatar */}
        <div className={cn('flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center', 
          isUser 
            ? 'bg-blue-500 text-white ml-3' 
            : 'bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-300 mr-3'
        )}>
          {isUser ? <User size={16} /> : <Bot size={16} />}
        </div>

        {/* Message Content */}
        <div className="flex flex-col">
          <div className={cn(
            'px-4 py-3 rounded-2xl shadow-sm',
            isUser
              ? 'bg-blue-500 text-white rounded-br-md'
              : 'bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 border border-gray-200 dark:border-gray-700 rounded-bl-md'
          )}>
            {isStreaming ? (
              <TypingIndicator />
            ) : (
              <div className="whitespace-pre-wrap break-words">
                {message.content}
              </div>
            )}
          </div>

          {/* Timestamp */}
          {showTimestamp && !isStreaming && (
            <div className={cn(
              'text-xs text-gray-500 dark:text-gray-400 mt-1 px-1',
              isUser ? 'text-right' : 'text-left'
            )}>
              {formatTime(message.timestamp)}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}