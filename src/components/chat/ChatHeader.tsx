import React from 'react';
import { MessageSquare, Sun, Moon, Trash2 } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { useTheme } from '@/hooks/useTheme';

interface ChatHeaderProps {
  onClearMessages: () => void;
  messageCount: number;
}

export function ChatHeader({ onClearMessages, messageCount }: ChatHeaderProps) {
  const { theme, toggleTheme } = useTheme();

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
    </div>
  );
}