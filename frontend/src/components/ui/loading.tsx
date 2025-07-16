import React from 'react';
import { clsx } from 'clsx';

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export function LoadingSpinner({ size = 'md', className }: LoadingSpinnerProps) {
  return (
    <div
      className={clsx(
        'animate-spin rounded-full border-2 border-gray-300 border-t-blue-600',
        {
          'h-4 w-4': size === 'sm',
          'h-6 w-6': size === 'md',
          'h-8 w-8': size === 'lg',
        },
        className
      )}
    />
  );
}

interface LoadingProps {
  children?: React.ReactNode;
  className?: string;
}

export function Loading({ children, className }: LoadingProps) {
  return (
    <div className={clsx('flex items-center justify-center p-8', className)}>
      <div className="text-center">
        <LoadingSpinner size="lg" className="mx-auto mb-4" />
        {children && (
          <p className="text-sm text-gray-600">{children}</p>
        )}
      </div>
    </div>
  );
} 