'use client';

import { useEffect } from 'react';

export default function Error({ error, reset }: { error: Error & { digest?: string }; reset: () => void }) {
  useEffect(() => {
    console.error(error);
  }, [error]);

  return (
    <div className="flex flex-col items-center justify-center min-h-screen gap-4 p-8 text-center">
      <h2 className="text-xl font-semibold text-red-400">Something went wrong</h2>
      <pre className="text-sm text-gray-400 bg-gray-800 rounded p-4 max-w-2xl overflow-auto text-left whitespace-pre-wrap">
        {error.message}
      </pre>
      <button
        onClick={reset}
        className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md text-sm"
      >
        Try again
      </button>
    </div>
  );
}
