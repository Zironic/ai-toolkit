'use client';

import { useEffect, useMemo, useState } from 'react';
import { TopBar, MainContent } from '@/components/layout';
import { RefreshCw } from 'lucide-react';

interface TicketComment {
  author: string;
  timestamp: number;
  message: string;
}
interface Ticket {
  id: string;
  fullId: string;
  title: string;
  status: 'open' | 'closed';
  labels: string[];
  author: string;
  createdAt: number;
  updatedAt: number;
  comments: TicketComment[];
}

const fmt = (ts: number) => (ts ? new Date(ts * 1000).toLocaleString() : '');

export default function TicketsPage() {
  const [tickets, setTickets] = useState<Ticket[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<'all' | 'open' | 'closed'>('open');
  const [expanded, setExpanded] = useState<string | null>(null);

  const load = () => {
    setLoading(true);
    fetch('/api/tickets')
      .then(r => r.json())
      .then(d => {
        if (d.error) throw new Error(d.error);
        setTickets(d.tickets ?? []);
        setError(null);
      })
      .catch(e => setError(e.message ?? 'failed to load'))
      .finally(() => setLoading(false));
  };

  useEffect(load, []);

  const shown = useMemo(
    () => tickets.filter(t => filter === 'all' || t.status === filter),
    [tickets, filter],
  );
  const openCount = tickets.filter(t => t.status === 'open').length;

  return (
    <>
      <TopBar>
        <div>
          <h1 className="text-base sm:text-lg">Tickets</h1>
        </div>
        <div className="flex-1" />
        <div className="flex items-center gap-2 text-xs">
          {(['open', 'closed', 'all'] as const).map(f => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-2 py-1 rounded capitalize ${
                filter === f ? 'bg-gray-700 text-white' : 'text-gray-400 hover:text-gray-200'
              }`}
            >
              {f}
            </button>
          ))}
          <button
            onClick={load}
            className="p-1 text-gray-400 hover:text-gray-200"
            aria-label="Refresh"
            title="Refresh"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </TopBar>
      <MainContent>
        <div className="mb-3 text-xs text-gray-500">
          {openCount} open · {tickets.length} total · read-only view of{' '}
          <code>refs/bugs/*</code> (no git-bug lock)
        </div>

        {error && (
          <div className="rounded border border-red-800 bg-red-950/40 p-3 text-sm text-red-300">
            {error}
          </div>
        )}

        {!error && shown.length === 0 && !loading && (
          <div className="text-sm text-gray-500">No {filter === 'all' ? '' : filter} tickets.</div>
        )}

        <div className="flex flex-col gap-2">
          {shown.map(t => {
            const isOpen = expanded === t.fullId;
            return (
              <div
                key={t.fullId}
                className="rounded border border-gray-800 bg-gray-900/40 overflow-hidden"
              >
                <button
                  onClick={() => setExpanded(isOpen ? null : t.fullId)}
                  className="w-full flex items-center gap-3 px-3 py-2 text-left hover:bg-gray-800/40"
                >
                  <span
                    className={`shrink-0 text-[10px] uppercase font-semibold px-1.5 py-0.5 rounded ${
                      t.status === 'open'
                        ? 'bg-green-900/60 text-green-300'
                        : 'bg-gray-700/60 text-gray-300'
                    }`}
                  >
                    {t.status}
                  </span>
                  <code className="shrink-0 text-xs text-gray-500">{t.id}</code>
                  <span className="flex-1 text-sm text-gray-100 truncate">{t.title}</span>
                  {t.labels.map(l => (
                    <span
                      key={l}
                      className="hidden sm:inline shrink-0 text-[10px] px-1.5 py-0.5 rounded bg-blue-950/50 text-blue-300 border border-blue-900"
                    >
                      {l}
                    </span>
                  ))}
                  <span className="shrink-0 text-[11px] text-gray-600 hidden md:inline">
                    {fmt(t.updatedAt)}
                  </span>
                </button>

                {isOpen && (
                  <div className="border-t border-gray-800 px-3 py-3 space-y-3">
                    <div className="flex flex-wrap gap-1 sm:hidden">
                      {t.labels.map(l => (
                        <span
                          key={l}
                          className="text-[10px] px-1.5 py-0.5 rounded bg-blue-950/50 text-blue-300 border border-blue-900"
                        >
                          {l}
                        </span>
                      ))}
                    </div>
                    {t.comments.map((c, i) => (
                      <div key={i} className="text-sm">
                        <div className="text-[11px] text-gray-500 mb-0.5">
                          {i === 0 ? 'opened' : 'commented'} by {c.author} · {fmt(c.timestamp)}
                        </div>
                        <div className="whitespace-pre-wrap text-gray-200">{c.message}</div>
                      </div>
                    ))}
                    <div className="text-[11px] text-gray-600 pt-1 border-t border-gray-800/60">
                      {t.fullId}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </MainContent>
    </>
  );
}
