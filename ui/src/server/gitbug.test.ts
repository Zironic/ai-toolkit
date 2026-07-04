import { describe, it, expect } from 'vitest';
import { reduceTicket, type TicketCommit } from './gitbug';

const commit = (author: string, date: string, ops: any[]): TicketCommit => ({ author, date, ops });

describe('reduceTicket', () => {
  it('reconstructs title, status, labels and comments by replaying ops', () => {
    const commits = [
      commit('Alice', '2026-01-01T00:00:00Z', [
        { type: 1, timestamp: 100, title: 'Initial title', message: 'body of the report' },
      ]),
      commit('Bob', '2026-01-01T01:00:00Z', [
        { type: 5, timestamp: 150, added: ['perf', 'bug'], removed: null },
      ]),
      commit('Alice', '2026-01-01T02:00:00Z', [
        { type: 3, timestamp: 200, message: 'a follow-up comment' },
      ]),
      commit('Alice', '2026-01-01T03:00:00Z', [
        { type: 2, timestamp: 250, title: 'Renamed title' },
        { type: 4, timestamp: 260, status: 2 },
      ]),
    ];

    const t = reduceTicket('abc1234', 'abc1234full', commits);

    expect(t.title).toBe('Renamed title');
    expect(t.status).toBe('closed');
    expect(t.labels).toEqual(['bug', 'perf']); // sorted, unique
    expect(t.author).toBe('Alice'); // author of the create op
    expect(t.createdAt).toBe(100);
    expect(t.updatedAt).toBe(260); // max op timestamp
    expect(t.comments).toHaveLength(2);
    expect(t.comments[0]).toMatchObject({ author: 'Alice', message: 'body of the report' });
    expect(t.comments[1]).toMatchObject({ author: 'Alice', message: 'a follow-up comment' });
  });

  it('defaults to open and removes labels that were later dropped', () => {
    const commits = [
      commit('Al', '2026-02-01T00:00:00Z', [{ type: 1, timestamp: 10, title: 'T', message: '' }]),
      commit('Al', '2026-02-01T00:01:00Z', [{ type: 5, timestamp: 20, added: ['x', 'y'] }]),
      commit('Al', '2026-02-01T00:02:00Z', [{ type: 5, timestamp: 30, removed: ['x'] }]),
    ];
    const t = reduceTicket('id', 'idfull', commits);
    expect(t.status).toBe('open');
    expect(t.labels).toEqual(['y']);
  });

  it('falls back to commit date when an op has no timestamp', () => {
    const commits = [
      commit('Al', '2026-03-01T00:00:00Z', [{ type: 1, title: 'T', message: '' }]),
    ];
    const t = reduceTicket('id', 'idfull', commits);
    expect(t.createdAt).toBe(Math.floor(Date.parse('2026-03-01T00:00:00Z') / 1000));
  });
});
