import { execFile } from 'child_process';
import { promisify } from 'util';
import { TOOLKIT_ROOT } from '@/paths';

const execFileAsync = promisify(execFile);

// git-bug stores each bug as a chain of git commits under refs/bugs/<full-id>.
// Every commit's tree carries an `ops` JSON blob: { author:{id}, ops:[ ... ] }.
// Operation `type` is an enum; replaying the ops oldest->newest rebuilds state.
// We read this purely with read-only git plumbing (never the git-bug binary), so
// there is no interaction with git-bug's single-access lock — the webui can stay
// running while this serves tickets.
const OP = {
  CREATE: 1,
  SET_TITLE: 2,
  ADD_COMMENT: 3,
  SET_STATUS: 4,
  LABEL_CHANGE: 5,
  EDIT_COMMENT: 6,
} as const;

export interface GitBugOp {
  type: number;
  timestamp?: number;
  title?: string;
  message?: string;
  status?: number; // 1 = open, 2 = closed
  added?: string[] | null;
  removed?: string[] | null;
}

// One git commit = one operation pack authored by one identity.
export interface TicketCommit {
  author: string; // display name (git commit author)
  date: string; // ISO commit author date
  ops: GitBugOp[];
}

export interface TicketComment {
  author: string;
  timestamp: number;
  message: string;
}

export interface Ticket {
  id: string; // 7-char short id
  fullId: string;
  title: string;
  status: 'open' | 'closed';
  labels: string[];
  author: string;
  createdAt: number; // unix seconds
  updatedAt: number; // unix seconds
  comments: TicketComment[];
}

/**
 * Pure reducer: fold a bug's commits (in oldest->newest order) into a Ticket.
 * Kept free of any git I/O so it is unit-testable with synthetic input.
 * Note: EditComment (type 6) is intentionally not remapped onto its target
 * comment — that needs operation-hash resolution and is unnecessary for a
 * read-only viewer; the original comment text is shown.
 */
export function reduceTicket(shortId: string, fullId: string, commits: TicketCommit[]): Ticket {
  let title = '';
  let status: 'open' | 'closed' = 'open';
  let author = '';
  let createdAt = 0;
  let updatedAt = 0;
  const labels = new Set<string>();
  const comments: TicketComment[] = [];

  for (const commit of commits) {
    const fallbackTs = Math.floor(Date.parse(commit.date) / 1000) || 0;
    for (const op of commit.ops) {
      const ts = op.timestamp ?? fallbackTs;
      if (ts > updatedAt) updatedAt = ts;
      switch (op.type) {
        case OP.CREATE:
          title = op.title ?? '';
          if (!author) author = commit.author;
          createdAt = ts;
          comments.push({ author: commit.author, timestamp: ts, message: op.message ?? '' });
          break;
        case OP.SET_TITLE:
          if (op.title != null) title = op.title;
          break;
        case OP.ADD_COMMENT:
          comments.push({ author: commit.author, timestamp: ts, message: op.message ?? '' });
          break;
        case OP.SET_STATUS:
          status = op.status === 2 ? 'closed' : 'open';
          break;
        case OP.LABEL_CHANGE:
          for (const l of op.added ?? []) labels.add(l);
          for (const l of op.removed ?? []) labels.delete(l);
          break;
        default:
          break;
      }
    }
  }

  return {
    id: shortId,
    fullId,
    title,
    status,
    labels: [...labels].sort(),
    author,
    createdAt,
    updatedAt,
    comments,
  };
}

const US = '\x1f'; // unit separator, safe in author names/dates

async function git(args: string[]): Promise<string> {
  const { stdout } = await execFileAsync('git', args, {
    cwd: TOOLKIT_ROOT,
    maxBuffer: 32 * 1024 * 1024,
    windowsHide: true,
  });
  return stdout;
}

async function readTicket(fullRef: string): Promise<Ticket | null> {
  const fullId = fullRef.replace('refs/bugs/', '');
  const log = (await git(['log', '--reverse', `--format=%H${US}%an${US}%aI`, fullRef])).trim();
  if (!log) return null;

  const commits: TicketCommit[] = [];
  for (const line of log.split('\n')) {
    const [hash, author, date] = line.split(US);
    let parsed: { ops?: GitBugOp[] };
    try {
      parsed = JSON.parse(await git(['cat-file', '-p', `${hash}:ops`]));
    } catch {
      continue; // skip a commit we can't decode rather than failing the whole bug
    }
    commits.push({ author, date, ops: parsed.ops ?? [] });
  }
  return reduceTicket(fullId.slice(0, 7), fullId, commits);
}

/** List every git-bug ticket, reconstructed read-only from refs/bugs/*. */
export async function listTickets(): Promise<Ticket[]> {
  const refs = (await git(['for-each-ref', '--format=%(refname)', 'refs/bugs']))
    .split('\n')
    .map(r => r.trim())
    .filter(Boolean);

  const tickets = (await Promise.all(refs.map(readTicket))).filter((t): t is Ticket => t !== null);

  // Open first, then most-recently-updated first.
  tickets.sort((a, b) => {
    if (a.status !== b.status) return a.status === 'open' ? -1 : 1;
    return b.updatedAt - a.updatedAt;
  });
  return tickets;
}
