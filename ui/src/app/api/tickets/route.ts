import { NextResponse } from 'next/server';
import { listTickets } from '@/server/gitbug';

// Read-only view of git-bug tickets straight from refs/bugs/*. Never invokes the
// git-bug binary, so it does not touch git-bug's lock — safe while the webui runs.
export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    const tickets = await listTickets();
    return NextResponse.json({ tickets });
  } catch (err: any) {
    return NextResponse.json(
      { error: err?.message ?? 'failed to read git-bug tickets' },
      { status: 500 },
    );
  }
}
