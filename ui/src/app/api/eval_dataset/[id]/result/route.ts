import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import fs from 'fs';

export async function GET(request: NextRequest, { params }: { params: { id: string } }) {
  try {
    const { id } = params;
    const job = await prisma.evalJob.findUnique({ where: { id } });
    if (!job) {
      return NextResponse.json({ error: 'Eval job not found' }, { status: 404 });
    }
    if (!job.out_json) {
      return NextResponse.json({ error: 'Result not yet available' }, { status: 404 });
    }
    try {
      const contents = fs.readFileSync(job.out_json, { encoding: 'utf-8' });
      try {
        const parsed = JSON.parse(contents);
        return NextResponse.json(parsed);
      } catch (e) {
        return NextResponse.json({ error: 'Result file is not valid JSON', info: e?.message }, { status: 500 });
      }
    } catch (e) {
      return NextResponse.json({ error: 'Could not read result file', info: e?.message }, { status: 500 });
    }
  } catch (error) {
    console.error('Error fetching eval result:', error);
    return NextResponse.json({ error: 'Error fetching result' }, { status: 500 });
  }
}