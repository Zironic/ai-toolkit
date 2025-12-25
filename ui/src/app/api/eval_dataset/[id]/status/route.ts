import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';

export async function GET(request: NextRequest, { params }: { params: { id: string } }) {
  try {
    const { id } = params;
    const job = await prisma.evalJob.findUnique({ where: { id } });
    if (!job) {
      return NextResponse.json({ error: 'Eval job not found' }, { status: 404 });
    }
    return NextResponse.json({ job });
  } catch (error) {
    console.error('Error fetching eval job status:', error);
    return NextResponse.json({ error: 'Error fetching status' }, { status: 500 });
  }
}