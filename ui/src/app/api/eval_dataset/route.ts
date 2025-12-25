import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/../../cron/prisma';
import { getDatasetsRoot } from '@/server/settings';
import { join } from 'path';
import fs from 'fs';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const datasetPath = body.dataset_path;
    const model = body.model || 'default';
    const batch_size = body.batch_size || 1;
    const sample_fraction = 'sample_fraction' in body ? body.sample_fraction : 1.0;
    const max_samples = body.max_samples || null;
    const device = body.device || 'cpu';

    const datasetsRoot = await getDatasetsRoot();
    if (!datasetsRoot) {
      return NextResponse.json({ error: 'Datasets root not found' }, { status: 500 });
    }

    const absDataset = join(datasetsRoot, datasetPath);
    if (!fs.existsSync(absDataset)) {
      return NextResponse.json({ error: `Dataset ${datasetPath} not found` }, { status: 404 });
    }

    const evalJob = await prisma.evalJob.create({
      data: {
        dataset: absDataset,
        model: model,
        params: JSON.stringify({ batch_size, sample_fraction, max_samples, device }),
        status: 'queued',
      },
    });

    return NextResponse.json({ id: evalJob.id });
  } catch (error) {
    console.error('Error creating eval job:', error);
    return NextResponse.json({ error: 'Error creating eval job' }, { status: 500 });
  }
}

export async function GET(request: NextRequest) {
  // list recent eval jobs
  const jobs = await prisma.evalJob.findMany({ orderBy: { created_at: 'desc' }, take: 50 });
  return NextResponse.json({ jobs });
}