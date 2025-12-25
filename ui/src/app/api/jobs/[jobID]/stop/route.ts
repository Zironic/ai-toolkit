import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import { getTrainingFolder } from '@/server/settings';
import path from 'path';
import fs from 'fs';

const prisma = new PrismaClient();

export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;
  const searchParams = request.nextUrl.searchParams;
  const save = searchParams.get('save') === 'true';

  const job = await prisma.job.findUnique({
    where: { id: jobID },
  });

  // update job to request stop
  await prisma.job.update({
    where: { id: jobID },
    data: {
      stop: true,
      info: save ? 'Stopping job (saving first)...' : 'Stopping job...',
    },
  });

  // Update .job_config.json with save_before_stop flag
  if (job) {
    try {
      const trainingRoot = await getTrainingFolder();
      const trainingFolder = path.join(trainingRoot, job.name);
      const configPath = path.join(trainingFolder, '.job_config.json');
      
      if (fs.existsSync(configPath)) {
        const config = JSON.parse(fs.readFileSync(configPath, 'utf-8'));
        config.save_before_stop = save;
        fs.writeFileSync(configPath, JSON.stringify(config, null, 2));
      }
    } catch (e) {
      console.error('Error updating job config:', e);
    }
  }

  return NextResponse.json(job);
}
