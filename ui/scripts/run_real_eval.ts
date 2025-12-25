import prisma from '../cron/prisma';
import path from 'path';
import fs from 'fs';
import { execSync } from 'child_process';

async function main() {
  const datasetsRoot = path.join(process.cwd(), 'datasets');
  const target = path.join(datasetsRoot, 'jinx_references');
  if (!fs.existsSync(target)) {
    console.error('Dataset path not found:', target);
    process.exit(1);
  }

  // Try to find last finished eval for this dataset
  const last = await prisma.evalJob.findFirst({ where: { dataset: target, status: 'finished' }, orderBy: { created_at: 'desc' } });

  let model = 'default';
  let params: any = { batch_size: 1, sample_fraction: 1.0, device: 'cuda', samples_per_image: 4 };
  if (last) {
    console.log('Found previous finished eval job:', last.id, 'model:', last.model);
    model = last.model || 'default';
    try { params = JSON.parse(last.params || '{}') } catch (e) { }
  } else {
    // fallback to existing sample in DB or use default
    console.log('No previous finished job found; using default inferred parameters.');
  }

  // Create a fresh eval job with the same model/params
  const job = await prisma.evalJob.create({ data: { dataset: target, model: model, params: JSON.stringify(params), status: 'queued' } });
  console.log('Created eval job', job.id);

  // Run the worker once (spawn processEvalQueue via ts-node if available)
  try {
    console.log('Running processEvalQueue...');
    execSync('node ./ui/scripts/run_process_eval.js', { stdio: 'inherit', cwd: process.cwd() });
  } catch (e) {
    console.error('Failed to run processEvalQueue via node helper, trying ts-node...');
    try {
      execSync('ts-node ./ui/scripts/run_process_eval.ts', { stdio: 'inherit', cwd: process.cwd() });
    } catch (e2) {
      console.error('Failed to run worker:', e2);
    }
  }

  // fetch job status
  const updated = await prisma.evalJob.findUnique({ where: { id: job.id } });
  console.log('Final job status:', updated?.status, 'info:', updated?.info, 'out_json:', updated?.out_json);
}

main().catch(e => { console.error('Error:', e); process.exit(1); });