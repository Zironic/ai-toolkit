const { PrismaClient } = require('@prisma/client');
const prisma = new PrismaClient();
const path = require('path');
const fs = require('fs');
const { execSync } = require('child_process');

async function main() {
  const root = path.join(process.cwd());
  const datasetsRoot = path.join(root, 'datasets');
  const target = path.join(datasetsRoot, 'jinx_references');
  if (!fs.existsSync(target)) {
    console.error('Dataset path not found:', target);
    process.exit(1);
  }

  const last = await prisma.evalJob.findFirst({ where: { dataset: target, status: 'finished' }, orderBy: { created_at: 'desc' } });

  let model = 'default';
  let params = { batch_size: 1, sample_fraction: 1.0, device: 'cuda', samples_per_image: 4 };
  if (last) {
    console.log('Found previous finished eval job:', last.id, 'model:', last.model);
    model = last.model || 'default';
    try { params = JSON.parse(last.params || '{}'); } catch (e) {}
  } else {
    console.log('No previous finished job found; using default inferred parameters.');
  }

  const job = await prisma.evalJob.create({ data: { dataset: target, model: model, params: JSON.stringify(params), status: 'queued' } });
  console.log('Created eval job', job.id);

  // Run the worker action script
  try {
    console.log('Invoking processEvalQueue...');
    execSync('node ui/scripts/run_process_eval.js', { stdio: 'inherit' });
  } catch (e) {
    console.error('Failed to run processEvalQueue helper, attempting to run cron action directly with node');
    try {
      execSync('node -e "(async () => { const fn = require(\'./cron/actions/processEvalQueue\').default; await fn(); })()"', { stdio: 'inherit' });
    } catch (e2) {
      console.error('Failed to invoke worker:', e2);
    }
  }

  const updated = await prisma.evalJob.findUnique({ where: { id: job.id } });
  console.log('Final job status:', updated?.status, 'info:', updated?.info, 'out_json:', updated?.out_json);
  process.exit(0);
}

main().catch(e => { console.error(e); process.exit(1); });