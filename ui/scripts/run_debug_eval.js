const { PrismaClient } = require('@prisma/client');
const prisma = new PrismaClient();
const path = require('path');
const fs = require('fs');
const { execSync } = require('child_process');

async function main() {
  const root = path.join(process.cwd());
  const target = path.join(root, 'datasets', 'jinx_references');
  if (!fs.existsSync(target)) {
    console.error('Dataset path not found:', target);
    process.exit(1);
  }

  const params = { batch_size: 1, sample_fraction: 1.0, device: 'cuda', samples_per_image: 2, fixed_noise_std: 0.6, debug_captions: true };
  const job = await prisma.evalJob.create({ data: { dataset: target, model: 'Tongyi-MAI/Z-Image-Turbo', params: JSON.stringify(params), status: 'queued' } });
  console.log('Created debug eval job', job.id);

  // Run the worker action script to process one job
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
  await prisma.$disconnect();
  process.exit(0);
}

main().catch(e => { console.error(e); process.exit(1); });