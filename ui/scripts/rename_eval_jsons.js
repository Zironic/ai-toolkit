const { PrismaClient } = require('@prisma/client');
const prisma = new PrismaClient();
const fs = require('fs');
const path = require('path');

async function main() {
  const jobs = await prisma.evalJob.findMany({ where: { status: 'finished' } });
  console.log(`Found ${jobs.length} finished eval jobs`);
  for (const job of jobs) {
    try {
      const dataset = job.dataset;
      if (!dataset || !fs.existsSync(dataset)) continue;
      const jobBased = path.join(dataset, `${job.id}_${String(0).padStart(9, '0')}.json`);
      // Parse params for model info
      let params = {};
      try { params = JSON.parse(job.params || '{}'); } catch (e) {}
      let modelCandidate = null;
      if (params.model_config && params.model_config.name_or_path) modelCandidate = String(params.model_config.name_or_path);
      else if (params.model) modelCandidate = String(params.model);

      if (fs.existsSync(jobBased) && modelCandidate) {
        const sanitize = (n) => {
          if (!n) return 'model';
          const parts = n.split(/[\\/\\\\]/);
          const last = parts[parts.length - 1];
          return String(last).replace(/[^A-Za-z0-9_.-]/g, '_');
        };
        const candName = sanitize(modelCandidate) + '.json';
        const candPath = path.join(dataset, candName);
        if (fs.existsSync(candPath)) {
          console.log(`Model-based file already exists for job ${job.id}: ${candPath}`);
          // update DB if necessary
          if (job.out_json !== candPath) {
            await prisma.evalJob.update({ where: { id: job.id }, data: { out_json: candPath } });
            console.log(`Updated DB out_json for job ${job.id}`);
          }
        } else {
          try {
            fs.renameSync(jobBased, candPath);
            await prisma.evalJob.update({ where: { id: job.id }, data: { out_json: candPath } });
            console.log(`Renamed ${jobBased} -> ${candPath} and updated DB for job ${job.id}`);
          } catch (e) {
            console.error('Failed to rename for job', job.id, e.message || e);
          }
        }
      }
    } catch (e) {
      console.error('Error processing job', job.id, e.message || e);
    }
  }
  await prisma.$disconnect();
}

main().catch(e => { console.error(e); process.exit(1); });
