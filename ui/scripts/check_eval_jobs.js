const { PrismaClient } = require('@prisma/client');
const prisma = new PrismaClient();

async function main() {
  const jobs = await prisma.evalJob.findMany({ where: { dataset: { contains: 'jinx_references' } } });
  console.log(`Found ${jobs.length} jobs for datasets/jinx_references`);
  for (const job of jobs) {
    console.log('---');
    console.log('id:', job.id);
    console.log('status:', job.status);
    console.log('out_json:', job.out_json);
    console.log('params:', job.params);
  }
  await prisma.$disconnect();
}

main().catch(e => { console.error(e); process.exit(1); });
