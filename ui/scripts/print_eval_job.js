const { PrismaClient } = require('@prisma/client');
const prisma = new PrismaClient();
const id = process.argv[2];
(async () => {
  if (!id) {
    console.error('Usage: node print_eval_job.js <job-id>');
    process.exit(1);
  }
  const job = await prisma.evalJob.findUnique({ where: { id } });
  console.log(JSON.stringify(job, null, 2));
  await prisma.$disconnect();
})();