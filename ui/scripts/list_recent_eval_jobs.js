const { PrismaClient } = require('@prisma/client');
const prisma = new PrismaClient();
(async () => {
  try {
    const jobs = await prisma.evalJob.findMany({ where: { dataset: { contains: 'jinx_references' } }, orderBy: { created_at: 'desc' }, take: 10 });
    for (const j of jobs) {
      console.log('---');
      console.log('id:', j.id);
      console.log('created_at:', j.created_at);
      console.log('status:', j.status);
      console.log('params:', j.params);
    }
  } catch (e) {
    console.error(e);
    process.exit(1);
  } finally {
    await prisma.$disconnect();
  }
})();