const prisma = require('../ui/dist/cron/prisma').default;
(async () => {
  try {
    const jobs = await prisma.evalJob.findMany({ where: { status: 'running' }, orderBy: { created_at: 'desc' } });
    console.log(JSON.stringify(jobs, null, 2));
  } catch (e) {
    console.error('ERROR', e);
    process.exit(1);
  } finally {
    await prisma.$disconnect();
  }
})();
