const prisma = require('../ui/dist/cron/prisma').default;
(async () => {
  try {
    const jobs = await prisma.evalJob.findMany({ orderBy: { created_at: 'desc' }, take: 20 });
    console.log(JSON.stringify(jobs.map(j=>({id:j.id,status:j.status,info:j.info,out_json:j.out_json,created_at:j.created_at,updated_at:j.updated_at})), null, 2));
  } catch (e) {
    console.error('ERROR', e);
    process.exit(1);
  } finally {
    await prisma.$disconnect();
  }
})();
