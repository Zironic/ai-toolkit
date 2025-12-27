import checkRunningJobs from '../cron/actions/checkRunningJobs';

(async () => {
  try {
    await checkRunningJobs();
    console.log('Liveness check completed');
  } catch (e) {
    console.error('Liveness check failed', e);
    process.exit(1);
  }
})();
