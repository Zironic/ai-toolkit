import processQueue from './actions/processQueue';
import reconcileRunningJobs from './actions/reconcileRunningJobs';
class CronWorker {
  interval: number;
  reconcileInterval: number;
  lastReconcileAt: number;
  is_running: boolean;
  intervalId: NodeJS.Timeout;
  constructor() {
    this.interval = 1000; // Default interval of 1 second
    this.reconcileInterval = 10000;
    this.lastReconcileAt = 0;
    this.is_running = false;
    this.intervalId = setInterval(() => {
      this.run();
    }, this.interval);
    void this.reconcileRunningJobs(true).catch(error => console.error('Error reconciling running jobs on startup:', error));
  }
  async run() {
    if (this.is_running) {
      return;
    }
    this.is_running = true;
    try {
      // Loop logic here
      await this.loop();
    } catch (error) {
      console.error('Error in cron worker loop:', error);
    }
    this.is_running = false;
  }

  async loop() {
    await this.reconcileRunningJobs(false);
    await processQueue();
  }

  async reconcileRunningJobs(checkAll: boolean) {
    const now = Date.now();
    if (!checkAll && now - this.lastReconcileAt < this.reconcileInterval) {
      return;
    }
    this.lastReconcileAt = now;
    await reconcileRunningJobs({ checkAll, staleMs: this.reconcileInterval });
  }
}

// it automatically starts the loop
const cronWorker = new CronWorker();
console.log('Cron worker started with interval:', cronWorker.interval, 'ms');
