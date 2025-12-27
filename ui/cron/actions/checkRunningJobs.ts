import prisma from '../prisma';
import fs from 'fs';
import path from 'path';
import { getTrainingFolder } from '../paths';

let lastRun = 0;
const MIN_INTERVAL_MS = 60 * 1000; // run at most once per minute

function isProcessRunning(pid: number) {
  try {
    // Signal 0 does not kill the process, but will throw an error if it doesn't exist
    process.kill(pid, 0);
    return true;
  } catch (e) {
    return false;
  }
}

export default async function checkRunningJobs() {
  const now = Date.now();
  if (now - lastRun < MIN_INTERVAL_MS) return;
  lastRun = now;

  try {
    // Check EvalJobs
    const evals = await prisma.evalJob.findMany({ where: { status: 'running' } });
    for (const e of evals) {
      const info = e.info || '';
      const m = info.match(/pid:(\d+)/);
      if (!m) continue; // per-PID-only policy: skip if no PID recorded
      const pid = Number(m[1]);
      if (!pid) continue;
      if (!isProcessRunning(pid)) {
        console.log(`Liveness: EvalJob ${e.id} pid ${pid} not found — marking stopped`);
        await prisma.evalJob.update({
          where: { id: e.id },
          data: {
            status: 'stopped',
            info: `Auto-stopped by liveness checker: pid:${pid} not found at ${new Date().toISOString()}`,
          },
        });
      }
    }

    // Check training Jobs (they write pid.txt into their training folder)
    const jobs = await prisma.job.findMany({ where: { status: 'running' } });
    if (jobs.length > 0) {
      const trainingRoot = await getTrainingFolder();
      for (const j of jobs) {
        try {
          const pidPath = path.join(trainingRoot, j.name, 'pid.txt');
          if (!fs.existsSync(pidPath)) continue; // no pid file -> skip per PID-only policy
          const s = fs.readFileSync(pidPath, 'utf8').trim();
          const pid = Number(s);
          if (!pid) continue;
          if (!isProcessRunning(pid)) {
            console.log(`Liveness: Job ${j.id} pid ${pid} not found — marking stopped`);
            await prisma.job.update({
              where: { id: j.id },
              data: {
                status: 'stopped',
                info: `Auto-stopped by liveness checker: pid:${pid} not found at ${new Date().toISOString()}`,
              },
            });
          }
        } catch (e) {
          console.error('Error checking job liveness for', j.id, e);
        }
      }
    }
  } catch (e) {
    console.error('Error in liveness checker', e);
  }
}
