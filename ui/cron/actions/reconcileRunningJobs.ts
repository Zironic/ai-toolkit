import fs from 'fs';
import path from 'path';
import { Job } from '@prisma/client';
import prisma from '../prisma';
import { getTrainingFolder } from '../paths';

const DEFAULT_STALE_MS = 10_000;

export type ReconcileRunningJobsOptions = {
  staleMs?: number;
  checkAll?: boolean;
};

const processExists = (pid: number): boolean => {
  if (!Number.isFinite(pid) || pid <= 0) {
    return false;
  }
  try {
    process.kill(pid, 0);
    return true;
  } catch (error: any) {
    if (error?.code === 'EPERM') {
      return true;
    }
    return false;
  }
};

const readPidFile = (jobFolder: string): number | null => {
  const pidPath = path.join(jobFolder, 'pid.txt');
  try {
    const value = fs.readFileSync(pidPath, 'utf-8').trim();
    if (!value) {
      return null;
    }
    const pid = Number(value);
    return Number.isFinite(pid) && pid > 0 ? pid : null;
  } catch {
    return null;
  }
};

const jobActivityMs = (job: Job, jobFolder: string): number => {
  const logPath = path.join(jobFolder, 'log.txt');
  try {
    return fs.statSync(logPath).mtimeMs;
  } catch {
    return job.updated_at.getTime();
  }
};

const resolveDeadJobState = (job: Job) => {
  if (job.return_to_queue) {
    return {
      status: 'queued',
      stop: false,
      return_to_queue: false,
      info: 'Job queued',
    };
  }
  if (job.stop) {
    return {
      status: 'stopped',
      stop: true,
      info: 'Job stopped',
    };
  }
  return {
    status: 'error',
    stop: false,
    info: 'Process exited unexpectedly',
  };
};

export default async function reconcileRunningJobs(options: ReconcileRunningJobsOptions = {}) {
  const staleMs = options.staleMs ?? DEFAULT_STALE_MS;
  const nowMs = Date.now();
  const trainingRoot = await getTrainingFolder();
  const runningJobs: Job[] = await prisma.job.findMany({
    where: { status: 'running' },
    orderBy: { updated_at: 'asc' },
  });

  for (const job of runningJobs) {
    const jobFolder = path.join(trainingRoot, job.name);
    if (!options.checkAll && nowMs - jobActivityMs(job, jobFolder) < staleMs) {
      continue;
    }

    const pid = job.pid ?? readPidFile(jobFolder);
    if (pid != null && processExists(pid)) {
      continue;
    }

    const data = {
      ...resolveDeadJobState(job),
      pid: null,
    };
    await prisma.job.update({
      where: { id: job.id },
      data,
    });
    console.warn(`Reconciled stale running job ${job.id}: PID ${pid ?? 'unknown'} is not alive`);
  }
}
