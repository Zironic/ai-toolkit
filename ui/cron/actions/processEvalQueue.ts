import prisma from '../prisma';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import { TOOLKIT_ROOT } from '../paths';

export default async function processEvalQueue() {
  // find one queued eval job
  const evalJob = await prisma.evalJob.findFirst({ where: { status: 'queued' }, orderBy: { created_at: 'asc' } });
  if (!evalJob) return;

  console.log(`Starting eval job ${evalJob.id} on dataset ${evalJob.dataset}`);

  await prisma.evalJob.update({ where: { id: evalJob.id }, data: { status: 'running', info: 'Starting evaluation' } });

  let pythonPath = 'python';
  if (fs.existsSync(path.join(TOOLKIT_ROOT, '.venv'))) {
    if (process.platform === 'win32') {
      pythonPath = path.join(TOOLKIT_ROOT, '.venv', 'Scripts', 'python.exe');
    } else {
      pythonPath = path.join(TOOLKIT_ROOT, '.venv', 'bin', 'python');
    }
  } else if (fs.existsSync(path.join(TOOLKIT_ROOT, 'venv'))) {
    if (process.platform === 'win32') {
      pythonPath = path.join(TOOLKIT_ROOT, 'venv', 'Scripts', 'python.exe');
    } else {
      pythonPath = path.join(TOOLKIT_ROOT, 'venv', 'bin', 'python');
    }
  }

  const scriptPath = path.join(TOOLKIT_ROOT, 'tools', 'eval_dataset.py');
  if (!fs.existsSync(scriptPath)) {
    await prisma.evalJob.update({ where: { id: evalJob.id }, data: { status: 'error', info: 'eval_dataset.py not found' } });
    return;
  }

  // pass job.name as job id and step=0, write to dataset folder
  const args = [scriptPath, '--dataset-path', evalJob.dataset, '--model', evalJob.model, '--batch-size', '1', '--device', 'cpu', '--out-dir', evalJob.dataset, '--job-name', evalJob.id, '--step', '0'];

  try {
    const child = spawn(pythonPath, args, {
      detached: false,
      stdio: 'ignore',
      cwd: TOOLKIT_ROOT,
      env: {
        ...process.env,
      },
    });

    // store pid
    try {
      await prisma.evalJob.update({ where: { id: evalJob.id }, data: { info: `pid:${child.pid}` } });
    } catch (e) {
      console.error('Could not write pid to db', e);
    }

    child.on('close', async code => {
      console.log(`Eval job ${evalJob.id} finished with code ${code}`);
      if (code === 0) {
        // expected out_json name
        const filename = `${evalJob.id}_${String(0).padStart(9, '0')}.json`;
        const outPath = path.join(evalJob.dataset, filename);
        const exists = fs.existsSync(outPath);
        await prisma.evalJob.update({ where: { id: evalJob.id }, data: { status: exists ? 'finished' : 'finished', out_json: exists ? outPath : null, info: exists ? 'Finished' : 'Finished (no json)' } });
      } else {
        await prisma.evalJob.update({ where: { id: evalJob.id }, data: { status: 'error', info: `Exited with code ${code}` } });
      }
    });
  } catch (error: any) {
    console.error('Error launching eval process', error);
    await prisma.evalJob.update({ where: { id: evalJob.id }, data: { status: 'error', info: error?.message || 'unknown' } });
  }
}