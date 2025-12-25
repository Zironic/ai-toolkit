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

  // try several candidate locations for eval_dataset.py (project layout differences)
  const candidates = [
    path.join(TOOLKIT_ROOT, 'tools', 'eval_dataset.py'),
    path.join(TOOLKIT_ROOT, 'AI-Toolkit', 'tools', 'eval_dataset.py'),
    path.join(TOOLKIT_ROOT, '..', 'AI-Toolkit', 'tools', 'eval_dataset.py'),
  ];
  let scriptPath: string | null = null;
  for (const c of candidates) {
    if (fs.existsSync(c)) {
      scriptPath = c;
      console.log('Using eval_dataset script at', c);
      break;
    }
  }
  if (!scriptPath) {
    await prisma.evalJob.update({ where: { id: evalJob.id }, data: { status: 'error', info: 'eval_dataset.py not found' } });
    return;
  }

  // parse params stored as JSON
  const params = (() => { try { return JSON.parse(evalJob.params || '{}'); } catch (e) { return {}; } })();

  // build args: prefer params values and support passing full model config via a temp file
  const args: string[] = [scriptPath, '--dataset-path', evalJob.dataset, '--out-dir', evalJob.dataset, '--job-name', evalJob.id, '--step', '0'];

  // Forced batch size of 1 for per-example evaluation
  args.push('--batch-size', '1');
  if (params.device) args.push('--device', String(params.device));
  // force sample fraction to 1 (process every image)
  args.push('--sample-fraction', '1');
  // samples per image: prefer params value, otherwise use default 4
  if (params.samples_per_image) args.push('--samples-per-image', String(params.samples_per_image));
  else args.push('--samples-per-image', '4');
  // do not pass max_samples (not supported for full per-image evaluation)


  // model config: if provided, write a temp JSON file and pass --model-config-file, otherwise pass model string
  if (params.model_config) {
    const tmpDir = path.join(TOOLKIT_ROOT, 'tmp');
    try {
      fs.mkdirSync(tmpDir, { recursive: true });
      const tmpPath = path.join(tmpDir, `eval_${evalJob.id}_model.json`);
      fs.writeFileSync(tmpPath, JSON.stringify(params.model_config, null, 2));
      args.push('--model-config-file', tmpPath);
      // if model_config has dtype, pass it
      if (params.model_config.dtype) {
        args.push('--dtype', String(params.model_config.dtype));
      }
    } catch (e) {
      console.error('Failed to write model config temp file', e);
    }
  } else {
    args.push('--model', evalJob.model);
  }

  try {
    // ensure PYTHONPATH includes the toolkit root so imports like `import toolkit` work
    const envVars: any = { ...process.env };
    const existingPyPath = process.env?.PYTHONPATH || '';
    envVars.PYTHONPATH = existingPyPath ? `${existingPyPath}${path.delimiter}${TOOLKIT_ROOT}` : TOOLKIT_ROOT;
    console.log('Spawning python with PYTHONPATH:', envVars.PYTHONPATH);

    // Run Python in unbuffered mode so logs appear promptly
    const spawnArgs = ['-u', ...args];

    // capture stdio so we can log failures
    const child = spawn(pythonPath, spawnArgs, {
      detached: false,
      stdio: ['ignore', 'pipe', 'pipe'],
      cwd: TOOLKIT_ROOT,
      env: envVars,
    });

    // collect output
    let stdoutBuf = '';
    let stderrBuf = '';
    if (child.stdout) {
      child.stdout.on('data', d => {
        const s = d.toString();
        stdoutBuf += s;
        // also write to console for visibility
        console.log(`[eval:${evalJob.id}] stdout:`, s.trim());
      });
    }
    if (child.stderr) {
      child.stderr.on('data', d => {
        const s = d.toString();
        stderrBuf += s;
        console.error(`[eval:${evalJob.id}] stderr:`, s.trim());
      });
    }

    // store pid (if available)
    try {
      await prisma.evalJob.update({ where: { id: evalJob.id }, data: { info: `pid:${child.pid || 'unknown'}` } });
    } catch (e) {
      console.error('Could not write pid to db', e);
    }

    child.on('error', async err => {
      console.error('Error spawning eval process', err);
      try {
        await prisma.evalJob.update({ where: { id: evalJob.id }, data: { status: 'error', info: `spawn error: ${err?.message || String(err)}` } });
      } catch (e) {
        console.error('Failed to write spawn error to db', e);
      }
    });

    child.on('close', async code => {
      console.log(`Eval job ${evalJob.id} finished with code ${code}`);
      // prefer stderr for message, then stdout, then exit code
      const msg = (stderrBuf || stdoutBuf || `Exited with code ${code}`).slice(0, 2000);
      if (code === 0) {
        // Try model-based json filename first (sanitized model name), then fallback to job-based filename
        let outPath: string | null = null;
        try {
          const paramsParsed = params || {};
          let modelCandidate: string | null = null;
          if (paramsParsed.model_config && paramsParsed.model_config.name_or_path) modelCandidate = String(paramsParsed.model_config.name_or_path);
          else if (paramsParsed.model) modelCandidate = String(paramsParsed.model);

          const sanitize = (n: string) => {
            if (!n) return 'model';
            const parts = n.split(/[\/\\\\]/);
            const last = parts[parts.length - 1];
            return String(last).replace(/[^A-Za-z0-9_.-]/g, '_');
          };

          if (modelCandidate) {
            const candName = sanitize(modelCandidate) + '.json';
            const candPath = path.join(evalJob.dataset, candName);
            // If model-based file exists, use it. Otherwise, if the CLI wrote a
            // job-based JSON file, rename it to the model-based name so future
            // lookups use the model filename. If neither exists, set the expected
            // model path so UI/worker components know where to look.
            if (fs.existsSync(candPath)) {
              outPath = candPath;
            } else {
              const filename = `${evalJob.id}_${String(0).padStart(9, '0')}.json`;
              const jobCandidate = path.join(evalJob.dataset, filename);
              if (fs.existsSync(jobCandidate)) {
                try {
                  fs.renameSync(jobCandidate, candPath);
                  outPath = candPath;
                  console.log(`Renamed job JSON ${jobCandidate} -> ${candPath}`);
                } catch (renameErr) {
                  console.error('Failed to rename job JSON to model-based name', renameErr);
                  outPath = jobCandidate; // fallback
                }
              } else {
                // No file yet; set expected model-based path so UI can record it.
                outPath = candPath;
                console.log(`Model-based json ${candPath} not found; setting expected path.`);
              }
            }
          }
        } catch (e) {
          console.error('Error checking model-based json name', e);
        }

        if (!outPath) {
          const filename = `${evalJob.id}_${String(0).padStart(9, '0')}.json`;
          const candidate = path.join(evalJob.dataset, filename);
          if (fs.existsSync(candidate)) outPath = candidate;
        }

        try {
          await prisma.evalJob.update({ where: { id: evalJob.id }, data: { status: outPath ? 'finished' : 'finished', out_json: outPath, info: outPath ? 'Finished' : `Finished (no json) - ${msg}` } });
        } catch (e) {
          console.error('Failed to update job on close', e);
        }
      } else {
        try {
          await prisma.evalJob.update({ where: { id: evalJob.id }, data: { status: 'error', info: `Exit ${code}: ${msg}` } });
        } catch (e) {
          console.error('Failed to write exit error to db', e);
        }
      }
    });
  } catch (error: any) {
    console.error('Error launching eval process', error);
    await prisma.evalJob.update({ where: { id: evalJob.id }, data: { status: 'error', info: error?.message || 'unknown' } });
  }
}