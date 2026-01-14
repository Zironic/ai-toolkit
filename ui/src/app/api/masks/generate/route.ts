import { NextRequest, NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import { getDatasetsRoot } from '@/server/settings';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { datasetName, imagePath, identifiers, merge_texts = true, mode = 'sam3', color } = body;

    if (!datasetName) {
      return NextResponse.json({ success: false, error: 'Dataset name is required' }, { status: 400 });
    }

    // Validate mode-specific inputs
    if (mode === 'sam3') {
      if (!identifiers || identifiers.length === 0) {
        return NextResponse.json({ success: false, error: 'Identifiers are required for SAM3' }, { status: 400 });
      }
    } else if (mode === 'background') {
      if (!color) {
        return NextResponse.json({ success: false, error: 'Color is required for Background Mask mode' }, { status: 400 });
      }
    }

    // Resolve dataset folder via configured datasets root (honors admin setting)
    const datasetsRoot = await getDatasetsRoot();
    const datasetPath = path.join(datasetsRoot, datasetName);

    // Check if dataset exists
    if (!fs.existsSync(datasetPath)) {
      return NextResponse.json(
        { success: false, error: `Dataset not found: ${datasetName}` },
        { status: 404 }
      );
    }

    // Use `masks` folder (public-facing masks)
    const masksPath = path.join(datasetPath, 'masks');

    // Ensure masks directory exists
    if (!fs.existsSync(masksPath)) {
      fs.mkdirSync(masksPath, { recursive: true });
    }

    // Locate repo root (try a few reasonable candidates so Next server doesn't fail when CWD is inside ui/)
    const candidates = [process.cwd(), path.resolve(process.cwd(), '..'), path.resolve(process.cwd(), '..', '..')];
    let repoRootFound = candidates.find((r) => fs.existsSync(path.join(r, 'scripts', 'generate_masks_sam2.py')));
    if (!repoRootFound) repoRootFound = process.cwd();

    // Get Python executable path (search common venv locations)
    const pythonCandidates = [
      path.join(repoRootFound, 'venv', process.platform === 'win32' ? 'Scripts' : 'bin', process.platform === 'win32' ? 'python.exe' : 'python'),
      path.join(repoRootFound, '.venv', process.platform === 'win32' ? 'Scripts' : 'bin', process.platform === 'win32' ? 'python.exe' : 'python'),
      path.join(repoRootFound, 'env', process.platform === 'win32' ? 'Scripts' : 'bin', process.platform === 'win32' ? 'python.exe' : 'python'),
    ];
    const pythonPath = pythonCandidates.find(p => fs.existsSync(p));

    // Verify python exists
    if (!pythonPath) {
      return NextResponse.json(
        { success: false, error: 'Python virtual environment not found' },
        { status: 500 }
      );
    }

    const scriptPath = path.join(repoRootFound, 'scripts', 'generate_masks_sam2.py');

    // Build command arguments for SAM3 mode only (deferred until we know mode)
    let args: string[] = [];
    if (mode === 'sam3') {
      args = [
        scriptPath,
        '--dataset', imagePath || datasetPath,
        '--output', masksPath,
        '--use-comfyui',
        '--model', 'sam3.pt',
        '--text', (identifiers || []).join(','),
      ];

      // If merge_texts is true, include the flag so the script will merge CSV identifiers into a single mask
      if (merge_texts) {
        args.push('--merge-texts');
      }
    }

    // If background mode requested, call the dedicated Python background mask generator
    if (mode === 'background') {
      try {
        const bgScript = path.join(repoRootFound, 'scripts', 'generate_background_masks.py');
        const bgArgs = [bgScript, '--color', color, '--output', masksPath];
        if (imagePath) {
          bgArgs.push('--image', imagePath);
        } else {
          bgArgs.push('--dataset', datasetPath);
        }

        const bgProcess = spawn(pythonPath, bgArgs, { cwd: repoRootFound, env: { ...process.env } });
        let out = '';
        let err = '';
        bgProcess.stdout.on('data', (d) => { out += d.toString(); console.log('[BG-MASK]', d.toString()); });
        bgProcess.stderr.on('data', (d) => { err += d.toString(); console.error('[BG-MASK-ERR]', d.toString()); });

        return new Promise((resolve) => {
          bgProcess.on('close', (code) => {
            if (code === 0) {
              try {
                const parsed = JSON.parse(out.trim() || '{}');
                resolve(NextResponse.json({ success: true, message: 'Background masks generated', masks: parsed.masks || [] }));
              } catch (e) {
                resolve(NextResponse.json({ success: true, message: 'Background masks generated', output: out }, { status: 200 }));
              }
            } else {
              resolve(NextResponse.json({ success: false, error: 'Background mask generation failed', details: err || out }, { status: 500 }));
            }
          });
        });
      } catch (e) {
        console.error('Background mask error:', e);
        return NextResponse.json({ success: false, error: String(e) }, { status: 500 });
      }
    }

    // Spawn the Python process (SAM3) for sam3 mode
    const pythonProcess = spawn(pythonPath, args, {
      cwd: repoRootFound,
      env: { ...process.env },
    });

    let stdout = '';
    let stderr = '';

    pythonProcess.stdout.on('data', (data) => {
      stdout += data.toString();
      console.log(`[Mask Generation] ${data}`);
    });

    pythonProcess.stderr.on('data', (data) => {
      stderr += data.toString();
      console.error(`[Mask Generation Error] ${data}`);
    });

    return new Promise((resolve) => {
      pythonProcess.on('close', (code) => {
        if (code === 0) {
          resolve(
            NextResponse.json({
              success: true,
              message: imagePath
                ? 'Mask generated successfully'
                : 'Masks generated successfully for all images',
              output: stdout,
            })
          );
        } else {
          resolve(
            NextResponse.json(
              {
                success: false,
                error: 'Mask generation failed',
                details: stderr || stdout,
              },
              { status: 500 }
            )
          );
        }
      });

      // Set timeout (30 minutes)
      setTimeout(() => {
        pythonProcess.kill();
        resolve(
          NextResponse.json(
            {
              success: false,
              error: 'Mask generation timed out (30 minutes)',
            },
            { status: 408 }
          )
        );
      }, 30 * 60 * 1000);
    });
  } catch (error: any) {
    console.error('Error in mask generation API:', error);
    return NextResponse.json(
      { success: false, error: error.message || 'Internal server error' },
      { status: 500 }
    );
  }
}
