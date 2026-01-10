import { NextRequest, NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { datasetName, imagePath, identifiers } = body;

    if (!datasetName || !identifiers || identifiers.length === 0) {
      return NextResponse.json(
        { success: false, error: 'Dataset name and identifiers are required' },
        { status: 400 }
      );
    }

    // Construct paths
    const repoRoot = process.cwd();
    const datasetPath = path.join(repoRoot, 'datasets', datasetName);
    
    // Check if dataset exists
    if (!fs.existsSync(datasetPath)) {
      return NextResponse.json(
        { success: false, error: `Dataset not found: ${datasetName}` },
        { status: 404 }
      );
    }

    const masksPath = path.join(datasetPath, 'masks');
    
    // Ensure masks directory exists
    if (!fs.existsSync(masksPath)) {
      fs.mkdirSync(masksPath, { recursive: true });
    }

    // Get Python executable path
    const pythonPath = process.platform === 'win32'
      ? path.join(repoRoot, 'venv', 'Scripts', 'python.exe')
      : path.join(repoRoot, 'venv', 'bin', 'python');

    // Verify python exists
    if (!fs.existsSync(pythonPath)) {
      return NextResponse.json(
        { success: false, error: 'Python virtual environment not found' },
        { status: 500 }
      );
    }

    const scriptPath = path.join(repoRoot, 'scripts', 'generate_masks_sam2.py');

    // Build command arguments
    const args = [
      scriptPath,
      '--dataset', imagePath || datasetPath,
      '--output', masksPath,
      '--use-comfyui',
      '--model', 'sam3.pt',
      '--text', identifiers.join(',')
    ];

    // Spawn the Python process
    const pythonProcess = spawn(pythonPath, args, {
      cwd: repoRoot,
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
