import { NextRequest, NextResponse } from 'next/server';
import { getDatasetsRoot } from '@/server/settings';
import path from 'path';
import fs from 'fs';

export async function GET(request: NextRequest) {
  try {
    const url = new URL(request.url);
    const dataset = url.searchParams.get('dataset');
    if (!dataset) return NextResponse.json({ error: 'dataset param required' }, { status: 400 });

    const datasetsRoot = await getDatasetsRoot();
    if (!datasetsRoot) return NextResponse.json({ error: 'Datasets root not found' }, { status: 500 });

    const absDataset = path.join(datasetsRoot, dataset);
    if (!fs.existsSync(absDataset)) return NextResponse.json({ error: 'Dataset not found' }, { status: 404 });

    const entries = fs.readdirSync(absDataset, { withFileTypes: true });
    const jsonFiles = entries
      .filter(e => e.isFile() && e.name.endsWith('.json'))
      .map(e => {
        const p = path.join(absDataset, e.name);
        // Parse file and validate it looks like an eval report with model metadata
        let modelName: string | null = null;
        let mtimeMs: number | null = null;
        let isValidEval = false;
        try {
          const contents = fs.readFileSync(p, 'utf-8');
          const parsed = JSON.parse(contents);
          // Basic validation: must have a 'datasets' object and model info in config
          if (parsed && typeof parsed === 'object' && parsed.datasets && typeof parsed.datasets === 'object') {
            const cfg: any = parsed.config || {};
            modelName = cfg.model || (cfg.model_config && cfg.model_config.name_or_path) || null;
            if (modelName) isValidEval = true;
          }
        } catch (err) {
          // ignore parse/validation errors
        }
        try {
          const st = fs.statSync(p);
          mtimeMs = st.mtimeMs;
        } catch (err) {
          mtimeMs = null;
        }
        if (!isValidEval) return null;
        return { filename: e.name, path: p, modelName, mtimeMs };
      })
      .filter(Boolean);

    return NextResponse.json({ files: jsonFiles });
  } catch (error) {
    console.error('Error listing eval files:', error);
    return NextResponse.json({ error: 'Error listing eval files' }, { status: 500 });
  }
}