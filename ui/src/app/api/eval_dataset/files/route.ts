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
        // Try to parse model name from file contents and record modification time
        let modelName: string | null = null;
        let mtimeMs: number | null = null;
        try {
          const contents = fs.readFileSync(p, 'utf-8');
          const parsed = JSON.parse(contents);
          if (parsed && parsed.config && parsed.config.model) modelName = parsed.config.model;
        } catch (e) {
          // ignore parse errors
        }
        try {
          const st = fs.statSync(p);
          mtimeMs = st.mtimeMs;
        } catch (e) {
          mtimeMs = null;
        }
        return { filename: e.name, path: p, modelName, mtimeMs };
      });

    return NextResponse.json({ files: jsonFiles });
  } catch (error) {
    console.error('Error listing eval files:', error);
    return NextResponse.json({ error: 'Error listing eval files' }, { status: 500 });
  }
}