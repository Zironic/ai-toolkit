import { NextRequest, NextResponse } from 'next/server';
import { getDatasetsRoot } from '@/server/settings';
import path from 'path';
import fs from 'fs';

export async function GET(request: NextRequest) {
  try {
    const url = new URL(request.url);
    const filePath = url.searchParams.get('path');
    if (!filePath) return NextResponse.json({ error: 'path param required' }, { status: 400 });

    const datasetsRoot = await getDatasetsRoot();
    if (!datasetsRoot) return NextResponse.json({ error: 'Datasets root not found' }, { status: 500 });

    // Only allow files inside the datasets root for safety
    const resolved = path.resolve(filePath);
    if (!resolved.startsWith(path.resolve(datasetsRoot))) {
      return NextResponse.json({ error: 'Invalid file path' }, { status: 400 });
    }

    if (!fs.existsSync(resolved)) return NextResponse.json({ error: 'File not found' }, { status: 404 });

    try {
      const contents = fs.readFileSync(resolved, 'utf-8');
      const parsed = JSON.parse(contents);
      return NextResponse.json(parsed);
    } catch (e: any) {
      return NextResponse.json({ error: 'Could not read/parse file', info: e?.message }, { status: 500 });
    }
  } catch (error) {
    console.error('Error reading eval file:', error);
    return NextResponse.json({ error: 'Error reading eval file' }, { status: 500 });
  }
}