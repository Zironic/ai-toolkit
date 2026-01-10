/* eslint-disable */
import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot, getTrainingFolder, getDataRoot } from '@/server/settings';

export async function GET(request: NextRequest, { params }: { params: { imagePath: string[] } }) {
  const { imagePath } = await params;
  try {
    // Join array segments and decode the path
    const decodedPath = decodeURIComponent(Array.isArray(imagePath) ? imagePath.join('/') : imagePath);
    console.log('[API /api/img] Requested path:', decodedPath, 'Type:', Array.isArray(imagePath) ? 'array' : 'string', 'Raw:', imagePath);

    // Get allowed directories
    const datasetRoot = await getDatasetsRoot();
    const trainingRoot = await getTrainingFolder();
    const dataRoot = await getDataRoot();
    console.log('[API /api/img] Allowed roots:', { datasetRoot, trainingRoot, dataRoot });

    // Resolve relative paths to absolute paths
    let filepath = decodedPath;
    if (!path.isAbsolute(filepath)) {
      // Try to resolve relative to known roots
      const possiblePaths = [
        path.resolve(datasetRoot, filepath),
        path.resolve(trainingRoot, filepath),
        path.resolve(dataRoot, filepath),
      ];
      
      console.log('[API /api/img] Trying paths:', possiblePaths);
      
      // Use the first path that exists
      const existingPath = possiblePaths.find(p => fs.existsSync(p));
      if (existingPath) {
        filepath = existingPath;
        console.log('[API /api/img] Found existing file at:', filepath);
      } else {
        // If no path exists, use the first possibility for security check
        filepath = possiblePaths[0];
        console.log('[API /api/img] No file found, using first path for security check:', filepath);
      }
    }

    const allowedDirs = [datasetRoot, trainingRoot, dataRoot];

    // Security check: Ensure path is in allowed directory
    const isAllowed = allowedDirs.some(allowedDir => filepath.startsWith(allowedDir)) && !filepath.includes('..');

    if (!isAllowed) {
      console.warn(`[API /api/img] Access denied: ${filepath} not in ${allowedDirs.join(', ')}`);
      return new NextResponse('Access denied', { status: 403 });
    }

    // Check if file exists
    if (!fs.existsSync(filepath)) {
      console.warn(`[API /api/img] File not found: ${filepath}`);
      return new NextResponse('File not found', { status: 404 });
    }

    // Get file info
    const stat = fs.statSync(filepath);
    if (!stat.isFile()) {
      return new NextResponse('Not a file', { status: 400 });
    }

    // Determine content type
    const ext = path.extname(filepath).toLowerCase();
    const contentTypeMap: { [key: string]: string } = {
      // Images
      '.jpg': 'image/jpeg',
      '.jpeg': 'image/jpeg',
      '.png': 'image/png',
      '.gif': 'image/gif',
      '.webp': 'image/webp',
      '.svg': 'image/svg+xml',
      '.bmp': 'image/bmp',
      // Videos
      '.mp4': 'video/mp4',
      '.avi': 'video/x-msvideo',
      '.mov': 'video/quicktime',
      '.mkv': 'video/x-matroska',
      '.wmv': 'video/x-ms-wmv',
      '.m4v': 'video/x-m4v',
      '.flv': 'video/x-flv'
    };

    const contentType = contentTypeMap[ext] || 'application/octet-stream';

    // Read file as buffer
    const fileBuffer = fs.readFileSync(filepath);

    // Return file with appropriate headers
    return new NextResponse(fileBuffer, {
      headers: {
        'Content-Type': contentType,
        'Content-Length': String(stat.size),
        'Cache-Control': 'public, max-age=86400',
      },
    });
  } catch (error) {
    console.error('Error serving image:', error);
    return new NextResponse('Internal Server Error', { status: 500 });
  }
}
