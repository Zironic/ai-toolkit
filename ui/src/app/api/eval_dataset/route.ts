import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import { getDatasetsRoot } from '@/server/settings';
import path, { join } from 'path';
import fs from 'fs';
import { modelArchs } from '@/app/jobs/new/options';

// Helpers to mirror the jobs/new model defaults without changing toolkit behavior
const getArchForModelName = (modelName: string) => {
  for (const arch of modelArchs) {
    const nameDefault = arch.defaults?.['config.process[0].model.name_or_path']?.[0];
    if (nameDefault && nameDefault === modelName) return arch;
  }
  return null;
};

const findBaseForExtras = (extrasName: string) => {
  // find an arch whose extras_name_or_path default equals extrasName, then return that arch's base name
  for (const arch of modelArchs) {
    const extrasDefault = arch.defaults?.['config.process[0].model.extras_name_or_path']?.[0];
    if (extrasDefault && extrasDefault === extrasName) {
      return arch.defaults?.['config.process[0].model.name_or_path']?.[0] || null;
    }
  }
  return null;
};

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const datasetPath = body.dataset_path;
    const model = body.model || 'default';
    const model_config = body.model_config || null;
    // Batch size is forced to 1 for per-example loss evaluation
    const batch_size = 1;
    // sample_fraction is forced to 1 to process every image
    const sample_fraction = 1.0;
    // max_samples not supported for full per-image evaluation (unset)
    const max_samples = null;
    // Force device to 'cuda' for evaluation consistency
    const device = 'cuda';

    const datasetsRoot = await getDatasetsRoot();
    if (!datasetsRoot) {
      return NextResponse.json({ error: 'Datasets root not found' }, { status: 500 });
    }

    const absDataset = join(datasetsRoot, datasetPath);
    if (!fs.existsSync(absDataset)) {
      return NextResponse.json({ error: `Dataset ${datasetPath} not found` }, { status: 404 });
    }

    // attempt to resolve a local model path (mirror resolver used in training flow)
    const resolveLocalModelPath = (m: string | undefined) => {
      if (!m) return null;
      const root = process.cwd();
      const candidates = [
        m,
        path.join(root, 'models', m),
        path.join(root, 'models', m.split('/').pop() || ''),
        path.join(root, 'models', m.replace('/', '_')),
        path.join(root, 'toolkit', 'diffusers_configs', m),
        path.join(root, 'toolkit', 'diffusers_configs', m.split('/').pop() || ''),
        path.join(root, 'toolkit', 'orig_configs', m),
        path.join(root, 'toolkit', 'orig_configs', m.split('/').pop() || ''),
      ];
      for (const c of candidates) {
        try {
          if (fs.existsSync(String(c))) return String(c);
        } catch (e) { continue; }
      }
      return null;
    };

    // prefer model_config.name_or_path if provided and try to resolve local paths, and enrich using jobs/new defaults
    // Respect UI-provided options: samples_per_image, fixed_noise_std, and debug_captions
    const paramsObj: any = { batch_size, sample_fraction: 1.0, device, samples_per_image: (body.samples_per_image ?? 8) };
    // Coerce and validate fixed_noise_std (numbers or numeric strings allowed)
    if (typeof body.fixed_noise_std !== 'undefined') {
      const fn = Number(body.fixed_noise_std);
      if (Number.isFinite(fn) && fn >= 0.0 && fn <= 1.0) {
        paramsObj.fixed_noise_std = fn;
      } else {
        // ignore invalid values and keep CLI default
        console.warn('Ignored invalid fixed_noise_std from request:', body.fixed_noise_std);
      }
    }
    // Coerce samples_per_image to an integer, with a sensible lower bound of 1
    if (typeof body.samples_per_image !== 'undefined') {
      const spi = Math.max(1, parseInt(String(body.samples_per_image), 10) || 1);
      paramsObj.samples_per_image = spi;
    }
    if (typeof body.debug_captions !== 'undefined') paramsObj.debug_captions = body.debug_captions;
    if (typeof body.ablation_compare !== 'undefined') paramsObj.ablation_compare = body.ablation_compare;
    if (model_config) {
      // if model_config.name_or_path points to a known local path, prefer that
      const resolved = resolveLocalModelPath(model_config.name_or_path);
      if (resolved) {
        model_config.name_or_path = resolved;
        model = resolved;
      }

      // infer arch from name_or_path if missing
      if (!model_config.arch && model_config.name_or_path) {
        const inferred = getArchForModelName(model_config.name_or_path);
        if (inferred) model_config.arch = inferred.name;
      }

      // if model appears to be the 'extras' entry, find the corresponding base model and set extras_name_or_path
      if (!model_config.extras_name_or_path && model_config.name_or_path) {
        const base = findBaseForExtras(model_config.name_or_path);
        if (base) model_config.extras_name_or_path = base;
      }

      // copy safe defaults from the arch spec when available
      if (model_config.arch) {
        const archSpec = modelArchs.find(a => a.name === model_config.arch);
        if (archSpec) {
          const assistant = archSpec.defaults?.['config.process[0].model.assistant_lora_path']?.[0];
          if (assistant && !model_config.assistant_lora_path) model_config.assistant_lora_path = assistant;
          const quant = archSpec.defaults?.['config.process[0].model.quantize']?.[0];
          if (typeof quant !== 'undefined' && model_config.quantize === undefined) model_config.quantize = quant;
          const lowV = archSpec.defaults?.['config.process[0].model.low_vram']?.[0];
          if (typeof lowV !== 'undefined' && model_config.low_vram === undefined) model_config.low_vram = lowV;
        }
      }

      paramsObj.model_config = model_config;
    } else {
      // No model_config provided: try to infer one from jobs/new defaults for the model string
      const arch = getArchForModelName(model);
      if (arch) {
        const inferredModelConfig: any = { name_or_path: model, arch: arch.name };
        const assistant = arch.defaults?.['config.process[0].model.assistant_lora_path']?.[0];
        if (assistant) inferredModelConfig.assistant_lora_path = assistant;
        const quant = arch.defaults?.['config.process[0].model.quantize']?.[0];
        if (typeof quant !== 'undefined') inferredModelConfig.quantize = quant;
        const lowV = arch.defaults?.['config.process[0].model.low_vram']?.[0];
        if (typeof lowV !== 'undefined') inferredModelConfig.low_vram = lowV;
        const extras = arch.defaults?.['config.process[0].model.extras_name_or_path']?.[0];
        if (extras) inferredModelConfig.extras_name_or_path = extras;
        paramsObj.model_config = inferredModelConfig;
      } else {
        const resolved = resolveLocalModelPath(model);
        if (resolved) model = resolved;
      }
    }

    const evalJob = await prisma.evalJob.create({
      data: {
        dataset: absDataset,
        model: model_config?.name_or_path || model,
        params: JSON.stringify(paramsObj),
        status: 'queued',
      },
    });

    return NextResponse.json({ id: evalJob.id });
  } catch (error) {
    console.error('Error creating eval job:', error);
    return NextResponse.json({ error: 'Error creating eval job' }, { status: 500 });
  }
}

export async function GET(request: NextRequest) {
  // list recent eval jobs
  const jobs = await prisma.evalJob.findMany({ orderBy: { created_at: 'desc' }, take: 50 });
  return NextResponse.json({ jobs });
}