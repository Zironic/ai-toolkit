'use client';

import React, { Fragment, useState, useEffect } from 'react';
import { Dialog, Transition } from '@headlessui/react';
import { Button } from '@headlessui/react';
import { apiClient } from '@/utils/api';
import { createEvalJob } from '@/utils/eval';
import { defaultJobConfig } from '@/app/jobs/new/jobConfig';
import { SelectInput } from '@/components/formInputs';
import { groupedModelOptions, modelArchs } from '@/app/jobs/new/options';

export default function EvalDatasetModal({ datasetName, isOpen, onClose, onStarted }:
  { datasetName: string; isOpen: boolean; onClose: () => void; onStarted?: (id: string) => void }) {
  const defaultModel = defaultJobConfig.config.process[0].model || { name_or_path: 'Tongyi-MAI/Z-Image-Turbo', arch: 'zimage:turbo' };
  const [model, setModel] = useState<string>(defaultModel.name_or_path || 'Tongyi-MAI/Z-Image-Turbo');
  const [modelConfig, setModelConfig] = useState<any>({ ...defaultModel });
  const [modelArch, setModelArch] = useState<string>(defaultModel.arch || 'zimage:turbo');
  const [useJobModel, setUseJobModel] = useState<boolean>(false);
  const [samplesPerImage, setSamplesPerImage] = useState<number>(8);
  const [fixedNoiseStd, setFixedNoiseStd] = useState<number>(0.6);
  const [debugCaptions, setDebugCaptions] = useState<boolean>(false);
  const [ablationCompare, setAblationCompare] = useState<boolean>(false);
  const [device, setDevice] = useState<string>('cuda');

  // sample_fraction and max_samples are enforced server-side and removed from the UI


  const applyModelArchDefaults = (archName: string) => {
    const arch = modelArchs.find(a => a.name === archName);
    if (!arch) return;
    // pick model name default
    const namePath = arch.defaults?.['config.process[0].model.name_or_path']?.[0];
    const quant = arch.defaults?.['config.process[0].model.quantize']?.[0];
    const quant_te = arch.defaults?.['config.process[0].model.quantize_te']?.[0];
    const qtype = arch.defaults?.['config.process[0].model.qtype']?.[0];
    const qtype_te = arch.defaults?.['config.process[0].model.qtype_te']?.[0];

    const newCfg: any = { ...modelConfig };
    if (namePath) {
      newCfg.name_or_path = namePath;
      setModel(namePath);
    }
    if (typeof quant !== 'undefined') newCfg.quantize = quant;
    if (typeof quant_te !== 'undefined') newCfg.quantize_te = quant_te;
    if (typeof qtype !== 'undefined') newCfg.qtype = qtype;
    if (typeof qtype_te !== 'undefined') newCfg.qtype_te = qtype_te;
    newCfg.arch = arch.name;
    setModelConfig(newCfg);
  };

  // Apply defaults for Z-Image on mount
  useEffect(() => {
    applyModelArchDefaults('zimage:turbo');
    setModelArch('zimage:turbo');
    setUseJobModel(false);
  }, []);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    setLoading(true);
    setError(null);
    try {
      const payload: any = {
        dataset_path: datasetName,
        model: model,
        device: device,
        samples_per_image: samplesPerImage,
        fixed_noise_std: fixedNoiseStd,
        debug_captions: debugCaptions,
        ablation_compare: ablationCompare,
      };
      if (useJobModel && modelConfig) {
        payload.model_config = modelConfig;
        // ensure model string is consistent
        payload.model = modelConfig.name_or_path || payload.model;
      }

      const res = await createEvalJob(payload);
      if (res?.id) {
        if (onStarted) onStarted(res.id);
        onClose();
      } else {
        setError('Failed to create eval job');
      }
    } catch (e: any) {
      console.error('Error creating eval job', e);
      setError(e?.message || 'Failed to enqueue eval job');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Transition show={isOpen} as={Fragment}>
      <Dialog as="div" className="relative z-50" onClose={onClose}>
        <Transition.Child
          as={Fragment}
          enter="ease-out duration-300"
          enterFrom="opacity-0"
          enterTo="opacity-100"
          leave="ease-in duration-200"
          leaveFrom="opacity-100"
          leaveTo="opacity-0"
        >
          <div className="fixed inset-0 bg-black/30" aria-hidden="true" />
        </Transition.Child>

        <div className="fixed inset-0 flex items-center justify-center p-4">
          <Transition.Child
            as={Fragment}
            enter="ease-out duration-300"
            enterFrom="opacity-0 scale-95"
            enterTo="opacity-100 scale-100"
            leave="ease-in duration-200"
            leaveFrom="opacity-100 scale-100"
            leaveTo="opacity-0 scale-95"
          >
            <Dialog.Panel className="mx-auto max-w-lg rounded bg-white dark:bg-gray-900 p-6 shadow-lg">
              <Dialog.Title className="text-lg font-semibold mb-4">Run Dataset Evaluation</Dialog.Title>

              <div className="space-y-3">
                <div>
                  <label className="block text-sm font-medium mb-1">Dataset</label>
                  <div className="text-sm text-gray-700 dark:text-gray-300">{datasetName}</div>
                </div>
                <SelectInput
                  label="Model Architecture"
                  value={modelArch}
                  onChange={(value) => { setModelArch(value); applyModelArchDefaults(value); setUseJobModel(false); }}
                  options={groupedModelOptions}
                />
                <div>
                  <label className="block text-sm font-medium mb-1">Name or Path</label>
                  <input value={model} onChange={(e) => { setModel(e.target.value); setUseJobModel(false); }} className="w-full rounded bg-gray-100 dark:bg-gray-800 p-2" />
                </div>
                <div className="flex items-center gap-2">
                  <input type="checkbox" checked={useJobModel} onChange={(e) => { setUseJobModel(e.target.checked); if (e.target.checked) { setModel(defaultModel.name_or_path || 'default'); setModelConfig({ ...defaultModel }); setModelArch(defaultModel.arch || ''); } }} id="useJobModel" />
                  <label htmlFor="useJobModel" className="text-sm">Use training job model configuration</label>
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">Samples per image</label>
                  <input type="number" value={samplesPerImage} min={1} onChange={(e) => setSamplesPerImage(Number(e.target.value))} className="w-full rounded bg-white dark:bg-gray-800 p-2" />
                  <div className="text-xs text-gray-500 mt-1">Number of independent stochastic forward passes to average per image (default: 8). Batch size is fixed to 1 and sample_fraction is fixed to 1.0 for per-example loss evaluation.</div>
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">Fixed noise std</label>
                  <input type="number" value={fixedNoiseStd} min={0} max={1} step={0.01} onChange={(e) => setFixedNoiseStd(Number(e.target.value))} className="w-full rounded bg-white dark:bg-gray-800 p-2" />
                  <div className="text-xs text-gray-500 mt-1">If set (0.0–1.0), use fixed noise magnitude instead of sampling timesteps (default: 0.6).</div>
                </div>
                <div className="flex items-center gap-2">
                  <input type="checkbox" checked={debugCaptions} onChange={(e) => setDebugCaptions(e.target.checked)} id="debugCaptions" />
                  <label htmlFor="debugCaptions" className="text-sm">Enable caption debug logging</label>
                </div>
                <div className="flex items-center gap-2">
                  <input type="checkbox" checked={ablationCompare} onChange={(e) => setAblationCompare(e.target.checked)} id="ablationCompare" />
                  <label htmlFor="ablationCompare" className="text-sm">Enable ablation compare (run with captions and blank captions and store their difference as the loss)</label>
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">Device</label>
                  <input type="text" value={"cuda"} disabled className="w-full rounded bg-gray-100 dark:bg-gray-800 p-2 opacity-50 cursor-not-allowed" />
                  <div className="text-xs text-gray-500 mt-1">Device is fixed to <code>cuda</code> (GPU) for evaluation.</div>
                </div>
                {error && <div className="text-sm text-red-600">{error}</div>}
              </div>

              <div className="mt-6 flex justify-end gap-2">
                <Button className="px-4 py-2 bg-gray-200 text-gray-900 rounded" onClick={onClose}>Cancel</Button>
                <Button className="px-4 py-2 bg-blue-600 text-white rounded" onClick={run} disabled={loading}>{loading ? 'Starting...' : 'Run'}</Button>
              </div>
            </Dialog.Panel>
          </Transition.Child>
        </div>
      </Dialog>
    </Transition>
  );
}
