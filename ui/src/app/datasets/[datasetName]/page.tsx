'use client';

import { useEffect, useState, use, useMemo } from 'react';
import { LuImageOff, LuLoader, LuBan } from 'react-icons/lu';
import { FaChevronLeft } from 'react-icons/fa';
import DatasetImageCard from '@/components/DatasetImageCard';
import { Button } from '@headlessui/react';
import AddImagesModal, { openImagesModal } from '@/components/AddImagesModal';
import EvalDatasetModal from '@/components/EvalDatasetModal';
import EvalJobsList from '@/components/EvalJobsList';
import { TopBar, MainContent } from '@/components/layout';
import { apiClient } from '@/utils/api';
import FullscreenDropOverlay from '@/components/FullscreenDropOverlay';

export default function DatasetPage({ params }: { params: { datasetName: string } }) {
  const [imgList, setImgList] = useState<{ img_path: string }[]>([]);
  const usableParams = use(params as any) as { datasetName: string };
  const datasetName = usableParams.datasetName;
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [showEvalModal, setShowEvalModal] = useState(false);
  const [evalRefreshKey, setEvalRefreshKey] = useState(0);
  const [evalMap, setEvalMap] = useState<Record<string, { raw?: number; norm?: number }>>({});

  // modelsList: array of { label, jobId }
  const [modelsList, setModelsList] = useState<Array<{ label: string; jobId: string }>>([]);
  const [selectedEvalJobId, setSelectedEvalJobId] = useState<string | null>(null);
  const [loadingModels, setLoadingModels] = useState(false);

  // Load list of evaluated models (finished jobs for this dataset) and pick one to display
  const loadEvaluatedModels = async (dsName: string) => {
    setLoadingModels(true);
    try {
      const res = await apiClient.get('/api/eval_dataset');
      const jobs = res.data?.jobs || [];
      // filter finished jobs that reference this dataset exactly or by suffix
      const matched = jobs.filter((j: any) => j.status === 'finished' && j.dataset && (String(j.dataset) === dsName || String(j.dataset).endsWith(dsName) || String(j.dataset).includes(dsName)));
      // group by model name and pick most recent job for each model
      const perModel: Record<string, any> = {};
      for (const j of matched) {
        // prefer explicit job.model, but fall back to parsing params.model_config.name_or_path
        let modelName = '';
        try {
          modelName = (j.model || '').toString();
          if (!modelName && j.params) {
            const p = JSON.parse(j.params || '{}');
            if (p && p.model_config && p.model_config.name_or_path) modelName = String(p.model_config.name_or_path);
          }
        } catch (e) {
          modelName = (j.model || '').toString();
        }
        if (!modelName) continue;
        const existing = perModel[modelName];
        if (!existing || new Date(j.created_at) > new Date(existing.created_at)) {
          perModel[modelName] = j;
        }
      }
      const list = Object.entries(perModel).map(([m, j]) => ({ label: m, jobId: j.id }));
      // sort by label for deterministic order
      list.sort((a, b) => a.label.localeCompare(b.label));

      // pick the latest finished job overall as a default selection (if any)
      const latestJobOverall = matched.sort((a: any, b: any) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())[0];

      // If the latest finished job isn't represented in the models list (no model label), add it as '(Latest)'
      if (latestJobOverall) {
        const alreadyHas = list.find(l => l.jobId === latestJobOverall.id);
        const latestLabel = (latestJobOverall.model && String(latestJobOverall.model)) || '(Latest)';
        if (!alreadyHas) {
          list.unshift({ label: latestLabel, jobId: latestJobOverall.id });
        }
      }

      setModelsList(list);
      if (latestJobOverall) {
        setSelectedEvalJobId(latestJobOverall.id);
        // load results for that job
        await fetchResultsForJob(latestJobOverall.id, dsName);
      } else {
        setSelectedEvalJobId(null);
        setEvalMap({});
      }
    } catch (e) {
      console.error('Failed to load evaluated models:', e);
      setModelsList([]);
      setSelectedEvalJobId(null);
      setEvalMap({});
    }
    setLoadingModels(false);
  };

  const fetchResultsForJob = async (jobId: string | null, dsName?: string) => {
    if (!jobId) {
      setEvalMap({});
      return;
    }
    try {
      const resJson = await apiClient.get(`/api/eval_dataset/${jobId}/result`);
      const j = resJson.data;
      if (!j || !j.datasets) {
        setEvalMap({});
        return;
      }
      // find dataset key inside report (match by last path segment)
      const datasetKey = Object.keys(j.datasets).find(k => {
        if (!dsName) return true;
        return k.endsWith(dsName) || k.includes(dsName) || k === dsName;
      });
      if (!datasetKey) {
        setEvalMap({});
        return;
      }
      const itemStats = j.datasets[datasetKey]?.item_stats || {};
      const map: Record<string, { raw?: number; norm?: number }> = {};
      for (const [pathKey, stats] of Object.entries(itemStats)) {
        // Support both forward and back slashes so Windows absolute paths are handled correctly
        const parts = String(pathKey).split(/[\\/]/);
        const b = parts[parts.length - 1];
        const s: any = stats as any;
        const raw = typeof s.average_loss_raw !== 'undefined' && s.average_loss_raw !== null ? Number(s.average_loss_raw) : undefined;
        const norm = typeof s.average_loss !== 'undefined' && s.average_loss !== null ? Number(s.average_loss) : undefined;
        map[b] = { raw, norm };
      }
      console.log('Loaded eval map for job', jobId, map);
      setEvalMap(map);
    } catch (e) {
      console.error('Failed to load eval results for job', jobId, e);
      setEvalMap({});
    }
  };


  useEffect(() => {
    // reload the model list and selected evaluation when datasetName or evalRefreshKey changes
    if (datasetName) loadEvaluatedModels(datasetName);
  }, [datasetName, evalRefreshKey]);

  // whenever the user changes selection, fetch that job's results
  useEffect(() => {
    if (selectedEvalJobId) {
      fetchResultsForJob(selectedEvalJobId, datasetName);
    } else {
      setEvalMap({});
    }
  }, [selectedEvalJobId]);

  const refreshImageList = (dbName: string) => {
    setStatus('loading');
    console.log('Fetching images for dataset:', dbName);
    apiClient
      .post('/api/datasets/listImages', { datasetName: dbName })
      .then((res: any) => {
        const data = res.data;
        console.log('Images:', data.images);

        // Defensive handling: ensure we have an array of images and each image path is a string.
        let images: { img_path: string }[] = [];
        if (Array.isArray(data.images)) {
          images = data.images
            .map((img: any) => {
              // some backends may accidentally return parsed objects; coerce to string safely
              const imgPath = img && typeof img.img_path === 'string' ? img.img_path : String(img?.img_path ?? img?.path ?? img ?? '');
              return { img_path: imgPath };
            })
            .filter((i: { img_path: string }) => i.img_path.length > 0);

          // sort safely
          images.sort((a: { img_path: string }, b: { img_path: string }) => a.img_path.localeCompare(b.img_path));
        }

        setImgList(images);
        setStatus('success');
      })
      .catch(error => {
        console.error('Error fetching images:', error);
        setStatus('error');
      });
  };
  useEffect(() => {
    if (datasetName) {
      refreshImageList(datasetName);
    }
  }, [datasetName]);

  const PageInfoContent = useMemo(() => {
    let icon = null;
    let text = '';
    let subtitle = '';
    let showIt = false;
    let bgColor = '';
    let textColor = '';
    let iconColor = '';

    if (status == 'loading') {
      icon = <LuLoader className="animate-spin w-8 h-8" />;
      text = 'Loading Images';
      subtitle = 'Please wait while we fetch your dataset images...';
      showIt = true;
      bgColor = 'bg-gray-50 dark:bg-gray-800/50';
      textColor = 'text-gray-900 dark:text-gray-100';
      iconColor = 'text-gray-500 dark:text-gray-400';
    }
    if (status == 'error') {
      icon = <LuBan className="w-8 h-8" />;
      text = 'Error Loading Images';
      subtitle = 'There was a problem fetching the images. Please try refreshing the page.';
      showIt = true;
      bgColor = 'bg-red-50 dark:bg-red-950/20';
      textColor = 'text-red-900 dark:text-red-100';
      iconColor = 'text-red-600 dark:text-red-400';
    }
    if (status == 'success' && imgList.length === 0) {
      icon = <LuImageOff className="w-8 h-8" />;
      text = 'No Images Found';
      subtitle = 'This dataset is empty. Click "Add Images" to get started.';
      showIt = true;
      bgColor = 'bg-gray-50 dark:bg-gray-800/50';
      textColor = 'text-gray-900 dark:text-gray-100';
      iconColor = 'text-gray-500 dark:text-gray-400';
    }

    if (!showIt) return null;

    return (
      <div
        className={`mt-10 flex flex-col items-center justify-center py-16 px-8 rounded-xl border-2 border-gray-700 border-dashed ${bgColor} ${textColor} mx-auto max-w-md text-center`}
      >
        <div className={`${iconColor} mb-4`}>{icon}</div>
        <h3 className="text-lg font-semibold mb-2">{text}</h3>
        <p className="text-sm opacity-75 leading-relaxed">{subtitle}</p>
      </div>
    );
  }, [status, imgList.length]);

  return (
    <>
      {/* Fixed top bar */}
      <TopBar>
        <div>
          <Button className="text-gray-500 dark:text-gray-300 px-3 mt-1" onClick={() => history.back()}>
            <FaChevronLeft />
          </Button>
        </div>
        <div>
          <h1 className="text-lg">Dataset: {String(datasetName)}</h1>
        </div>
        <div className="flex-1"></div>
        <div className="flex items-center gap-2">
          <Button
            className="text-gray-200 bg-slate-600 px-3 py-1 rounded-md"
            onClick={() => openImagesModal(datasetName, () => refreshImageList(datasetName))}
          >
            Add Images
          </Button>
          {/* Model evaluation selector */}
          <div className="text-sm text-gray-200 bg-gray-800 rounded px-2 py-1 flex items-center">
            <label className="mr-2 text-xs text-gray-300">Eval:</label>
            <select
              value={selectedEvalJobId || ''}
              onChange={e => setSelectedEvalJobId(e.target.value || null)}
              className="bg-gray-800 text-white text-sm outline-none rounded px-1 py-0.5 appearance-none"
            >
              <option className="bg-gray-800 text-white" value="">None</option>
              {modelsList.map(m => (
                <option key={m.jobId} className="bg-gray-800 text-white" value={m.jobId}>{m.label}</option>
              ))}
            </select>
          </div>
          <Button
            className="text-white bg-emerald-600 px-3 py-1 rounded-md"
            onClick={() => setShowEvalModal(true)}
          >
            Run Evaluation
          </Button>
        </div>
      </TopBar>
      <MainContent>
        {PageInfoContent}
        {status === 'success' && imgList.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {imgList.map((img, idx) => {
              const parts = String(img.img_path).split(/[\\/]/);
              const bname = parts[parts.length - 1];
              const e = (evalMap as any)[bname] || {};
              const rawLoss = typeof e.raw !== 'undefined' ? e.raw : null;
              const normLoss = typeof e.norm !== 'undefined' ? e.norm : null;
              return (
                <DatasetImageCard
                  key={img.img_path || String(idx)}
                  alt="image"
                  imageUrl={String(img.img_path)}
                  onDelete={() => refreshImageList(String(datasetName))}
                  rawLoss={rawLoss}
                  normLoss={normLoss}
                />
              );
            })}
          </div>
        )}

        <div className="pt-6 px-4">
          <EvalJobsList
            datasetName={datasetName}
            refreshKey={evalRefreshKey}
            onNewFinishedJob={(job) => {
              // automatically switch to the newly finished job if it's for this dataset
              setSelectedEvalJobId(job.id);
            }}
          />
        </div>
      </MainContent>

      <AddImagesModal />
      <FullscreenDropOverlay
        datasetName={datasetName}
        onComplete={() => refreshImageList(datasetName)}
      />

      <EvalDatasetModal datasetName={datasetName} isOpen={showEvalModal} onClose={() => setShowEvalModal(false)} onStarted={(id) => { setShowEvalModal(false); setEvalRefreshKey(k=>k+1); }} />
    </>
  );
}
