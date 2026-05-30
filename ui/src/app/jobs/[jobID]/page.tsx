'use client';

import { useMemo, use } from 'react';
import { FaChevronLeft } from 'react-icons/fa';
import { MdDashboard, MdImage, MdShowChart, MdCode, MdBarChart, MdCompareArrows, MdLayers } from 'react-icons/md';
import { Button } from '@headlessui/react';
import { TopBar, MainContent } from '@/components/layout';
import useJob from '@/hooks/useJob';
import SampleImages, { SampleImagesMenu } from '@/components/SampleImages';
import JobOverview from '@/components/JobOverview';
import { redirect, useRouter, useSearchParams } from 'next/navigation';
import JobActionBar from '@/components/JobActionBar';
import JobConfigViewer from '@/components/JobConfigViewer';
import JobLossGraph from '@/components/JobLossGraph';
import JobMetricsGraph from '@/components/JobMetricsGraph';
import JobMetricsCompareGraph from '@/components/JobMetricsCompareGraph';
import DepthPreviews from '@/components/DepthPreviews';
import { Job } from '@prisma/client';
import { JobConfig } from '@/types';

type PageKey = 'overview' | 'samples' | 'depth_previews' | 'config' | 'metrics' | 'metrics_compare' | 'loss_log';
const PAGE_KEYS = new Set<PageKey>(['overview', 'samples', 'depth_previews', 'config', 'metrics', 'metrics_compare', 'loss_log']);

interface Page {
  name: string;
  value: PageKey;
  icon?: React.ComponentType<{ className?: string }>;
  component: React.ComponentType<{ job: Job }>;
  menuItem?: React.ComponentType<{ job?: Job | null }> | null;
  mainCss?: string;
  jobTypes?: string[];
  /** Hide the tab unless the predicate (run against the loaded job) returns true. */
  condition?: (job: Job) => boolean;
}

function hasDepthPreviews(job: Job): boolean {
  if (!job.job_config) return false;
  try {
    const cfg = JSON.parse(job.job_config) as JobConfig;
    return (cfg.config?.process?.[0]?.depth_consistency?.preview_every ?? 0) > 0;
  } catch {
    return false;
  }
}

const pages: Page[] = [
  {
    name: 'Overview',
    value: 'overview',
    icon: MdDashboard,
    component: JobOverview,
    mainCss: 'pt-24',
  },
  {
    name: 'Samples',
    value: 'samples',
    icon: MdImage,
    component: SampleImages,
    menuItem: SampleImagesMenu,
    mainCss: 'pt-24',
  },
  {
    name: 'Loss Graph',
    value: 'loss_log',
    icon: MdShowChart,
    component: JobLossGraph,
    mainCss: 'pt-24 pb-4',
    jobTypes: ['train'],
  },
  {
    name: 'Depth Previews',
    value: 'depth_previews',
    icon: MdLayers,
    component: DepthPreviews,
    mainCss: 'pt-24',
    condition: hasDepthPreviews,
  },
  {
    name: 'Metrics',
    value: 'metrics',
    icon: MdBarChart,
    component: JobMetricsGraph,
    mainCss: 'pt-24',
  },
  {
    name: 'Compare Jobs',
    value: 'metrics_compare',
    icon: MdCompareArrows,
    component: JobMetricsCompareGraph,
    mainCss: 'pt-24',
  },
  {
    name: 'Config File',
    value: 'config',
    icon: MdCode,
    component: JobConfigViewer,
    mainCss: 'pt-[80px] px-0 pb-0',
  },
];

export default function JobPage({ params }: { params: { jobID: string } }) {
  const usableParams = use(params as any) as { jobID: string };
  const jobID = usableParams.jobID;
  const { job, status, refreshJob } = useJob(jobID, 5000);

  // Tab selection lives in the URL (`?tab=…`) so refresh + tab-switch-and-return
  // both preserve it, and the URL is shareable. Per-tab interior state (filters,
  // sort, etc.) is the tab component's own responsibility — see DepthPreviews
  // for the pattern.
  const router = useRouter();
  const searchParams = useSearchParams();
  const rawTab = searchParams.get('tab');
  const pageKey: PageKey = rawTab && PAGE_KEYS.has(rawTab as PageKey) ? (rawTab as PageKey) : 'overview';
  const setPageKey = (k: PageKey) => {
    const params = new URLSearchParams(searchParams.toString());
    params.set('tab', k);
    router.replace(`?${params.toString()}`, { scroll: false });
  };

  const visiblePages = useMemo(() => (job ? pages.filter(p => !p.condition || p.condition(job)) : pages.filter(p => !p.condition)), [job]);
  // If the previously selected tab no longer applies (e.g. preview_every was
  // turned off after the user landed on it), bounce back to overview.
  const page = visiblePages.find(p => p.value === pageKey) ?? visiblePages[0];
  const effectivePageKey = page?.value ?? 'overview';

  return (
    <>
      {/* Fixed top bar */}
      <TopBar>
        <div className="flex-shrink-0">
          <Button className="text-gray-500 dark:text-gray-300 pl-0 pr-1 sm:px-3 mt-1" onClick={() => redirect('/jobs')}>
            <FaChevronLeft />
          </Button>
        </div>
        <div className="min-w-0 flex-shrink">
          <h1 className="text-base sm:text-lg truncate">Job: {job?.name}</h1>
        </div>
        <div className="flex-1"></div>
        {job && (
          <JobActionBar
            job={job}
            onRefresh={refreshJob}
            hideView
            afterDelete={() => {
              redirect('/jobs');
            }}
            autoStartQueue={true}
          />
        )}
      </TopBar>
      <MainContent className={page?.mainCss}>
        {status === 'loading' && job == null && <p>Loading...</p>}
        {status === 'error' && job == null && <p>Error fetching job</p>}
        {/* All tabs mount once and stay mounted; we hide inactive ones with
            display:none rather than unmounting so per-tab local state (zoom,
            selected series, scroll position, etc.) survives a tab switch.
            Tabs that need to persist across *refresh* still mirror to the
            URL on their own (see DepthPreviews for the pattern). */}
        {job && (
          <>
            {visiblePages.map(p => {
              const Component = p.component;
              const isActive = p.value === effectivePageKey;
              return (
                <div key={p.value} className={isActive ? 'contents' : 'hidden'} aria-hidden={!isActive}>
                  <Component job={job} />
                </div>
              );
            })}
          </>
        )}
      </MainContent>
      <div className="bg-gray-800 absolute top-12 left-0 w-full h-8 flex items-center px-0 sm:px-2 text-sm sm:overflow-x-auto whitespace-nowrap">
        {visiblePages.map(p => {
          if (p.jobTypes && job && !p.jobTypes.includes(job.job_type ?? 'train')) {
            return null;
          }
          const Icon = p.icon;
          return (
            <Button
              key={p.value}
              onClick={() => setPageKey(p.value)}
              className={`flex-1 sm:flex-initial justify-center px-2 sm:px-4 py-1 h-8 flex items-center gap-1.5 sm:flex-shrink-0 ${p.value === effectivePageKey ? 'bg-gray-300 dark:bg-gray-700 text-white' : ''}`}
            >
              {Icon && <Icon className="text-sm" />}
              <span className="hidden sm:inline">{p.name}</span>
            </Button>
          );
        })}
        {page?.menuItem && (
          <>
            <div className="hidden sm:block flex-grow"></div>
            <page.menuItem job={job} />
          </>
        )}
      </div>
    </>
  );
}
