'use client';

import { useEffect, useState, useRef } from 'react';
import { getEvalStatus, getEvalResult } from '@/utils/eval';
import { apiClient } from '@/utils/api';
import Link from 'next/link';

export default function EvalJobsList({ datasetName, refreshKey = 0, onNewFinishedJob }: { datasetName: string; refreshKey?: number; onNewFinishedJob?: (job:any) => void }) {
  const [jobs, setJobs] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const latestFinishedRef = useRef<string | null>(null);

  const fetchJobs = async () => {
    setLoading(true);
    try {
      const res = await apiClient.get('/api/eval_dataset');
      let j = res.data.jobs || [];
      // filter by dataset (may be absolute path); include those that end with datasetName
      j = j.filter((x: any) => (x.dataset || '').endsWith(datasetName) || x.dataset === datasetName);
      setJobs(j.slice(0, 20));

      // If a new finished job is present and a callback is provided, notify
      if (onNewFinishedJob) {
        const finished = j.filter((x:any) => x.status === 'finished');
        if (finished.length > 0) {
          // most recent finished by created_at
          finished.sort((a:any,b:any) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());
          const mostRecent = finished[0];
          if (mostRecent && mostRecent.id && latestFinishedRef.current !== mostRecent.id) {
            latestFinishedRef.current = mostRecent.id;
            try { onNewFinishedJob(mostRecent); } catch (e) { console.error('onNewFinishedJob callback error', e); }
          }
        }
      }
    } catch (e) {
      console.error('Error fetching eval jobs', e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchJobs();
    const iv = setInterval(fetchJobs, 15000);
    return () => clearInterval(iv);
  }, [datasetName, refreshKey]);

  return (
    <div className="mt-6">
      <h3 className="text-sm font-medium mb-2">Recent Evaluations</h3>
      <div className="bg-gray-50 dark:bg-gray-800 p-3 rounded">
        {loading && <div className="text-sm">Loading...</div>}
        {!loading && jobs.length === 0 && <div className="text-sm">No evaluations yet.</div>}
        {jobs.length > 0 && (
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left">
                <th className="p-2">ID</th>
                <th className="p-2">Status</th>
                <th className="p-2">Info</th>
                <th className="p-2">Result</th>
                <th className="p-2">Started</th>
              </tr>
            </thead>
            <tbody>
              {jobs.map((job) => (
                <tr key={job.id} className="border-t border-gray-200 dark:border-gray-700">
                  <td className="p-2 break-all">{job.id}</td>
                  <td className="p-2">{job.status}</td>
                  <td className="p-2 max-w-xs truncate">{job.info || ''}</td>
                  <td className="p-2">
                    {job.out_json ? (
                      <a className="text-blue-500" href={`/api/eval_dataset/${job.id}/result`} target="_blank" rel="noreferrer">View JSON</a>
                    ) : (
                      <span className="text-gray-400">-</span>
                    )}
                  </td>
                  <td className="p-2">{new Date(job.created_at).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
