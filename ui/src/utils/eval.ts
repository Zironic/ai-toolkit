import { apiClient } from './api';

export async function createEvalJob(payload: any) {
  const res = await apiClient.post('/api/eval_dataset', payload);
  return res.data;
}

export async function getEvalStatus(id: string) {
  const res = await apiClient.get(`/api/eval_dataset/${id}/status`);
  return res.data;
}

export async function getEvalResult(id: string) {
  const res = await apiClient.get(`/api/eval_dataset/${id}/result`);
  return res.data;
}
