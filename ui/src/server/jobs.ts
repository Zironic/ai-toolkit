import fs from 'fs';
import path from 'path';
import { getTrainingFolder } from '@/server/settings';

export const isValidJobConfig = (raw: string): boolean => {
  try {
    const parsed = JSON.parse(raw);
    return (
      parsed !== null &&
      typeof parsed === 'object' &&
      Array.isArray(parsed.config?.process) &&
      parsed.config.process.length > 0 &&
      parsed.config.process[0] !== null &&
      typeof parsed.config.process[0] === 'object'
    );
  } catch {
    return false;
  }
};

export const saveDiskJobConfig = async (jobName: string, jobConfig: string): Promise<void> => {
  const trainingFolder = await getTrainingFolder();
  const jobFolder = path.join(trainingFolder, jobName);
  if (!fs.existsSync(jobFolder)) {
    fs.mkdirSync(jobFolder, { recursive: true });
  }
  const configPath = path.join(jobFolder, '.job_config.json');
  fs.writeFileSync(configPath, jobConfig, 'utf-8');
};

export const loadDiskJobConfig = async (jobName: string): Promise<string | null> => {
  const trainingFolder = await getTrainingFolder();
  const configPath = path.join(trainingFolder, jobName, '.job_config.json');
  if (!fs.existsSync(configPath)) {
    return null;
  }
  const raw = fs.readFileSync(configPath, 'utf-8');
  if (!isValidJobConfig(raw)) {
    console.warn(`Disk config for job "${jobName}" failed validation — ignoring`);
    return null;
  }
  return raw;
};
