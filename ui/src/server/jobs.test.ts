import { describe, it, expect } from 'vitest';
import { isValidJobConfig } from './jobs';

const validConfig = JSON.stringify({
  config: {
    process: [
      { type: 'sd_trainer', train: { steps: 1000 } },
    ],
  },
});

describe('isValidJobConfig', () => {
  it('accepts a well-formed job config', () => {
    expect(isValidJobConfig(validConfig)).toBe(true);
  });

  it('rejects empty string', () => {
    expect(isValidJobConfig('')).toBe(false);
  });

  it('rejects non-JSON', () => {
    expect(isValidJobConfig('not json')).toBe(false);
  });

  it('rejects null literal', () => {
    expect(isValidJobConfig('null')).toBe(false);
  });

  it('rejects empty object', () => {
    expect(isValidJobConfig('{}')).toBe(false);
  });

  it('rejects missing config.process', () => {
    expect(isValidJobConfig(JSON.stringify({ config: {} }))).toBe(false);
  });

  it('rejects empty process array', () => {
    expect(isValidJobConfig(JSON.stringify({ config: { process: [] } }))).toBe(false);
  });

  it('rejects process array with non-object entry', () => {
    expect(isValidJobConfig(JSON.stringify({ config: { process: [null] } }))).toBe(false);
  });

  it('rejects process array with primitive entry', () => {
    expect(isValidJobConfig(JSON.stringify({ config: { process: [42] } }))).toBe(false);
  });

  it('rejects config with no process key at all', () => {
    expect(isValidJobConfig(JSON.stringify({ config: { name: 'test' } }))).toBe(false);
  });
});
