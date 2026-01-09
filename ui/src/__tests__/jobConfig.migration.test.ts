import { migrateJobConfig, defaultJobConfig } from '../app/jobs/new/jobConfig';

describe('migrateJobConfig - controlnet migration', () => {
  test('copies train.controlnet_model to model.controlnet_name_or_path', () => {
    const cfg = JSON.parse(JSON.stringify(defaultJobConfig));
    (cfg as any).config.process[0].train.controlnet_model = '  my/controlnet-repo  ';

    const migrated = migrateJobConfig(cfg as any);

    expect((migrated as any).config.process[0].model.controlnet_name_or_path).toBe('my/controlnet-repo');
    expect((migrated as any).config.process[0].model.controlnet_enabled).toBe(true);
  });

  test('default job config includes controlnet defaults', () => {
    const cfg = JSON.parse(JSON.stringify(defaultJobConfig));
    expect(cfg.config.process[0].model.controlnet_enabled).toBe(false);
    expect(cfg.config.process[0].model.controlnet_name_or_path).toBe(null);
    expect(cfg.config.process[0].model.controlnet_streaming).toBe(false);
    expect(cfg.config.process[0].model.controlnet_offload_strategy).toBe('none');
  });

  test('migrateJobConfig adds performance defaults', () => {
    const cfg = JSON.parse(JSON.stringify(defaultJobConfig));
    delete (cfg as any).config.process[0].performance;
    const migrated = migrateJobConfig(cfg as any);
    expect((migrated as any).config.process[0].performance).toBeDefined();
    expect((migrated as any).config.process[0].performance.precise_gpu_timing).toBe(false);
    expect((migrated as any).config.process[0].performance.vram_diagnostics.enabled).toBe(false);
  });
});