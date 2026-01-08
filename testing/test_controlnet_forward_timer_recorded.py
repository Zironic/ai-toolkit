import torch
from types import SimpleNamespace
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer


def make_job_and_cfg():
    job = SimpleNamespace()
    job.training_folder = "./tmp_training"
    job.raw_config = {}
    job.name = 'test_job'
    job.log_dir = None
    job.training_seed = None
    job.meta = {}
    cfg = {}
    cfg['model'] = {'name_or_path': 'dummy_model'}
    cfg['train'] = {'steps': 1}
    return job, cfg


class DummyAdapter(torch.nn.Module):
    def forward(self, x):
        # return a list of one tensor to mimic T2IAdapter behavior
        return (torch.zeros((x.shape[0], 4)),)


def test_controlnet_forward_timer_recorded():
    job, cfg = make_job_and_cfg()
    trainer = SDTrainer(0, job, cfg)

    # Ensure CPU device for unit test
    trainer.device_torch = torch.device('cpu')
    trainer.train_config.controlnet_offload_strategy = 'none'
    trainer.assistant_adapter = None

    trainer.adapter = DummyAdapter()
    adapter_images = torch.zeros((1, 3, 64, 64))

    # Simulate the code path that computes adapter residuals
    with torch.set_grad_enabled(trainer.adapter is not None):
        from toolkit.controlnet_offload import offload_adapter, bring_adapter
        strategy = trainer.train_config.controlnet_offload_strategy
        try:
            if strategy == 'accelerate':
                bring_adapter(trainer.adapter, device=trainer.device_torch, strategy='accelerate')

            adapter_images_dev = adapter_images.to(trainer.device_torch)

            with trainer.timer('encode_adapter'):
                with trainer.timer('controlnet_forward'):
                    down_block_additional_residuals = trainer.adapter(adapter_images_dev)

                # apply multiplier path similar to trainer code
                if trainer.assistant_adapter:
                    down_block_additional_residuals = [sample.to(dtype=torch.float32).detach() * 1.0 for sample in down_block_additional_residuals]
                else:
                    down_block_additional_residuals = [sample.to(dtype=torch.float32) * 1.0 for sample in down_block_additional_residuals]

        finally:
            try:
                if strategy in ('accelerate', 'manual_swap'):
                    with trainer.timer('controlnet_offload'):
                        offload_adapter(trainer.adapter, strategy=strategy)
            except Exception:
                pass

    assert 'controlnet_forward' in trainer.timer.timers
