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


def test_after_unet_predict_handles_hooks_and_attentions():
    job, cfg = make_job_and_cfg()
    trainer = SDTrainer(0, job, cfg)

    # enable attention align weight so the method does work
    trainer.train_config.attention_align_weight = 1.0

    # create fake hook handles that have a remove() method
    class DummyHandle:
        def remove(self):
            return None

    trainer._attn_hook_handles = [DummyHandle(), DummyHandle()]

    # create a small attention tensor list: list of tensors with shape [B, H, T, S]
    # use small dimensions to be fast
    a = torch.randn((1, 2, 4, 4))
    trainer._collected_attentions = [a, a]

    # set a last_batch_for_attn with latents so sizes are available
    trainer._last_batch_for_attn = SimpleNamespace(latents=torch.zeros((1, 4, 8, 8)))

    # call and ensure no exception and loss is set (float)
    trainer.after_unet_predict()
    assert hasattr(trainer, '_latest_attention_align_loss')
    try:
        float(trainer._latest_attention_align_loss)
    except Exception:
        raise AssertionError('after_unet_predict did not set a numeric _latest_attention_align_loss')
