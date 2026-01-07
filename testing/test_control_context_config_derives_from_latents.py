from toolkit.config_modules import DatasetConfig


def test_control_context_defaults_to_latents():
    cfg = DatasetConfig(dataset_path='.', cache_latents=True, cache_latents_to_disk=True)
    assert cfg.cache_control_contexts is True
    assert cfg.cache_control_contexts_to_disk is True


def test_control_context_can_override():
    cfg = DatasetConfig(dataset_path='.', cache_latents=True, cache_latents_to_disk=True, cache_control_contexts_to_disk=False)
    assert cfg.cache_control_contexts is True
    assert cfg.cache_control_contexts_to_disk is False
