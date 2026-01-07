from toolkit.config_modules import DatasetConfig


def test_cache_latents_to_disk_enables_other_caches_default():
    cfg = DatasetConfig(cache_latents_to_disk=True)

    assert cfg.cache_latents_to_disk is True
    # cache_latents should default to True if not explicitly provided
    assert cfg.cache_latents is True
    # control contexts (in-memory) should default to True
    assert cfg.cache_control_contexts is True
    # control contexts on-disk should default to True
    assert cfg.cache_control_contexts_to_disk is True
    # clip vision disk cache should remain at its default (False) unless explicitly set
    assert cfg.cache_clip_vision_to_disk is False
    # text embeddings caching should default to True
    assert cfg.cache_text_embeddings is True


def test_explicit_overrides_are_respected():
    cfg = DatasetConfig(
        cache_latents_to_disk=True,
        cache_control_contexts=False,
        cache_clip_vision_to_disk=False,
        cache_text_embeddings=False,
    )

    assert cfg.cache_control_contexts is False
    assert cfg.cache_clip_vision_to_disk is False
    assert cfg.cache_text_embeddings is False
    # cache_latents should still default to True if not explicitly set
    assert cfg.cache_latents is True


def test_explicit_disable_of_cache_latents_respected():
    # If user explicitly disables cache_latents, it should be honored even when
    # cache_latents_to_disk is True
    cfg = DatasetConfig(cache_latents_to_disk=True, cache_latents=False)
    assert cfg.cache_latents is False
    # but disk flag remains True
    assert cfg.cache_latents_to_disk is True