from toolkit.config_modules import validate_configs, TrainConfig, ModelConfig, SaveConfig, DatasetConfig


def test_validate_configs_allows_dop_with_caching():
    # Should NOT raise now that DOP caching is implemented
    train = TrainConfig()
    train.diff_output_preservation = True

    model = ModelConfig(name_or_path='dummy_model')
    save = SaveConfig()

    # single dataset with cache_text_embeddings True
    ds = DatasetConfig(folder_path='.', cache_text_embeddings=True)

    # Should not raise
    validate_configs(train, model, save, [ds])


def test_validate_configs_requires_all_datasets_cached():
    # When caching is enabled, all dataset entries must have cache_text_embeddings True
    train = TrainConfig()
    train.diff_output_preservation = True

    model = ModelConfig(name_or_path='dummy_model')
    save = SaveConfig()

    ds1 = DatasetConfig(folder_path='.', cache_text_embeddings=True)
    ds2 = DatasetConfig(folder_path='.', cache_text_embeddings=False)

    try:
        validate_configs(train, model, save, [ds1, ds2])
        raise AssertionError("Expected ValueError when not all datasets have cache_text_embeddings")
    except ValueError as e:
        assert 'All datasets must have cache_text_embeddings set to True' in str(e)