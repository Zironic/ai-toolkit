from toolkit.config_modules import SampleConfig


def test_sample_start_step_defaults_to_zero_and_accepts_override():
    assert SampleConfig().sample_start_step == 0
    assert SampleConfig(sample_start_step=7).sample_start_step == 7
