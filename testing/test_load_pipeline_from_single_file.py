import os
import tempfile
import tools.visualize_lora_attention as vla


def test_load_pipeline_from_single_file(monkeypatch, tmp_path):
    # create a fake file to represent single-file model
    p = tmp_path / "model.safetensors"
    p.write_text("fake")

    called = {'from_single': False}

    class FakePipe:
        pass

    def fake_from_single_file(path, torch_dtype=None):
        called['from_single'] = True
        return FakePipe()

    monkeypatch.setattr('diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.from_single_file', fake_from_single_file)

    pipe = vla.load_pipeline(str(p))
    assert called['from_single'] is True
    assert isinstance(pipe, FakePipe)
