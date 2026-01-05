import os
from pathlib import Path


def test_requirements_contains_control_deps():
    req_path = Path(__file__).resolve().parents[1] / 'requirements.txt'
    assert req_path.exists(), "requirements.txt not found"
    txt = req_path.read_text()
    assert 'controlnet_aux' in txt, "controlnet_aux should be in requirements"
    assert 'onnxruntime' in txt, "onnxruntime should be in requirements"
    assert 'easy_dwpose' in txt or 'easy_dwpose' in txt.lower(), "easy_dwpose (git link) should be in requirements"