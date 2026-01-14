def test_pipeline_symbols_present():
    txt = open('toolkit/model_utils.py', encoding='utf-8').read()
    assert 'load_pipeline' in txt or 'StableDiffusion' in txt or 'load_model' in txt, 'Pipeline/model loader symbols not found in toolkit/model_utils.py'