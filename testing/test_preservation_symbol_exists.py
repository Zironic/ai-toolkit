def test_preservation_symbol_present():
    txt = open('extensions_built_in/sd_trainer/SDTrainer.py', encoding='utf-8').read()
    assert 'def _compute_and_apply_preservation_loss' in txt, 'Preservation loss function is missing in SDTrainer.py'