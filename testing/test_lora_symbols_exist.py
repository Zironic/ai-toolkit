def test_lora_symbols_present():
    txt = open('extensions_built_in/sd_trainer/SDTrainer.py', encoding='utf-8').read()
    assert 'LoRA' in txt or 'assistant_lora_path' in txt, 'LoRA-related symbols not found in SDTrainer.py'
    ui_txt = open('ui/src/app/jobs/new/SimpleJob.tsx', encoding='utf-8').read()
    assert 'LoRA' in ui_txt, 'LoRA UI option not present in SimpleJob.tsx'