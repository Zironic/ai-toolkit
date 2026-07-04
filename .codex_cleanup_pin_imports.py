from pathlib import Path
for file in [r'tests/test_pin_quantized_weights.py', r'tests/test_pinned_budget_cycle.py']:
    path = Path(file)
    text = path.read_text(encoding='utf-8')
    text = text.replace('import os\n', '')
    text = text.replace('from unittest import mock\n\n', '')
    path.write_text(text, encoding='utf-8')
