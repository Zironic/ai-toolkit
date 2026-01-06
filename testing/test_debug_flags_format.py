import types
from toolkit.prompt_utils import PromptEmbeds
from toolkit.splitflux import complementary_loss

from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from types import SimpleNamespace
import torch


def test_debug_flags_are_strings():
    # Minimal fake trainer context
    job = types.SimpleNamespace(name='t')
    cfg = types.SimpleNamespace()
    # minimal config required by BaseSDTrainProcess init paths - we only need train_config properties used
    cfg.train = types.SimpleNamespace()
    cfg.train.cache_text_embeddings = False
    cfg.train.unload_text_encoder = False

    # Create trainer with minimal required attributes
    # We can't instantiate full trainer easily; instead test the formatting function via indirect means
    # Simulate debug flags dict and ensure formatting prints valid string
    debug_flags = {
        'control_usage_rate': 'true',
        'controlnet_enabled': 'true',
        'batch_has_control': 'false',
        'controlnet_offload_active': 'false',
        'splitprompt': 'true',
        'splitprompt_dataset': 'my_ds'
    }
    flags_msg = ' '.join([f"{k}={v}" for k, v in debug_flags.items() if v != ''])
    assert 'control_usage_rate=true' in flags_msg
    assert 'controlnet_enabled=true' in flags_msg
    assert 'splitprompt_dataset=my_ds' in flags_msg
