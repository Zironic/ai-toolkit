import toolkit.print as tprint
from jobs.process.BaseProcess import BaseProcess

class DummyJob:
    name = 'dummy'
    meta = {}


def test_model_and_control_coverage(monkeypatch, capsys):
    monkeypatch.setattr(tprint, 'print_acc', print)
    bp = BaseProcess(0, DummyJob(), {'name': 'p', 'performance_log_every': 0})

    timing_dict = {
        'train_loop': 10.0,
        # ControlNet pieces
        'get_adapter_images': 0.2,
        'encode_adapter': 0.1,
        'controlnet_forward': 0.35,
        # Model pieces
        'predict_unet': 0.8,
        'encode_images': 0.05,
        'to_device': 0.02,
        'cpu_transfer': 0.03,
        'calculate_loss': 0.1,
        'backward': 0.4,
        'optimizer_step': 0.05,
        'ema_update': 0.01,
        # Other unrelated
        'preservation_backward': 0.25,
        'dop_predict': 0.2,
    }

    bp._print_timer_composites(timing_dict)
    out = capsys.readouterr().out
    assert 'ControlNet:' in out
    assert 'Model:' in out

    # parse ControlNet and Model values
    ctrl_part = [p for p in out.split('|') if 'ControlNet' in p][0]
    ctrl_val = float(ctrl_part.split(':')[1].strip().split('s')[0])
    expected_ctrl = timing_dict['get_adapter_images'] + timing_dict['encode_adapter'] + timing_dict['controlnet_forward']
    assert abs(ctrl_val - expected_ctrl) < 1e-6

    model_part = [p for p in out.split('|') if 'Model' in p][0]
    model_val = float(model_part.split(':')[1].strip().split('s')[0])
    expected_model = timing_dict['predict_unet'] + timing_dict['encode_images'] + timing_dict['to_device'] + timing_dict['cpu_transfer'] + timing_dict['calculate_loss'] + timing_dict['backward'] + timing_dict['optimizer_step'] + timing_dict['ema_update']
    assert abs(model_val - expected_model) < 1e-6
