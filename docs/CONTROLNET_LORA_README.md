ControlNet LoRA training (VideoX-Fun compatibility)

Quick test/run notes ✅

- Unit tests to run locally (fast):
  - pytest testing/test_controlnet_compat.py -q
  - pytest testing/test_base_process_controlnet_wrapper.py -q
  - pytest testing/test_zimage_control_routing_unit.py -q
  - pytest testing/test_lora_smoke_cpu.py -q

- Minimal CPU smoke run (single optimizer step):
  - `python -m pytest testing/test_lora_smoke_cpu.py -q`

Safety checks and notes ⚠️

- The trainer auto-detects VideoX/Z-Image-style ControlNets and will enable `zimage` routing. If you manually supply a ControlNet adapter ensure `adapter_config.controlnet_mode = 'zimage'` for deterministic routing.
- The data loader validates control image shapes and will pad a zero alpha channel if the model advertises 4 channels and dataset provides 3 channels. Conversely, the VideoX wrapper will drop an extra alpha channel (4 -> 3) when required.
- Fail-fast validation detects missing adapters or invalid control image tensors and raises clear RuntimeError messages with diagnostic artifacts written into the job folder for offline analysis.

If you need GPU runs or long experiments, request explicit human confirmation before starting.
