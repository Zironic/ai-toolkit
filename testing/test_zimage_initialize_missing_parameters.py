import torch
from extensions_built_in.diffusion_models.z_image_transformer2d import initialize_missing_parameters


def test_initialize_missing_parameters_copies_control_and_inits():
    # model_state_dict: two params with shapes
    model_state = {
        "control_some_layer.weight": torch.empty((4, 4)),
        "some_layer.weight": torch.empty((4, 4)),
        "norm.weight": torch.empty((4,)),
        "bias": torch.empty((4,)),
        "running_mean": torch.empty((4,)),
        "running_var": torch.empty((4,)),
        "num_batches_tracked": torch.empty((1,), dtype=torch.long),
    }

    # external state contains only 'some_layer.weight', which should be copied into control_ counterpart
    external = {
        "some_layer.weight": torch.randn(4, 4)
    }

    missing = [k for k in model_state.keys()]
    init = initialize_missing_parameters(missing, model_state, external_state_dict=external, torch_dtype=torch.float32)

    # control_some_layer.weight should be copied from external
    assert ("control_some_layer.weight" in init) and torch.allclose(init["control_some_layer.weight"], external["some_layer.weight"].to(init["control_some_layer.weight"].dtype))

    # norm weight should be ones
    assert torch.all(init["norm.weight"] == 1.0)

    # bias/running_mean/running_var initialized appropriately
    assert torch.all(init["bias"] == 0.0)
    assert torch.all(init["running_mean"] == 0.0)
    assert torch.all(init["running_var"] == 1.0)
    assert init["num_batches_tracked"].dtype == torch.long
