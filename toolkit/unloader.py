import gc

import torch
from toolkit.basic import flush
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from toolkit.models.base_model import BaseModel


class FakeTextEncoder(torch.nn.Module):
    def __init__(self, device, dtype):
        super().__init__()
        # register a dummy parameter to avoid errors in some cases
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))
        self._device = device
        self._dtype = dtype

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "This is a fake text encoder and should not be used for inference."
        )
        return None

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    def to(self, *args, **kwargs):
        return self


def unload_text_encoder(model: "BaseModel"):
    """Replace all text encoders with FakeTextEncoder stubs and free GPU memory.

    Does NOT call .to('cpu') — that would spike committed RAM by creating a CPU
    copy of a model we're discarding.  Instead we drop every reference and let
    Python's refcount/GC free the CUDA tensors in place.
    """
    if model.text_encoder is None:
        return

    if isinstance(model.text_encoder, list):
        text_encoder_list = []
        pipe = model.pipeline
        old_encoders = []

        if hasattr(pipe, "text_encoder"):
            old_encoders.append(pipe.text_encoder)
            te = FakeTextEncoder(device=model.device_torch, dtype=model.torch_dtype)
            text_encoder_list.append(te)
            pipe.text_encoder = te

        i = 2
        while hasattr(pipe, f"text_encoder_{i}"):
            old_encoders.append(getattr(pipe, f"text_encoder_{i}"))
            te = FakeTextEncoder(device=model.device_torch, dtype=model.torch_dtype)
            text_encoder_list.append(te)
            setattr(pipe, f"text_encoder_{i}", te)
            i += 1

        # Replace model's list reference before deleting old encoders so all
        # refcounts drop to zero at the same time.
        model.text_encoder = text_encoder_list
        for old_te in old_encoders:
            del old_te
        del old_encoders
    else:
        old_te = model.text_encoder
        model.text_encoder = FakeTextEncoder(device=model.device_torch, dtype=model.torch_dtype)
        del old_te

    gc.collect()
    torch.cuda.empty_cache()
