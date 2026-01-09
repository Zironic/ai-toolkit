import torch
from toolkit.print import print_acc


def _infer_dtype(obj):
    try:
        if obj is None:
            return None
        if isinstance(obj, torch.Tensor):
            return obj.dtype
        if isinstance(obj, (list, tuple)) and len(obj) > 0:
            first = obj[0]
            if isinstance(first, torch.Tensor):
                return first.dtype
            if hasattr(first, 'dtype'):
                return getattr(first, 'dtype')
        if hasattr(obj, 'text_embeds'):
            te = obj.text_embeds
            if isinstance(te, torch.Tensor):
                return te.dtype
            if isinstance(te, (list, tuple)) and len(te) > 0 and isinstance(te[0], torch.Tensor):
                return te[0].dtype
    except Exception:
        return None
    return None


def _maybe_log_cast(obj, target_dtype, name: str):
    try:
        if target_dtype is None:
            return
        found = _infer_dtype(obj)
        if found is None:
            return
        if found != target_dtype:
            try:
                print_acc(f"[CONTROLNET] casting {name} from {found} -> {target_dtype}")
            except Exception:
                pass
    except Exception:
        return


def _cast_and_move(obj, device, dtype=None, non_blocking=False):
    if device is None or obj is None:
        return obj

    def _cast_tensor(t: torch.Tensor):
        if not isinstance(t, torch.Tensor):
            return t
        if dtype is not None:
            try:
                return t.to(device=device, dtype=dtype, non_blocking=non_blocking)
            except TypeError:
                return t.to(device=device, non_blocking=non_blocking).to(dtype=dtype)
        return t.to(device=device, non_blocking=non_blocking)

    try:
        if isinstance(obj, torch.Tensor):
            return _cast_tensor(obj)

        if hasattr(obj, 'to'):
            try:
                return obj.to(device=device, dtype=dtype)
            except TypeError:
                return obj.to(device=device)

        if isinstance(obj, (list, tuple)):
            moved = []
            for x in obj:
                moved.append(_cast_and_move(x, device, dtype=dtype, non_blocking=non_blocking))
            return tuple(moved) if isinstance(obj, tuple) else moved

        if hasattr(obj, 'text_embeds') or hasattr(obj, 'pooled_embeds'):
            try:
                if hasattr(obj, 'text_embeds'):
                    obj.text_embeds = _cast_and_move(obj.text_embeds, device, dtype=dtype, non_blocking=non_blocking)
                if hasattr(obj, 'pooled_embeds') and isinstance(obj.pooled_embeds, torch.Tensor):
                    obj.pooled_embeds = _cast_tensor(obj.pooled_embeds)
                return obj
            except Exception as e:
                raise RuntimeError(f"Failed to cast/move embed-like object to device {device} dtype {dtype}: {e}") from e

        return obj
    except Exception as e:
        raise RuntimeError(f"Failed to cast/move object to device {device} dtype {dtype}: {e}") from e
