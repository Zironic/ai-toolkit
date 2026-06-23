from typing import List
import torch


class Automagic2(torch.optim.Optimizer):
    """
    Automagic v2.

    A single scalar learning rate is kept per parameter (e.g. one lr for the
    full weight matrix of a Linear layer rather than one per element). The lr
    is nudged up when the per-element update direction stays consistent with
    the previous step and nudged down when it flips, clamped to [min_lr, max_lr].

    The optimizer step is fused into the backward pass via
    ``register_post_accumulate_grad_hook``: each parameter is updated and its
    grad freed as soon as autograd finishes accumulating into it. ``.step()``
    therefore does no real work and peak VRAM stays low.

    Second-moment EMA state is stored in ``p.dtype`` (math runs in fp32 when
    the state is lower precision). Stochastic rounding is applied only when
    writing back to a bf16 parameter.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-6,
        min_lr: float = 1e-7,
        max_lr: float = 1e-3,
        lr_bump: float = 1e-6,
        beta2: float = 0.999,
        eps: float = 1e-30,
        clip_threshold: float = 1.0,
        weight_decay: float = 0.0,
        agreement_threshold: float = 0.5,
    ):
        if lr > 1e-3:
            print(f"Warning! Start lr {lr} is very high; forcing to 1e-6.")
            lr = 1e-6
        defaults = dict(
            lr=lr,
            min_lr=min_lr,
            max_lr=max_lr,
            lr_bump=lr_bump,
            beta2=beta2,
            eps=eps,
            clip_threshold=clip_threshold,
            weight_decay=weight_decay,
            agreement_threshold=agreement_threshold,
        )
        super().__init__(params, defaults)

        # Per-param scratch buffers, kept OUT of self.state so they are never
        # serialized into the checkpoint (they are pure working memory, lazily
        # rebuilt on demand). Storing them in self.state previously bloated the
        # saved optimizer state ~10x.
        self._scratch_buffers = {}

        self._hook_handles = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    handle = p.register_post_accumulate_grad_hook(
                        self._make_backward_hook(group)
                    )
                    self._hook_handles.append(handle)

        total = sum(p.numel() for g in self.param_groups for p in g["params"])
        print(f"Total training paramiters: {total:,}")

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _rms(t: torch.Tensor) -> torch.Tensor:
        return t.norm(2) / (t.numel() ** 0.5)

    def _scratch(self, p: torch.Tensor, key: str, dtype: torch.dtype) -> torch.Tensor:
        # Persistent per-param scratch buffer, reused every step and stored on the
        # optimizer itself (NOT in self.state, so it is never checkpointed). Reusing
        # one buffer per (param, key) keeps the caching allocator's block sizes
        # stable, which is what avoids the reserved-pool fragmentation that surfaced
        # as a spurious OOM mid-backward (a scalar .norm() failing at 21/28 GB).
        bufs = self._scratch_buffers.get(p)
        if bufs is None:
            bufs = {}
            self._scratch_buffers[p] = bufs
        buf = bufs.get(key)
        if (
            buf is None
            or buf.shape != p.shape
            or buf.dtype != dtype
            or buf.device != p.device
        ):
            buf = torch.empty(p.shape, dtype=dtype, device=p.device)
            bufs[key] = buf
        return buf

    def _init_state(self, p: torch.Tensor, group: dict) -> None:
        state = self.state[p]
        state["step"] = 0
        state["lr"] = torch.full(
            (), float(group["lr"]), dtype=torch.float32, device=p.device
        )
        state["last_polarity"] = torch.zeros(p.shape, dtype=torch.bool, device=p.device)
        if p.dim() >= 2:
            state["exp_avg_sq_row"] = torch.zeros(
                p.shape[:-1], dtype=p.dtype, device=p.device
            )
            state["exp_avg_sq_col"] = torch.zeros(
                p.shape[:-2] + p.shape[-1:], dtype=p.dtype, device=p.device
            )
        else:
            state["exp_avg_sq"] = torch.zeros(p.shape, dtype=p.dtype, device=p.device)

    def _make_backward_hook(self, group):
        def _hook(p: torch.Tensor):
            self._update_param(p, group)

        return _hook

    # -------------------------------------------------------------- per-param

    @torch.no_grad()
    def _update_param(self, p: torch.Tensor, group: dict) -> None:
        if p.grad is None:
            return
        state = self.state[p]
        if len(state) == 0:
            self._init_state(p, group)

        grad = p.grad
        if grad.is_sparse:
            raise RuntimeError("Automagic2 does not support sparse gradients.")

        beta2 = group["beta2"]
        eps = group["eps"]

        # Single fp32 scratch buffer, reused in turn for the squared grad, the
        # update, and finally the SR noise. bf16 -> fp32 is lossless, and the final
        # "* grad" below casts the bf16 grad up exactly, so we never need a separate
        # fp32 copy of the grad.
        ubuf = self._scratch(p, "update", torch.float32)
        ubuf.copy_(grad)            # fp32 copy of grad
        ubuf.mul_(ubuf).add_(eps)   # ubuf = grad^2 + eps

        if p.dim() >= 2:
            row_state = state["exp_avg_sq_row"]
            col_state = state["exp_avg_sq_col"]
            row = row_state if row_state.dtype == torch.float32 else row_state.to(torch.float32)
            col = col_state if col_state.dtype == torch.float32 else col_state.to(torch.float32)
            row.mul_(beta2).add_(ubuf.mean(dim=-1), alpha=1.0 - beta2)
            col.mul_(beta2).add_(ubuf.mean(dim=-2), alpha=1.0 - beta2)
            if row_state.dtype != torch.float32:
                row_state.copy_(row.to(row_state.dtype))
                col_state.copy_(col.to(col_state.dtype))
            # Approx second-moment update written straight into ubuf, then * grad.
            r = (row / row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
            c = col.unsqueeze(-2).rsqrt()
            torch.mul(r, c, out=ubuf).mul_(grad)
        else:
            v_state = state["exp_avg_sq"]
            v = v_state if v_state.dtype == torch.float32 else v_state.to(torch.float32)
            v.mul_(beta2).add_(ubuf, alpha=1.0 - beta2)
            if v_state.dtype != torch.float32:
                v_state.copy_(v.to(v_state.dtype))
            torch.rsqrt(v, out=ubuf).mul_(grad)

        ubuf.div_((self._rms(ubuf) / group["clip_threshold"]).clamp_(min=1.0))

        # Per-element sign agreement collapsed to a single bump decision. Bool
        # temporaries are 1 byte/elem so they are not the fragmentation driver;
        # the previous .to(float32) full-size buffer was, and is now gone. Kept
        # on-device as a 0-D tensor to avoid a CPU<->GPU sync in the hot path.
        cur_polarity = ubuf > 0
        last_polarity = state["last_polarity"]
        agreement = (cur_polarity == last_polarity).sum() / cur_polarity.numel()
        state["last_polarity"] = cur_polarity

        lr_t = state["lr"]
        if state["step"] > 0:
            direction = (agreement >= group["agreement_threshold"]).to(lr_t.dtype) * 2.0 - 1.0
            lr_t.add_(direction, alpha=group["lr_bump"]).clamp_(
                min=group["min_lr"], max=group["max_lr"]
            )
        state["step"] += 1

        ubuf.mul_(lr_t)
        wd = group["weight_decay"]

        if p.dtype == torch.bfloat16:
            # fp32 working copy of the param, reused across steps, shared by weight
            # decay and SR.
            pbuf = self._scratch(p, "param", torch.float32)
            pbuf.copy_(p)
            if wd != 0.0:
                ubuf.addcmul_(pbuf, lr_t, value=wd)
            pbuf.sub_(ubuf)
            # Stochastic rounding fp32 -> bf16: add random noise into the lower 16
            # mantissa bits, then truncate. ubuf is finished as an update, so we
            # reuse its storage as the int32 noise buffer (no new allocation).
            noise = ubuf.view(torch.int32)
            noise.random_(0, 1 << 16)
            as_int = pbuf.view(torch.int32)
            as_int.add_(noise).bitwise_and_(-65536)
            p.copy_(pbuf)
        else:
            if wd != 0.0:
                if p.dtype == torch.float32:
                    ubuf.addcmul_(p, lr_t, value=wd)
                else:
                    pbuf = self._scratch(p, "param", torch.float32)
                    pbuf.copy_(p)
                    ubuf.addcmul_(pbuf, lr_t, value=wd)
            p.add_(ubuf if p.dtype == torch.float32 else ubuf.to(p.dtype), alpha=-1.0)

        p.grad = None

    # ----------------------------------------------------------- optimizer API

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        return loss

    def get_learning_rates(self) -> List[float]:
        out = []
        for group in self.param_groups:
            lrs = [
                float(self.state[p]["lr"])
                for p in group["params"]
                if p in self.state and "lr" in self.state[p]
            ]
            out.append(sum(lrs) / len(lrs) if lrs else float(group["lr"]))
        return out

    def get_avg_learning_rate(self) -> float:
        lrs = self.get_learning_rates()
        return sum(lrs) / len(lrs) if lrs else float(self.defaults["lr"])

    def load_state_dict(self, state_dict):
        # Parent casts every fp state tensor to param.dtype; force lr back to fp32
        # so subsequent lr_bump (default 1e-6) isn't rounded away on bf16 weights.
        super().load_state_dict(state_dict)
        # Constructor args always win over whatever was saved in the checkpoint.
        for group in self.param_groups:
            for k, v in self.defaults.items():
                group[k] = v
            for p in group["params"]:
                st = self.state.get(p)
                if st is not None:
                    if isinstance(st.get("lr"), torch.Tensor):
                        st["lr"] = st["lr"].to(torch.float32)
                    # Drop scratch buffers that older checkpoints saved into state
                    # (they belong on self._scratch_buffers now and are rebuilt
                    # lazily); otherwise they'd re-bloat every subsequent save.
                    for legacy in ("_grad_fp32", "_scratch_fp32", "_param_fp32"):
                        st.pop(legacy, None)
