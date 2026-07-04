from pathlib import Path
path = Path(r'toolkit/memory_management/manager.py')
text = path.read_text(encoding='utf-8')
insert_after = '''    def _training_bounce_pool_budget_defaults(
        cls,
        module,
        offload_ids=None,
        *,
        block_stream_only=False,
        sources=None,
        history=None,
    ):
'''
# Insert helper after the existing _training_bounce_pool_budget_defaults function by marker before memory_managed_to.
marker = '''    def memory_managed_to(self, *args, **kwargs):
'''
helper = '''    @classmethod
    def _planned_bounce_reserve_bytes(
        cls,
        module,
        offload_ids,
        device,
        *,
        block_stream_only=False,
    ):
        if not (_OFFLOAD_PREFETCH_ENABLED and torch.device(device).type == "cuda"):
            return 0
        gib = 1024 ** 3
        env_budget = _env("AI_TOOLKIT_BOUNCE_POOL_GIB", None)
        if env_budget is not None:
            bounce_budget_gib = float(env_budget)
        else:
            bounce_budget_gib, _target_gib, _mode = cls._training_bounce_pool_budget_defaults(
                module,
                offload_ids,
                block_stream_only=block_stream_only,
            )
        bounce_budget_gib = min(
            float(_env("AI_TOOLKIT_BOUNCE_MAX_POOL_GIB", "6.0")),
            bounce_budget_gib,
        )
        return int(max(0.0, bounce_budget_gib) * gib)

'''
if helper not in text:
    if marker not in text:
        raise SystemExit('memory_managed_to marker not found')
    text = text.replace(marker, helper + marker, 1)
old_plain = '''        # Auto pin budget (default): pin the offloaded weights, capped by what
        # the WDDM shared pinned-memory proxy can actually give (see
        # _cap_auto_pin_budget). Pinning is not free, but psutil available RAM is
        # not the relevant budget for already-loaded model weights.
        # This mirrors attach_smart_training's auto-budget for the plain path.
        # An explicit pinned_weight_gib >= 0 overrides (set in __init__); only
        # auto (None / negative) is resized here.
        _auto_pin = pinned_weight_gib is None
        try:
            _auto_pin = _auto_pin or float(pinned_weight_gib) < 0
        except (TypeError, ValueError):
            _auto_pin = True
        if _auto_pin:
            managed_bytes = 0
            for _n, child in module.named_modules():
                if id(child) not in selected_offload_ids:
                    continue
                for _pn in ("weight", "bias"):
                    prm = getattr(child, _pn, None)
                    if isinstance(prm, torch.nn.Parameter):
                        managed_bytes += prm.numel() * prm.element_size()
            bounce_reserve_bytes = 0
            if _OFFLOAD_PREFETCH_ENABLED and torch.device(device).type == "cuda":
                env_budget = _env("AI_TOOLKIT_BOUNCE_POOL_GIB", None)
                if env_budget is not None:
                    bounce_budget_gib = float(env_budget)
                else:
                    bounce_budget_gib, _target_gib, _mode = cls._training_bounce_pool_budget_defaults(
                        module,
                        selected_offload_ids,
                        block_stream_only=False,
                    )
                bounce_budget_gib = min(
                    float(_env("AI_TOOLKIT_BOUNCE_MAX_POOL_GIB", "6.0")),
                    bounce_budget_gib,
                )
                bounce_reserve_bytes = int(bounce_budget_gib * (1024 ** 3))
            budget = cls._cap_auto_pin_budget(
                int(managed_bytes * 1.03),
                reserve_bytes=bounce_reserve_bytes,
            )
            module._memory_manager.pinned_weight_budget_bytes = budget
'''
new_plain = '''        # Pin budget: pin the offloaded weights, capped by the WDDM shared
        # pinned-memory proxy after reserving the planned bounce-pool window.
        # A positive config value is a requested budget, not permission to starve
        # bounce; set the bounce pool budget to 0 if the pool should get no share.
        _auto_pin = pinned_weight_gib is None
        try:
            _auto_pin = _auto_pin or float(pinned_weight_gib) < 0
        except (TypeError, ValueError):
            _auto_pin = True
        managed_bytes = 0
        for _n, child in module.named_modules():
            if id(child) not in selected_offload_ids:
                continue
            for _pn in ("weight", "bias"):
                prm = getattr(child, _pn, None)
                if isinstance(prm, torch.nn.Parameter):
                    managed_bytes += prm.numel() * prm.element_size()
        desired_pin_bytes = (
            int(managed_bytes * 1.03)
            if _auto_pin
            else int(max(0.0, float(pinned_weight_gib)) * (1024 ** 3))
        )
        bounce_reserve_bytes = cls._planned_bounce_reserve_bytes(
            module,
            selected_offload_ids,
            device,
            block_stream_only=False,
        )
        budget = cls._cap_auto_pin_budget(
            desired_pin_bytes,
            reserve_bytes=bounce_reserve_bytes,
        )
        module._memory_manager.pinned_weight_budget_bytes = budget
'''
if old_plain not in text:
    raise SystemExit('plain attach pin block not found')
text = text.replace(old_plain, new_plain, 1)
old_smart = '''        # Auto-size the pinned-weight budget to the offloaded (streamed) weights
        # unless the job set an explicit value. Those weights already occupy CPU
        # RAM; pinning them just stops the OS paging them out and re-faulting them
        # on every fetch (the slow bounce copies + system lag). Capped by free
        # host RAM so it degrades gracefully on RAM-tight machines.
        resolved_pin_gib = pinned_weight_gib
        auto_pin = pinned_weight_gib is None
        try:
            auto_pin = auto_pin or float(pinned_weight_gib) < 0
        except (TypeError, ValueError):
            auto_pin = True
        if auto_pin:
            gib = 1024 ** 3
            # Size the budget to the WHOLE model, not just the initially-offloaded
            # part. Any layer the live controller later demotes (resident -> CPU)
            # then always has budget to be pinned, so "every CPU-resident weight
            # is pinned" holds without special-casing demotion. Only weights that
            # actually land on CPU are pinned, so resident layers on GPU cost no
            # RAM here; the extra budget is headroom, not an allocation.
            # Cap by the WDDM shared pinned-memory proxy. psutil available RAM is
            # not authoritative for already-loaded weights; the failure mode is
            # exhausting shared pinned-memory commit, not ordinary pageable RAM.
            bounce_reserve_bytes = 0
            if _OFFLOAD_PREFETCH_ENABLED and torch.device(device).type == "cuda":
                env_budget = _env("AI_TOOLKIT_BOUNCE_POOL_GIB", None)
                if env_budget is not None:
                    bounce_budget_gib = float(env_budget)
                else:
                    bounce_budget_gib, _target_gib, _mode = cls._training_bounce_pool_budget_defaults(
                        module,
                        plan["offload_ids"],
                        block_stream_only=block_stream_only,
                    )
                bounce_budget_gib = min(
                    float(_env("AI_TOOLKIT_BOUNCE_MAX_POOL_GIB", "6.0")),
                    bounce_budget_gib,
                )
                bounce_reserve_bytes = int(bounce_budget_gib * gib)
            pin_bytes = cls._cap_auto_pin_budget(
                int(int(plan["model_bytes"]) * 1.03),
                reserve_bytes=bounce_reserve_bytes,
            )
            resolved_pin_gib = pin_bytes / gib
'''
new_smart = '''        # Pin budget: size to the whole model in auto mode, or honor a positive
        # config value as a requested budget, then cap it after reserving the
        # planned bounce-pool window. This keeps explicit pin budgets from
        # consuming the pool's share of the same WDDM pinned-memory ceiling.
        gib = 1024 ** 3
        auto_pin = pinned_weight_gib is None
        try:
            auto_pin = auto_pin or float(pinned_weight_gib) < 0
        except (TypeError, ValueError):
            auto_pin = True
        desired_pin_bytes = (
            int(int(plan["model_bytes"]) * 1.03)
            if auto_pin
            else int(max(0.0, float(pinned_weight_gib)) * gib)
        )
        bounce_reserve_bytes = cls._planned_bounce_reserve_bytes(
            module,
            plan["offload_ids"],
            device,
            block_stream_only=block_stream_only,
        )
        pin_bytes = cls._cap_auto_pin_budget(
            desired_pin_bytes,
            reserve_bytes=bounce_reserve_bytes,
        )
        resolved_pin_gib = pin_bytes / gib
'''
if old_smart not in text:
    raise SystemExit('smart attach pin block not found')
text = text.replace(old_smart, new_smart, 1)
path.write_text(text, encoding='utf-8')
