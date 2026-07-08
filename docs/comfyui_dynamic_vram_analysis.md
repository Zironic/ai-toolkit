# ComfyUI Dynamic VRAM / comfy-aimdo Analysis

Date inspected: 2026-07-07  
Repositories inspected:

- `Comfy-Org/ComfyUI`
- `Comfy-Org/comfy-aimdo`, commit observed in connector results: `ce3c59e39d00d363a317b6ef1ed565f28ada0d67`

This note is written to compare ComfyUI/comfy-aimdo’s dynamic VRAM behavior against another residency/offload implementation.

---

## 1. High-level conclusion

ComfyUI’s current DynamicVRAM path is not primarily an “estimate how many bytes of model weights should be resident” algorithm.

The older `lowvram_model_memory` path still exists in `comfy/model_management.py`, but when DynamicVRAM is active ComfyUI switches `CoreModelPatcher` to `ModelPatcherDynamic`. That dynamic patcher largely ignores the old `extra_memory` residency target and instead stages model weights into `comfy-aimdo` VBAR objects. Actual residency then becomes demand-paged at layer execution time.

The practical model is:

```text
model load:
    create/stage VBAR-backed virtual ranges for model weights

layer execution:
    call vbar_fault(weight range)
    if pages are resident:
        reuse resident GPU mapping
    else:
        free lower-priority VBAR pages if needed
        map missing 32 MiB page(s)
        copy/cast weight data into the mapped page(s)
    pin pages while the op uses them
    unpin after the op

pressure:
    computed from internal accounting + sampled CUDA/WDDM/meminfo state
    eviction is page/watermark based, not LRU
```

The most important comparison points are:

```text
page size:       32 MiB VBAR pages
eviction policy: priority + watermark
pin safety:      pinned pages are not freed by ordinary pressure eviction
pressure model:  max(simple capacity model, sampled external-pressure delta model)
poll interval:   2000 ms for external CUDA/WDDM/meminfo pressure
Windows signal:  WDDM local segment Budget + cuMemGetInfo fallback
```

---

## 2. DynamicVRAM activation in ComfyUI

ComfyUI enables dynamic VRAM when either:

```text
--enable-dynamic-vram
```

is passed, or when dynamic VRAM is not disabled by one of:

```text
--disable-dynamic-vram
--highvram
--gpu-only
--novram
--cpu
```

At startup, when `comfy-aimdo` initializes successfully, ComfyUI does:

```python
comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
comfy.memory_management.aimdo_enabled = True
```

That means the rest of ComfyUI can still call the usual model-loading APIs, but the model patcher implementation changes.

---

## 3. Old ComfyUI reserve / working-set preflight

`comfy/model_management.py` still computes a minimum inference reserve:

```python
minimum_inference_memory = 0.8 GiB + extra_reserved_memory()
```

`extra_reserved_memory()` is normally:

```text
non-Windows: 400 MiB
Windows:     600 MiB
Windows + >15 GiB total VRAM: 700 MiB
```

unless `--reserve-vram` overrides it.

`load_models_gpu()` then computes:

```python
inference_memory = minimum_inference_memory()
extra_mem = max(inference_memory, memory_required + extra_reserved_memory())
```

So in the legacy path, ComfyUI tries to leave:

```text
max(0.8 GiB + reserve, caller_estimated_working_memory + reserve)
```

available.

For DynamicVRAM this is still relevant as a **preflight/unload threshold**, but it is not the final weight-residency allocator. `ModelPatcherDynamic.partially_load()` calls dynamic `load()` and does not consume `extra_memory` the way the legacy patcher does.

---

## 4. Legacy low-VRAM residency formula

The legacy path computes a partial model weight budget approximately as:

```python
loaded_memory = loaded_model.model_loaded_memory()
current_free_mem = get_free_memory(torch_dev) + loaded_memory

lowvram_model_memory = max(
    0,
    current_free_mem - minimum_memory_required,
    min(
        current_free_mem * MIN_WEIGHT_MEMORY_RATIO,
        current_free_mem - minimum_inference_memory()
    )
)

lowvram_model_memory -= loaded_memory
```

`MIN_WEIGHT_MEMORY_RATIO` is normally `0.4`, but is `0.0` on Nvidia.

If the result is exactly zero, ComfyUI uses `0.1` as a sentinel to avoid full-load semantics. In `NO_VRAM`, it also forces `0.1`.

This is mostly useful for contrast. DynamicVRAM does not use this value as the main page-residency target.

---

## 5. ModelPatcherDynamic preparation

`ModelPatcherDynamic` creates per-device state:

```text
dynamic_vbars
dynamic_pins
```

Each load device gets a pin-state entry with separate subsets for:

```text
weights
patches
```

When `ModelPatcherDynamic.load()` runs:

1. It gets or creates a `ModelVBAR`.
2. The `ModelVBAR` virtual size is `model_size * 10`.
3. The comment says this huge virtual range is meant to cover model-defined casts, even extreme cases like FP4 to FP32.
4. It walks the model modules and prepares dynamic handling.

For modules with `comfy_cast_weights`:

```text
module_mem <= 16 KiB:
    force-load normally

module_mem > 16 KiB:
    set up VBAR-backed dynamic weight loading

LoRA/patch changes weight shape:
    force-load normally

other non-weight/bias params:
    force-load normally

buffers:
    force-load normally
```

For VBAR-backed modules, the module gets an `_v` allocation:

```python
m._v = vbar.alloc(v_weight_size)
```

The allocation is a virtual address range, not necessarily resident physical VRAM.

The patcher logs a staged-size value, but this is not the same as resident VRAM. It is the VBAR range staged for dynamic loading.

---

## 6. VBAR structure

In `comfy-aimdo/src/model-vbar.c`, VBAR page size is:

```c
#define VBAR_PAGE_SIZE (32 << 20)
```

So VBAR residency granularity is **32 MiB**.

A VBAR tracks:

```c
typedef struct ResidentPage {
    CUmemGenericAllocationHandle handle;
    uint32_t pin_count;
    size_t serial;
} ResidentPage;

typedef struct ModelVBAR {
    CUdeviceptr vbar;
    size_t nr_pages;
    size_t watermark;
    size_t watermark_limit;
    int device;
    void *higher;
    void *lower;
    size_t resident_count;
    ResidentPage residency_map[1];
} ModelVBAR;
```

The important fields are:

```text
handle:
    nonzero means page is physically resident/mapped

pin_count:
    number of active users; ordinary pressure eviction respects it

serial:
    incremented when a page is newly allocated; used as a residency signature

watermark:
    pages above watermark are considered outside the currently allowed resident range

watermark_limit:
    lower bound for explicit freeing; prevents reducing below a configured floor

higher/lower:
    linked-list priority ordering across VBARs
```

---

## 7. Priority ordering

`vbar_prioritize()` removes a VBAR from the list and inserts it near highest priority. It also resets:

```c
mv->watermark = mv->nr_pages;
```

`vbar_deprioritize()` inserts the VBAR at the low-priority end.

Freeing walks from lower priority toward higher priority. This means pressure tends to shrink lower-priority VBARs first.

This is not LRU at the page level. It is VBAR-level priority plus page watermark.

---

## 8. VBAR fault path

At execution time, ComfyUI calls:

```python
signature = comfy_aimdo.model_vbar.vbar_fault(s._v)
resident = comfy_aimdo.model_vbar.vbar_signature_compare(signature, s._v_signature)
```

If the signature matches, ComfyUI treats the weight as already resident and reuses `_v_weight` / `_v_bias`.

If it does not match, the module is materialized/copied/cast into the VBAR range or a temporary buffer.

The C-side `vbar_fault()` does this:

```text
1. run vbars_free(budget_deficit(0))
2. compute page_end for the requested range
3. if page_end > watermark:
       return VBAR_FAULT_OOM
4. for each page in requested range:
       if already resident:
           append serial to signature
           continue

       on first missing page:
           run vbars_free_for_vbar(...)
           if watermark fell below requested range:
               return VBAR_FAULT_OOM

       try to allocate/map physical VRAM for page
       if budget_deficit(32 MiB) > 0 or allocation fails:
           vbars_free(32 MiB)
           retry allocation
       if retry fails:
           return error

       increment serial
       increment resident_count
5. pin all pages in requested range
6. return signature
```

The key line before a fault allocation is:

```c
vbars_free(budget_deficit(0));
```

The source comment says this is a stopgap for shared-memory spikes: the allocator may not be called reliably when spill pressure occurs, so the next layer’s VBAR fault collects the pressure.

---

## 9. VBAR unpin path

After the op, ComfyUI calls `vbar_unpin()`.

The C-side behavior is:

```text
for each page in range:
    if pin_count:
        pin_count--
    if page_nr >= watermark:
        free page, unless still pinned
```

So `unpin()` itself is not a global pressure scan. It mainly releases active protection. But if a page was pushed above the watermark while pinned, it will be freed immediately once unpinned.

---

## 10. Explicit VBAR free path

ComfyUI’s `ModelPatcherDynamic.partially_unload()` calls:

```python
vbar.free_memory(memory_to_free)
```

That maps to `vbar_free_memory()`.

`vbar_free_memory(size)`:

```text
pages_to_free = ceil(size / 32 MiB)

synchronize CUDA context

while pages_to_free and watermark > watermark_limit:
    watermark--
    try to free page at watermark

synchronize CUDA context

return pages_freed * 32 MiB
```

This respects pins in ordinary mode. If the page is pinned, it is not freed.

---

## 11. Global VBAR free path

`vbars_free(size)` is the global freeing primitive.

It computes:

```text
pages_needed = ceil(size / 32 MiB)
```

Then it walks VBARs from lower priority toward higher priority and lowers each VBAR’s watermark while `watermark > watermark_limit`.

It frees page `watermark - 1` each time. If the page has no resident handle, the watermark still moves, but the number of pages actually freed does not increase.

Important detail: `vbars_free(size)` returns `pages_needed`, not bytes freed. So callers use it mainly as a pressure-reduction attempt, not a precise accounting result.

---

## 12. Pressure signal: exact budget_deficit() formula

The core pressure formula is in `comfy-aimdo/src/plat.h`:

```c
poll_budget_deficit(&prevailing_deficit_method);

deficit_simple =
    total_vram_usage
  + request_size
  + simple_vram_headroom
  - vram_capacity;

deficit_delta =
    deficit_sync
  + total_vram_usage
  - total_vram_last_check
  + request_size;

deficit =
    max(deficit_simple, deficit_delta)
  + extra_vram_headroom;
```

Flattened:

```python
def budget_deficit(request_size):
    poll_budget_deficit_if_stale()

    simple = (
        total_vram_usage
        + request_size
        + simple_vram_headroom
        - physical_vram_capacity
    )

    delta = (
        deficit_sync
        + (total_vram_usage - total_vram_last_check)
        + request_size
    )

    return max(simple, delta) + extra_vram_headroom
```

There are two models:

```text
simple:
    internal accounted usage + requested allocation + reserve - physical capacity

delta:
    last sampled external deficit + internal usage growth since that sample + requested allocation
```

Then `extra_vram_headroom` is added on top.

This delta branch is the distinctive part. It avoids polling CUDA/DXGI every allocation, but still accounts for internal growth after the last external poll.

---

## 13. Meaning of pressure variables

### total_vram_usage

Aimdo’s internal accounting of GPU allocations it knows about.

It includes:

```text
VBAR pages allocated by comfy-aimdo
intercepted cuMemAlloc / cuMemAllocAsync allocations
Aimdo VramBuffer allocations
```

VBAR pages update `total_vram_usage` in `three_stooges()`:

```text
cuMemCreate
cuMemMap
cuMemSetAccess
total_vram_usage += size
```

VBAR frees decrement it when pages are unmapped/released.

### simple_vram_headroom

Default in comfy-aimdo:

```text
256 MiB
```

ComfyUI can override it by passing `--reserve-vram` into:

```python
comfy_aimdo.control.init(simple_vram_headroom=...)
```

So `simple_vram_headroom` is the C-side simple reserve.

### extra_vram_headroom

This is per-device. ComfyUI passes `--vram-headroom` into:

```python
comfy_aimdo.control.init_devices((device_id, extra_vram_headroom), ...)
```

This value is added after the `max(simple, delta)` result.

### total_vram_last_check

Snapshot of `total_vram_usage` at the last external pressure poll.

### deficit_sync

The sampled external pressure signal. Its meaning depends on platform:

```text
normal CUDA:
    256 MiB - cuMemGetInfo_free

Windows:
    max(
        total_vram_usage + 512 MiB - WDDM_budget,
        96 MiB - cuMemGetInfo_free
    )

Linux integrated CUDA:
    integrated_ram_headroom - MemAvailable
```

---

## 14. External pressure polling cadence

External pressure is only polled every **2000 ms**.

The timing source is:

```text
Windows: GetTickCount64()
Linux:   gettimeofday()
```

If the last check was less than 2000 ms ago, the poll returns early and keeps the previous `deficit_sync`.

That means the pressure model is intentionally stale for up to about 2 seconds, but internal allocations after the poll are still included via:

```text
total_vram_usage - total_vram_last_check
```

---

## 15. Non-Windows CUDA pressure

For ordinary non-Windows CUDA, `cuda_budget_deficit()` does:

```c
if now - control_timestamp_last_check < 2000:
    return true

control_timestamp_last_check = now
total_vram_last_check = total_vram_usage

cuMemGetInfo(&free_vram, &total_vram)

deficit_sync = 256 MiB - free_vram
prevailing_deficit_method = "cuMemGetInfo"
```

So:

```text
free_vram > 256 MiB:
    deficit_sync negative

free_vram < 256 MiB:
    deficit_sync positive
```

This is not using ComfyUI’s 0.8 GiB inference reserve. It is comfy-aimdo’s internal 256 MiB CUDA-side signal, later combined with `simple_vram_headroom` and `extra_vram_headroom`.

---

## 16. Linux integrated CUDA pressure

On non-Windows, non-ROCm integrated CUDA devices, Aimdo uses `/proc/meminfo`.

It computes:

```text
integrated_ram_headroom = total_memory / 16
clamped to [2 GiB, 8 GiB]
```

Then:

```text
deficit_sync = integrated_ram_headroom - MemAvailable
```

So if available system memory drops below the integrated RAM headroom target, Aimdo reports pressure.

If `/proc/meminfo` cannot be read, it sets a large negative “simple-only” deficit, effectively avoiding the integrated delta signal.

---

## 17. Windows WDDM pressure detection

Windows uses `src-win/shmem-detect.c`.

Initialization:

1. Get CUDA device LUID.
2. Enumerate DXGI adapters.
3. Match adapter by LUID.
4. Query `IDXGIAdapter3`.
5. Store it as `g_wddm_adapter`.

If this fails, Aimdo logs that it is blind to the CUDA Sysmem Fallback Policy.

At poll time, it calls:

```c
IDXGIAdapter3::QueryVideoMemoryInfo(
    node = 0,
    DXGI_MEMORY_SEGMENT_GROUP_LOCAL,
    &info
)
```

If successful, it uses:

```text
effective_budget = info.Budget
```

If the WDDM query fails, it falls back to physical `vram_capacity`.

The WDDM deficit branch is:

```text
deficit_sync = total_vram_usage + 512 MiB - effective_budget
```

The constants are:

```text
WDDM_BUDGET_HEADROOM = 512 MiB
CUDA_BUDGET_HEADROOM = 192 MiB
```

It also polls CUDA free memory and computes:

```text
deficit_cuda = 96 MiB - free_vram
```

because the code uses `CUDA_BUDGET_HEADROOM / 2`.

Then it uses whichever is larger:

```text
deficit_sync = max(
    total_vram_usage + 512 MiB - WDDM_budget,
    96 MiB - cuMemGetInfo_free
)
```

So the Windows signal is not based on `DXGI CurrentUsage`. It logs `CurrentUsage`, `CurrentReservation`, and `AvailableForReservation`, but the formula uses `Budget`.

This is the core Windows/shared-memory-spike detection model.

---

## 18. How pressure triggers eviction

The primary trigger points are:

```text
VBAR fault:
    vbars_free(budget_deficit(0))
    then possibly vbars_free_for_vbar(...) before missing page allocation

VBAR page allocation failure:
    vbars_free(32 MiB)
    retry

cuMemAlloc hook:
    vbars_free(budget_deficit(size + 128 MiB))
    try allocation
    if fail: vbars_free(size + 128 MiB), retry

cuMemAllocAsync hook:
    vbars_free(budget_deficit(size))
    try allocation
    if fail: vbars_free(size), retry

VramBuffer growth:
    vbars_free(budget_deficit(grow_amount))
    try allocation chunks
    if fail: vbars_free(chunk_size), retry
```

This means the system reacts both before allocations and after failures.

The VBAR fault call with `budget_deficit(0)` is important because it allows pressure to be detected even when the immediate layer does not call PyTorch’s allocator.

---

## 19. Allocation hooks and internal accounting

Aimdo hooks the CUDA allocation API:

```text
cuMemAlloc_v2
cuMemFree_v2
cuMemAllocAsync
cuMemAllocAsync_ptsz
cuMemFreeAsync
cuMemFreeAsync_ptsz
```

On Windows, hooks are installed with Detours. On POSIX, hooks are installed with funchook.

For intercepted allocations, Aimdo tracks size by pointer in a hash table and updates `total_vram_usage`.

The allocation-size estimate is:

```python
def accounted_alloc_size(size):
    if size <= 1 * KiB:
        return 1 * KiB
    if size > 1 * MiB:  # CUDA_PAGE_SIZE / 2; CUDA_PAGE_SIZE is 2 MiB
        return align_up(size, 2 * MiB)
    return size
```

This is a best-effort approximation of CUDA async allocator behavior. It rounds large allocations up to CUDA page size and clamps tiny allocations to 1 KiB.

---

## 20. VBAR allocation helper

The physical page allocation helper is `three_stooges()`.

It does:

```text
cuMemCreate(...)
cuMemMap(...)
cuMemSetAccess(...)
total_vram_usage += size
```

On failure after `cuMemCreate`, it unmaps/releases as needed.

This means VBAR pages are real CUDA VMM physical allocations, not just PyTorch tensors.

---

## 21. Pinned-page behavior

Ordinary freeing calls:

```c
mod1(mv, page_nr, do_free=true, do_unpin=false)
```

Inside `mod1()`:

```c
do_free = do_free && rp->handle && (do_unpin || rp->pin_count == 0)
```

So normal pressure eviction cannot free a page with `pin_count > 0`.

Pinned pages become eligible only after `vbar_unpin()` decrements `pin_count`.

If a page is already above the watermark when it is unpinned, `vbar_unpin()` immediately frees it.

This prevents active-layer weight pages from being unmapped mid-op.

---

## 22. Temporary tensor fallback

The Python wrapper around `vbar_fault()` returns `None` for VBAR OOM:

```python
if res == 0:
    return signature
elif res == 1:
    return None
else:
    raise RuntimeError(...)
```

ComfyUI treats `signature is None` as nonresident and uses a temporary GPU tensor path rather than persistent VBAR residency.

So VBAR OOM does not necessarily mean the whole operation fails immediately. It can mean “do not make this page resident; use transient allocation/cast path.”

Actual failure still can happen if the fallback allocation cannot fit.

---

## 23. Comparison notes

### Strengths of ComfyUI/comfy-aimdo design

```text
1. Avoids precise per-model working-set estimates.
2. Uses real driver/WDDM pressure rather than only model estimates.
3. Separates fast internal accounting from slower external polling.
4. Protects active pages with pin counts.
5. Handles external pressure reactively at layer boundaries.
6. Uses WDDM Budget on Windows, which is usually more relevant than physical VRAM.
7. Has allocation-failure retry paths that free VBAR pages before retrying.
```

### Weaknesses / tradeoffs

```text
1. External pressure is sampled only every 2000 ms.
2. External pressure inside the poll window is not immediately visible unless an allocation fails.
3. Page granularity is coarse at 32 MiB.
4. Eviction is priority/watermark based, not per-page LRU.
5. Windows WDDM logic uses Budget, not CurrentUsage or AvailableForReservation in the formula.
6. The WDDM headroom constant is admitted in comments to be imperfect for graphics spikes.
7. If WDDM init fails, Aimdo says it is blind to CUDA Sysmem Fallback Policy.
8. total_vram_usage depends on successfully intercepting relevant CUDA allocation APIs.
```

### Most distinctive design choice

The most interesting part to compare against another implementation is:

```text
deficit = max(
    internal_usage + requested + simple_headroom - physical_capacity,
    sampled_external_deficit + internal_usage_since_sample + requested
) + extra_headroom
```

That is a hybrid of:

```text
continuous internal accounting
+
periodic external pressure sampling
```

The design avoids querying `cuMemGetInfo()` / DXGI every time, while still adapting to allocations and frees that Aimdo itself sees.

---

## 24. Minimal pseudocode of the full system

```python
VBAR_PAGE = 32 * MiB

def poll_budget_deficit_if_stale(now):
    global deficit_sync, total_vram_last_check

    if now - last_poll < 2000_ms:
        return

    total_vram_last_check = total_vram_usage

    if windows:
        effective_budget = query_wddm_local_budget_or_physical_capacity()

        wddm_deficit = total_vram_usage + 512*MiB - effective_budget

        cuda_deficit = -infinity
        if cuMemGetInfo_ok:
            cuda_deficit = 96*MiB - cuda_free_vram

        deficit_sync = max(wddm_deficit, cuda_deficit)

    elif linux_integrated_cuda:
        deficit_sync = integrated_ram_headroom - mem_available

    else:
        deficit_sync = 256*MiB - cuda_free_vram


def budget_deficit(request_size):
    poll_budget_deficit_if_stale(now())

    simple = (
        total_vram_usage
        + request_size
        + simple_vram_headroom
        - physical_vram_capacity
    )

    delta = (
        deficit_sync
        + (total_vram_usage - total_vram_last_check)
        + request_size
    )

    return max(simple, delta) + extra_vram_headroom


def vbars_free(bytes_to_free):
    pages_needed = ceil(bytes_to_free / VBAR_PAGE)

    for vbar in vbars_from_low_to_high_priority:
        while pages_needed > 0 and vbar.watermark > vbar.watermark_limit:
            vbar.watermark -= 1
            if vbar.page[vbar.watermark].resident and not vbar.page[vbar.watermark].pinned:
                unmap_and_release(vbar.page[vbar.watermark])
                total_vram_usage -= VBAR_PAGE
                pages_needed -= 1


def vbar_fault(vbar, offset, size):
    # catch already-existing shared-memory / budget pressure
    vbars_free(budget_deficit(0))

    page_start = floor(offset / VBAR_PAGE)
    page_end = ceil((offset + size) / VBAR_PAGE)

    if page_end > vbar.watermark:
        return OOM

    for page in range(page_start, page_end):
        if vbar.page[page].resident:
            continue

        # first missing page: free enough lower-priority pages
        free_for_this_vbar(vbar, target=page_end)

        if page_end > vbar.watermark:
            return OOM

        if budget_deficit(VBAR_PAGE) > 0 or physical_alloc_fails():
            vbars_free(VBAR_PAGE)
            if page_end > vbar.watermark:
                return OOM
            if physical_alloc_fails_again():
                return ERROR

        map_page(page)
        page.serial += 1
        vbar.resident_count += 1
        total_vram_usage += VBAR_PAGE

    for page in range(page_start, page_end):
        vbar.page[page].pin_count += 1

    return signature_for_pages(page_start, page_end)


def vbar_unpin(vbar, offset, size):
    for page in pages(offset, size):
        if page.pin_count:
            page.pin_count -= 1

        if page.index >= vbar.watermark and page.pin_count == 0:
            unmap_and_release(page)
```

---

## 25. Source map

ComfyUI files:

- `comfy/model_management.py`
  - `minimum_inference_memory()`
  - `load_models_gpu()`
  - `free_memory()`
  - `LoadedModel.model_unload()`
- `comfy/model_patcher.py`
  - `ModelPatcherDynamic`
  - `partially_load()`
  - `partially_unload()`
  - VBAR staging and force-load handling
- `comfy/ops.py`
  - `cast_modules_with_vbar()`
  - `resolve_cast_module_with_vbar()`
  - `cast_bias_weight()`
  - `uncast_bias_weight()`
- `main.py`
  - DynamicVRAM initialization and `CoreModelPatcher` replacement
- `comfy/cli_args.py`
  - `--reserve-vram`
  - `--vram-headroom`
  - dynamic VRAM enable/disable flags

comfy-aimdo files:

- `src/model-vbar.c`
  - VBAR page size
  - page residency map
  - `vbar_fault()`
  - `vbar_unpin()`
  - `vbars_free()`
  - `vbars_free_for_vbar()`
  - `vbar_free_memory()`
  - VBAR priority and watermark behavior
- `src/plat.h`
  - `budget_deficit()`
  - `three_stooges()`
  - CUDA VMM allocation helper
- `src/control.c`
  - `simple_vram_headroom`
  - `cuda_budget_deficit()`
  - external pressure polling
  - device initialization
- `src/control.h`
  - `AimdoContext`
  - per-device state fields
- `src-win/shmem-detect.c`
  - WDDM adapter matching
  - `QueryVideoMemoryInfo`
  - Windows budget formula
- `src/pyt-cu-plug-alloc-async.c`
  - allocation hooks
  - internal allocation accounting
  - allocation retry/free logic
- `src/cuda-hooks-shared.h`
  - hooked CUDA allocation/free functions
- `src-win/cuda-detour.c`
  - Windows hook installation
- `src-posix/cuda-funchooks.c`
  - POSIX hook installation
- `comfy_aimdo/control.py`
  - Python init wrappers
  - simple and extra headroom passing
- `comfy_aimdo/model_vbar.py`
  - Python VBAR wrapper
  - `fault`, `unpin`, `free_memory`, `loaded_size`, residency query

---

## 26. Practical checklist for comparing against your implementation

Compare these directly:

```text
1. Do you use physical capacity, driver free memory, WDDM budget, or all three?
2. Do you poll external pressure every allocation or use a stale sample + internal delta?
3. Do you track your own allocations, and if so, which allocator APIs are intercepted?
4. Do you use physical VRAM capacity or WDDM Budget as the cap on Windows?
5. Do you reserve headroom before or after max(simple, external)?
6. Do you evict by LRU, priority, layer order, or watermark?
7. What is your eviction granularity?
8. Are active pages protected by pin counts?
9. Do you react to allocation failure by freeing and retrying?
10. How do you handle external graphics/shared-memory spikes between polls?
11. Do you have a fallback if WDDM / OS budget queries fail?
12. Do you distinguish cudaMalloc and cudaMallocAsync fragmentation risk?
13. Can temporary fallback allocations bypass residency when a page cannot be made persistent?
```

The most important behavioral difference is likely whether your system is **predictive** or **reactive**.

ComfyUI/comfy-aimdo is mostly reactive:

```text
internal allocations:
    tracked immediately

external pressure:
    sampled every 2 seconds

unexpected pressure spike:
    corrected at next VBAR fault or allocation failure

residency target:
    implicit through watermarks, not explicit bytes-per-model
```
