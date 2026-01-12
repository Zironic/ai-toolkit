# LoKr + DOP Blur: Quick Fix Guide

## TL;DR

**Problem:** LoKr training with 128px DOP produces blurry samples. LoRA works fine.

**Root Cause:** LoKr's Kronecker product structure creates scale-dependent spatial patterns. When trained at 128px, these patterns don't generalize to 512px inference → blur.

**Quick Fix:** Increase DOP resolution to 256px or higher.

---

## Immediate Solutions (Pick One)

### Option 1: Increase Base DOP Resolution ⭐ RECOMMENDED
```yaml
# In your training config:
diff_output_preservation_resolution: 256  # Up from 128
```
**Why:** 256px contains 4x more spatial information than 128px, reducing the frequency mismatch between training and inference.

**Pros:**
- Simple one-line change
- Still fast (256px is only 4x pixels vs 128px, but 4x fewer than 512px)
- Should significantly reduce blur

**Cons:**
- Slightly slower than 128px (but still fast)

### Option 2: Use Consistent High Resolution
```yaml
# In your training config:
diff_output_preservation_resolution: 384  # Or 256, or 512
# Remove the resolution boost logic entirely
```
**Why:** Eliminates resolution switching, provides consistent spatial scale for LoKr to learn.

**Pros:**
- No resolution confusion
- LoKr learns one scale well

**Cons:**
- Slower if using 384px or 512px
- No speed benefit from low-res training

### Option 3: More Frequent High-Res Boosts
```yaml
# Modify SDTrainer.py line ~3400
# Change from:
if (current_step % 5) == 0:
# To:
if (current_step % 2) == 0:  # Every 2 steps instead of 5
```
**Why:** Exposes LoKr to high-resolution patterns more frequently.

**Pros:**
- Keeps low-res speed benefit for most steps
- More high-res exposure

**Cons:**
- Still has resolution switching
- Requires code modification

---

## Why LoKr is Affected But Not LoRA

### LoKr (Sensitive to Resolution)
```
Structure: W = W_orig + (W1 ⊗ W2) × scale
```
- **Kronecker product (⊗) creates rigid spatial hierarchy**
- Factorization dimensions (e.g., 32×32, 16×16) encode specific spatial scales
- When trained at 128px, learns 128px-scale patterns
- Cannot flexibly adapt to 512px → produces blur

### LoRA (Robust to Resolution)
```
Structure: W = W_orig + B × A
```
- **Simple matrix multiplication, no spatial structure**
- Can learn features at any spatial scale
- More redundant representation = more flexibility
- Works fine at 128px DOP

---

## Testing Plan

### Phase 1: Confirm the Issue
1. **Test 1:** Train LoRA to 250 steps with current code
   - Expected: Sharp samples (based on user report)
   
2. **Test 2:** Train LoKr to 250 steps with current code
   - If blurry: Confirms LoKr-specific issue
   - If sharp: Bug was the resolution boost code (unlikely)

### Phase 2: Try Quick Fix
3. **Test 3:** Train LoKr with 256px DOP to 250 steps
   - Expected: Significantly sharper than Test 2
   - If still blurry: Try 384px or 512px

---

## Expected Outcomes

| Configuration | Expected Result |
|---------------|-----------------|
| LoRA + 128px DOP | ✅ Sharp (confirmed by user) |
| LoKr + 128px DOP | ❌ Blurry (current issue) |
| LoKr + 256px DOP | ✅ Should be sharp or much better |
| LoKr + 384px DOP | ✅ Should be very sharp |
| LoKr + 512px DOP | ✅ Should be perfect (but slower) |

---

## ControlNet Considerations

If using ControlNet, the issue may be amplified:
- ControlNet provides 512px spatial guidance
- LoKr trained at 128px has conflicting spatial patterns
- Result: Stronger blur

**Mitigation:** Increasing DOP resolution to 256px+ should help even more with ControlNet.

---

## Technical Explanation (Simplified)

**Think of it like learning to draw:**

**LoRA = Sketch Artist**
- Learns general techniques that work at any canvas size
- Can sketch rough outlines at small scale, add details at large scale
- Flexible, adaptable

**LoKr = Pixel Artist**
- Learns specific pixel patterns for a specific canvas size
- If trained on 128×128 canvas, those pixel patterns look wrong on 512×512
- Efficient but rigid

**DOP at 128px:**
- LoRA: "I'll learn general drawing techniques" → works at 512px
- LoKr: "I'll learn these specific 128×128 pixel patterns" → blurs at 512px

**DOP at 256px or higher:**
- LoKr: "These 256×256 patterns are closer to 512×512" → much better results

---

## Implementation Code

### Quick Fix #1: Increase DOP Resolution (Config Change)
```yaml
# training_config.yaml
process:
  - type: sd_trainer
    training:
      diff_output_preservation: true
      diff_output_preservation_resolution: 256  # ← Change this from 128
      diff_output_preservation_every: 1
```

### Quick Fix #2: More Frequent 512px Boosts (Code Change)
```python
# In SDTrainer.py around line 3400
# Find this line:
if (current_step % 5) == 0:

# Change to:
if (current_step % 2) == 0:  # Or 3 for less frequent
```

### Quick Fix #3: Disable Resolution Boost (Code Change)
```python
# In SDTrainer.py around line 3395
# Comment out the entire resolution boost block:
# effective_preservation_resolution = preservation_resolution
# if preservation_resolution is not None and preservation_resolution < 512:
#     try:
#         current_step = getattr(self, 'step_num', 0)
#         if (current_step % 5) == 0:
#             effective_preservation_resolution = 512
#             print_acc(...)
#     except Exception:
#         pass

# Then set higher base resolution in config:
diff_output_preservation_resolution: 256  # Or 384
```

---

## Next Steps

1. **Confirm diagnosis:** Run Phase 1 tests (LoRA then LoKr)
2. **Apply Quick Fix #1:** Increase DOP resolution to 256px
3. **Test again:** Train LoKr to 250 steps with 256px DOP
4. **If still blurry:** Try 384px or 512px
5. **If fixed:** Document the optimal resolution for your use case

---

## References

See `LoKr-DOP-Analysis.md` for full technical analysis including:
- Detailed mathematical explanation of Kronecker products
- Frequency domain analysis
- ControlNet interaction theory
- Advanced mitigation strategies
- Long-term research directions
