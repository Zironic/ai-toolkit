# LoKr + Low-Resolution DOP Analysis: Why Blur Occurs

## Critical Discovery

**The 512px DOP resolution boost never triggered in previous training runs.** Training logs confirm "divisible by 5" messages were never printed, meaning DOP always ran at 128px.

This changes everything: **The blur is caused by consistent 128px DOP training with LoKr, not by resolution switching.**

---

## Problem Statement

- **LoRA + 128px DOP**: Works perfectly (~80% as effective as full-res, no blur)
- **LoKr + 128px DOP**: Produces blurry samples at 250 steps
- **LoKr + ControlNet + 128px DOP**: Blur may be amplified

The question: **Why does 128px DOP cause blur specifically with LoKr but not LoRA?**

---

## LoKr vs LoRA Architecture

### LoRA (Low-Rank Adaptation)
```
W_new = W_original + B × A
```
- Simple matrix factorization
- Two matrices: `B` (out_dim × rank) and `A` (rank × in_dim)
- Additive update to original weights
- **Robust to scale variations** due to redundancy in representation

### LoKr (Low-Rank Kronecker Product)
```
W_new = W_original + (W1 ⊗ W2) × scale
where:
  W1 = lokr_w1  (or  lokr_w1_a @ lokr_w1_b for factorized version)
  W2 = lokr_w2  (or  lokr_w2_a @ lokr_w2_b for factorized version)
  ⊗ = Kronecker product
```

**Key Kronecker Properties:**
- **Hierarchical Structure**: `(A ⊗ B)(C ⊗ D) = (AC) ⊗ (BD)`
- **Spatial Decomposition**: Kronecker products preserve and enforce spatial structure
- **Parameter Efficiency**: Uses fewer parameters than LoRA for same expressiveness
- **Scale-Sensitive**: The Kronecker product structure depends heavily on the dimensional relationships

**Factorization Example (from code):**
```python
# For a Linear layer with out_dim=512, in_dim=512
in_m, in_n = factorization(512)   # e.g., (32, 16)
out_l, out_k = factorization(512) # e.g., (32, 16)

# Creates structure: ((32, 16), (32, 16))
# Final weight via: kron(W1[32×32], W2[16×16]) = W[512×512]
```

---

## Root Cause Analysis

### 1. **Frequency Content Mismatch** ⭐ PRIMARY CAUSE

**The Problem:**
- 128px images contain fundamentally different frequency characteristics than 512px images
- When you downsample an image from 512px to 128px, **you lose high-frequency details**
- The network learns patterns at 128px frequency scales
- At inference time (512px), the network needs high-frequency patterns it never learned

**Why LoKr is More Affected:**
- **Kronecker structure is inherently spatial**: `W1 ⊗ W2` creates a structured factorization that encodes spatial relationships
- The factorization dimensions (e.g., 32×32 and 16×16) correspond to specific spatial scales
- When DOP trains at 128px, the Kronecker factors learn to represent **low-resolution spatial patterns**
- These patterns don't generalize well to 512px because the Kronecker structure is scale-dependent

**Why LoRA is More Robust:**
- LoRA's simple factorization `B × A` has **no inherent spatial structure**
- It learns abstract feature transformations that are more **scale-invariant**
- The redundancy in LoRA's representation provides robustness across scales

**Mathematical Insight:**
```python
# Kronecker product preserves scale structure
A ⊗ B has dimensions (m×n) ⊗ (p×q) = (mp × nq)

# If A and B are learned for 128px spatial patterns,
# the resulting (mp × nq) structure encodes 128px-scale relationships

# At 512px, the spatial scale is 4x larger (512/128 = 4)
# But the Kronecker structure still encodes 128px patterns → blur
```

### 2. **Parameter Efficiency Trade-off**

**LoKr's Advantage Becomes a Liability:**
- LoKr uses **fewer parameters** than LoRA for equivalent expressiveness
- Each parameter carries **more responsibility** (represents more of the weight space)
- When training at 128px:
  - LoKr parameters overfit to low-res patterns
  - Limited parameters can't encode multi-scale behavior
  - At 512px inference, produces blurry results

**LoRA's Redundancy:**
- More parameters provide **multi-scale representation capacity**
- Even when trained at 128px, some LoRA parameters can capture higher-frequency patterns
- Redundancy = robustness to resolution mismatch

### 3. **Spatial Structure Rigidity**

**Kronecker Product Constraints:**
```python
# From lokr.py factorization function:
# dim=512 with factor=-1 → (32, 16)
# This creates a rigid spatial hierarchy

W1: 32×32  # Encodes one spatial scale
W2: 16×16  # Encodes another spatial scale
Result: kron(W1, W2) = 512×512

# If W1 and W2 learn 128px patterns,
# the 32×32 and 16×16 scales are optimized for 128px
# Cannot flexibly adapt to 512px patterns
```

**LoRA's Flexibility:**
```python
# LoRA has no fixed spatial structure
B: 512×rank
A: rank×512

# Can learn features at ANY spatial scale
# More flexible adaptation to different resolutions
```

---

## ControlNet Interaction Analysis

### Potential Amplification Mechanisms

**1. Spatial Conditioning Conflict**
- ControlNet provides **strong spatial guidance** at full resolution
- LoKr learns **spatial patterns via Kronecker structure** at 128px
- These two spatial representations may conflict:
  - ControlNet: "Here's the spatial structure at 512px"
  - LoKr (trained at 128px): "I learned spatial patterns at 128px"
  - Result: Network produces a "compromise" → blur

**2. Resolution Mismatch in Guidance**
- If ControlNet conditioning is at **full resolution** (512px)
- But DOP trains the network at **128px**
- LoKr's rigid Kronecker structure struggles to reconcile:
  - Detailed 512px control signal
  - Coarse 128px training patterns
  - Produces blurred output as a "best-fit" solution

**3. Frequency Domain Interference**
- ControlNet encodes high-frequency spatial edges/structures
- 128px DOP trains LoKr to produce low-frequency outputs
- LoKr's Kronecker factorization **rigidly enforces learned frequencies**
- Cannot adapt to ControlNet's high-frequency demands → blur

**Why LoRA is More Robust:**
- LoRA's simple structure doesn't enforce spatial hierarchies
- Can "follow" ControlNet's guidance more flexibly
- Less structural rigidity = better adaptation to multi-resolution scenarios

---

## Diagnostic Testing Plan

### Phase 1: Isolate LoKr vs Resolution-Switching
**Test 1A:** LoRA + 128px DOP (baseline - should be sharp based on user report)
**Test 1B:** LoKr + 128px DOP (test hypothesis - if blurry, confirms LoKr sensitivity)

### Phase 2: Resolution Experiments
**Test 2A:** LoKr + 256px DOP (higher base resolution)
**Test 2B:** LoKr + 512px DOP (full resolution)
**Test 2C:** LoKr + alternating 128px/512px (test resolution switching tolerance)

### Phase 3: ControlNet Interaction
**Test 3A:** LoKr + 128px DOP + ControlNet (current setup)
**Test 3B:** LoKr + 128px DOP WITHOUT ControlNet (isolate ControlNet effect)
**Test 3C:** LoRA + 128px DOP + ControlNet (LoRA baseline with ControlNet)

---

## Proposed Mitigations

### Short-Term Solutions (Quick Fixes)

#### 1. **Increase Base DOP Resolution** ⭐ RECOMMENDED FIRST
```yaml
# In training config:
diff_output_preservation_resolution: 256  # Up from 128
```
**Rationale:**
- 256px contains more high-frequency information
- 2x resolution = 4x pixel area = more spatial detail
- Still much faster than 512px (4x fewer pixels than 512px)
- Better match between training and inference frequencies

**Expected Outcome:**
- Sharper samples at 250 steps
- Minimal training speed impact (256px is still fast)
- Better generalization due to richer frequency content

#### 2. **More Frequent 512px Boosts**
```yaml
# Current: Every 5 steps
# Proposed: Every 3 or 2 steps
diff_output_preservation_resolution: 128
diff_output_preservation_boost_frequency: 3  # Or 2
```
**Rationale:**
- Exposes LoKr to high-resolution patterns more often
- Helps Kronecker factors learn multi-scale representations
- Trade-off: Slightly slower training

#### 3. **Disable Resolution Boost, Use Consistent Resolution**
```yaml
diff_output_preservation_resolution: 256  # Or 384
# Remove resolution boost logic entirely
```
**Rationale:**
- Eliminates resolution switching entirely
- Consistent training resolution = more stable learning
- LoKr learns one spatial scale well, no confusion

### Medium-Term Solutions (Architectural Changes)

#### 4. **Multi-Scale DOP Training** (NEW FEATURE)
Modify DOP to train at **multiple resolutions randomly**:
```python
# Pseudocode
resolutions = [128, 192, 256, 384, 512]
dop_resolution = random.choice(resolutions)
```
**Rationale:**
- Forces LoKr to learn multi-scale representations
- Kronecker factors must accommodate various spatial frequencies
- Better generalization to 512px inference

#### 5. **Frequency-Aware DOP Loss Weighting**
Weight DOP loss by frequency content:
```python
# Pseudocode
if dop_resolution < target_resolution:
    # Compute high-frequency component of loss
    freq_loss = high_pass_filter(pred - target)
    # Upweight high-frequency errors
    loss = base_loss + alpha * freq_loss
```
**Rationale:**
- Explicitly teaches network to preserve high frequencies
- Compensates for low-resolution training
- Directly addresses the frequency mismatch problem

#### 6. **Adaptive LoKr Factorization**
Modify LoKr to use **multi-scale Kronecker products**:
```python
# Pseudocode (advanced modification)
# Instead of: W1 ⊗ W2
# Use: (W1_coarse ⊗ W2_coarse) + (W1_fine ⊗ W2_fine)
```
**Rationale:**
- Separate Kronecker factors for coarse and fine scales
- Coarse factors learn from 128px DOP
- Fine factors learn from 512px DOP or regular training
- Requires LoKr architecture modification

### Long-Term Solutions (Research Directions)

#### 7. **LoKr with Scale-Invariant Factorization**
Research modification to make Kronecker products scale-invariant:
- Wavelet-based factorization
- Multi-resolution Kronecker decomposition
- Adaptive factor dimensions based on training resolution

#### 8. **Hybrid LoRA-LoKr Architecture**
Combine LoRA and LoKr:
- LoRA branch for high-frequency details
- LoKr branch for parameter-efficient coarse features
- Best of both worlds: efficiency + robustness

---

## Recommended Action Plan

### Immediate Next Steps (User Should Do First)

1. **Complete Diagnostic Testing** (from Phase 1 above)
   - Run LoRA to 250 steps (confirm it's still sharp)
   - Run LoKr to 250 steps (confirm if blur persists with fixed code)
   - This isolates whether issue is LoKr-specific or code-related

2. **If LoKr is Still Blurry:**
   - **Try Mitigation #1**: Increase base DOP resolution to 256px
   - Test again to 250 steps
   - Expected: Significant improvement in sharpness

3. **If 256px Helps but Not Enough:**
   - **Try Mitigation #3**: Use consistent 256px or 384px (no resolution boost)
   - Or **Try Mitigation #2**: More frequent 512px boosts (every 2-3 steps)

### Implementation Priority

**HIGH PRIORITY (Do These First):**
- ✅ Mitigation #1: Increase base DOP resolution to 256px
- ✅ Mitigation #3: Use consistent resolution (no boost)

**MEDIUM PRIORITY (If High Priority Insufficient):**
- ⚠️ Mitigation #2: More frequent 512px boosts
- ⚠️ Mitigation #4: Multi-scale DOP training (new feature)

**LOW PRIORITY (Research/Future Work):**
- 🔬 Mitigation #5: Frequency-aware loss weighting
- 🔬 Mitigation #6: Adaptive LoKr factorization
- 🔬 Mitigations #7-8: Long-term architectural research

---

## Technical Deep Dive: Why Kronecker Products are Scale-Sensitive

### Mathematical Explanation

The Kronecker product is defined as:
```
A ⊗ B = [a_ij * B] for all i,j
```

For matrices A (m×n) and B (p×q), the result is (mp × nq).

**Key Property: Non-Commutative Spatial Structure**
```
A ⊗ B ≠ B ⊗ A
```

This means the order matters, and the resulting structure encodes specific spatial relationships.

**Scale-Dependent Factorization:**
```python
# Example: 512×512 weight matrix
# Factorized as: kron(W1[32×32], W2[16×16])

# At 128px training:
W1 learns 32×32 = 1024-dimensional coarse patterns
W2 learns 16×16 = 256-dimensional fine patterns

# These dimensions are optimized for 128px spatial scale
# At 512px (4x larger), the 32×32 and 16×16 patterns
# are still at 128px scale → produces blur
```

**Frequency Domain View:**
```python
# Kronecker product in frequency domain:
# (A ⊗ B) in spatial domain corresponds to
# structured frequency decomposition

# If A and B encode 128px frequencies,
# kron(A, B) rigidly enforces those frequencies
# Cannot adapt to 512px high-frequency requirements
```

### Contrast with LoRA

LoRA's simple factorization:
```
W_new = W_orig + B @ A
```

Has **no inherent spatial structure** or frequency constraints:
- B and A are just matrices, no Kronecker structure
- Can learn features at **any spatial scale**
- More **redundant representation** = more flexibility
- Robustness to resolution changes

---

## Conclusion

**Root Cause:** LoKr's Kronecker product structure creates a **scale-dependent spatial hierarchy** that overfits to the 128px DOP training resolution. The rigid Kronecker factorization cannot flexibly adapt to 512px inference, producing blur.

**LoRA doesn't have this problem** because its simple matrix factorization lacks spatial structure constraints and provides redundancy for multi-scale robustness.

**Recommended Fix:** Increase DOP base resolution to **256px** as immediate solution. If insufficient, use consistent 256px or 384px training without resolution boost.

**ControlNet Amplification:** Likely amplifies the problem by providing 512px spatial guidance that conflicts with LoKr's 128px-trained Kronecker patterns.

---

## References

1. LyCORIS Paper: "Navigating Text-To-Image Customization: From LyCORIS Fine-Tuning to Model Evaluation" (arXiv:2309.14859)
2. LoKr Implementation: `toolkit/models/lokr.py` - Kronecker product factorization with scale parameter
3. DOP Implementation: `SDTrainer.py` lines 3380-3550 - Differential Output Preservation with downsampling
4. Factorization Function: `lokr.py` lines 17-58 - Dimensional decomposition for Kronecker products

---

## Appendix: Code References

### LoKr Kronecker Product Construction
```python
# From lokr.py line 224
weight = make_kron(
    self.lokr_w1 if self.use_w1 else self.lokr_w1_a@self.lokr_w1_b,
    (self.lokr_w2 if self.use_w2
     else make_weight_cp(self.lokr_t2, self.lokr_w2_a, self.lokr_w2_b) if self.cp
     else self.lokr_w2_a@self.lokr_w2_b),
    torch.tensor(self.multiplier * self.scale)
)
```

### DOP Downsampling (Fixed Version)
```python
# From SDTrainer.py lines 3480-3530
# Downsample clean latents
latents_small = torch.nn.functional.interpolate(
    batch.latents, size=(target_h, target_w), mode='bicubic', align_corners=False
)

# Reconstruct and downsample noise
t_frac = (timesteps.float() / 1000.0)
noise_reconstructed = (noisy_latents - (1.0 - t_frac) * batch.latents) / t_frac_safe
noise_small = torch.nn.functional.interpolate(
    noise_reconstructed, size=(target_h, target_w), mode='bicubic', align_corners=False
)

# Re-apply noise schedule at small resolution
noisy_small = (1.0 - t_frac) * latents_small + t_frac * noise_small
```

### Factorization Example
```python
# From lokr.py lines 17-58
def factorization(dimension: int, factor: int = -1) -> tuple[int, int]:
    # Returns (m, n) where m*n = dimension and m ≤ n
    # Examples:
    # 512 → (32, 16) with factor=-1
    # 1024 → (32, 32) with factor=-1
    
    # This creates the rigid spatial hierarchy
    # that makes LoKr scale-sensitive
```
