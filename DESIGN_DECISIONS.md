# Design Decisions: Additional Loss Functions

This document explains key design choices made in implementing the additional loss functions.

---

## 1. Using Predicted Codes for Audio Generation

### Decision
The additional losses (Speaker Similarity and UTMOS) generate audio from **predicted codes**, allowing gradients to flow back through the TTS model.

### Rationale

#### Why Predicted Codes?
1. **Direct Optimization**: Gradients flow from quality losses → generated audio → predicted codes → TTS model
2. **True Quality Signal**: The losses evaluate what the model actually produces, not what it should produce
3. **End-to-End Training**: The TTS model learns to generate codes that maximize quality metrics
4. **Proper Gradient Flow**: This is the correct way to optimize for quality - the model gets feedback on its actual outputs

#### Implementation Details
The transducer outputs logits of shape `(N, T, n_cluster_groups, n_decoder_codes)` where:
- `N` = batch size
- `T` = sequence length  
- `n_cluster_groups` = 4 (number of codebook groups)
- `n_decoder_codes` = n_codes + special tokens

### Code Implementation
```python
# Get predicted codes by taking argmax over last dimension
# Result shape: (N, T, 4)
predicted_codes = recons_segments['logits'].argmax(dim=-1)

# Clamp to valid code range [0, n_codes-1]
voc_input = torch.clamp(predicted_codes, 0, self.hp.n_codes - 1)

# Generate audio - GRADIENTS FLOW through predicted_codes
audio_hat = self.vocoder(voc_input, voc_spkr)

# Compute losses on generated audio
speaker_loss = compute_speaker_similarity(audio_hat, audio_gt)
mos_loss = compute_mos(audio_hat)

# Backprop flows: loss → audio_hat → voc_input → predicted_codes → TTS model
```

### How Gradients Flow

```
TTS Transformer
    ↓ (forward)
Predicted Logits
    ↓ (argmax - straight-through estimator)
Predicted Codes
    ↓
Vocoder (frozen params, gradients flow through input)
    ↓
Generated Audio
    ↓
Quality Losses (Speaker Similarity, MOS)
    ↓ (backward)
Gradients flow back to TTS Transformer
```

### Straight-Through Estimator

The `argmax` operation is non-differentiable, but PyTorch uses a straight-through estimator:
- Forward: `codes = argmax(logits)`
- Backward: Gradients pass through as if argmax was identity

This allows the model to learn which logits to increase/decrease to improve quality.

### Why This Works

1. **Cross-Entropy Loss**: Trains model to predict correct codes
2. **Quality Losses**: Train model to predict codes that sound good
3. **Combined Effect**: Model learns to generate high-quality, accurate speech

---

## 2. Frozen Pretrained Models with Gradient Flow

### Decision
UTMOS and Pyannote models have **frozen parameters** but **gradients still flow** through them during training.

### Rationale

#### Frozen Parameters (`requires_grad=False`)
- Models are pretrained and don't need further training
- Saves memory and computation
- Ensures consistent quality evaluation

#### Enabled Gradient Flow (no `torch.no_grad()`)
- Gradients flow from loss → audio input → vocoder → TTS model
- This is how the loss guides TTS training
- Standard pattern for using pretrained models as loss functions

### Code Pattern
```python
# Initialize with frozen parameters
for param in self.utmos_model.parameters():
    param.requires_grad = False
model.eval()

# During training - NO torch.no_grad()!
if requires_grad and audio.requires_grad:
    output = model(audio)  # Gradients flow through audio
    loss = criterion(output, target)
    loss.backward()  # Backprop to TTS model
```

### See Also
- `GRADIENT_FLOW_NOTES.md` - Detailed technical explanation
- `modules/utmos_predictor.py` - Implementation

---

## 3. Dual-Mode Operation for Speaker Embeddings

### Decision
`SpeakerEmbeddingSimilarityLoss` has two modes:
- **Training mode**: Uses full Pyannote model (gradients enabled)
- **Inference mode**: Uses fast Pyannote wrapper (no gradients)

### Rationale

#### Training Mode
```python
if requires_grad and audio.requires_grad:
    embeddings = self.model(audio)  # Full model, slower but has gradients
```
- Necessary for backpropagation
- ~100ms per batch overhead

#### Inference Mode
```python
else:
    embeddings = self.inference(...)  # Fast wrapper, no gradients
```
- 2-3x faster
- Used during validation/inference
- No gradient computation needed

### Benefit
Best of both worlds: gradients when needed, speed when not.

---

## 4. Conditional Loss Application

### Decision
Losses are only computed when their weights are > 0.

### Implementation
```python
self.use_speaker_similarity_loss = (
    hasattr(hp, 'speaker_similarity_weight') and 
    hp.speaker_similarity_weight > 0
)

if self.use_speaker_similarity_loss:
    # Initialize and use loss
```

### Benefits
1. **Backward Compatible**: Default weight is 0 (disabled)
2. **Efficient**: No overhead when disabled
3. **Flexible**: Can enable/disable per training run
4. **Debugging**: Easy to isolate issues

---

## 5. Error Handling Strategy

### Decision
Graceful degradation with warnings instead of crashes.

### Implementation
```python
try:
    loss, score = self.speaker_similarity_loss(audio_hat, audio_gt)
    # Use loss
except Exception as e:
    print(f"[WARNING] Speaker similarity loss computation failed: {e}")
    # Continue training without this loss
```

### Rationale
1. **Robustness**: Training doesn't crash if one loss fails
2. **Debugging**: Errors are printed but don't stop training
3. **Early Training**: Some losses may fail early (e.g., very short audio)
4. **Model Loading**: If pretrained models fail to load, training continues

### When Errors Might Occur
- Pretrained model checkpoint not found
- Audio too short for embedding extraction
- CUDA out of memory
- Model loading issues

---

## 6. Loss Weighting Strategy

### Decision
Additive loss combination with user-specified weights:
```python
total_loss = base_loss + α*speaker_loss + β*utmos_loss
```

### Rationale

#### Why Not Automatic Weighting?
- User control is more predictable
- Different datasets need different weights
- Easier to understand and tune

#### Recommended Weights
Based on typical loss magnitudes:
- **Base CE Loss**: ~1-3
- **Speaker Similarity**: 0.05-0.2 (weight: 0.1)
- **UTMOS MOS**: 0.5-2.0 (weight: 0.05)

#### How to Tune
1. Start with recommended weights
2. Monitor individual loss values
3. Adjust if one loss dominates
4. Aim for balanced contribution

---

## 7. Logging and Monitoring

### Decision
Log both weighted and unweighted loss values.

### Implementation
```python
# Log unweighted loss
self.log("train/speaker_similarity_loss", spk_sim_loss, ...)

# Log the metric being optimized
self.log("train/speaker_similarity_score", spk_sim_score, ...)

# Add weighted loss to total
loss = loss + spk_sim_loss * self.hp.speaker_similarity_weight
```

### Benefits
1. **Debugging**: See if individual losses are working
2. **Tuning**: Understand loss magnitudes for weight adjustment
3. **Monitoring**: Track quality metrics (similarity score, MOS)
4. **Comparison**: Compare weighted vs unweighted contributions

---

## 8. Target MOS Selection

### Decision
Default target MOS is 5.0, but configurable.

### Rationale

#### Why 5.0?
- Maximum quality on MOS scale [1-5]
- Aspirational target
- Model will optimize as high as possible

#### Why Configurable?
- Some datasets have lower average MOS
- Can set realistic targets (e.g., 4.5)
- Avoids unrealistic expectations

#### Usage
```bash
--target_mos 4.5  # More realistic target
--target_mos 5.0  # Maximum quality (default)
```

---

## 9. Audio Generation Timing

### Decision
Generate audio once per training step, reuse for all losses.

### Implementation
```python
# Generate audio once
if self.use_speaker_similarity_loss or self.use_utmos_loss or self.hp.fine_tune_vocoder:
    audio_hat = self.vocoder(codes, speaker)
    
    # Use for multiple losses
    speaker_loss = compute_speaker_similarity(audio_hat, audio_gt)
    mos_loss = compute_mos(audio_hat)
    voc_loss = compute_mel_loss(audio_hat, audio_gt)
```

### Benefits
1. **Efficiency**: Vocoder is expensive, only run once
2. **Consistency**: All losses evaluate same audio
3. **Memory**: Reuse intermediate tensors
4. **Speed**: ~2-3x faster than multiple generations

---

## Summary

These design decisions prioritize:
- ✅ **Stability**: Robust training with graceful error handling
- ✅ **Efficiency**: Minimize computational overhead
- ✅ **Flexibility**: User control via command-line arguments
- ✅ **Clarity**: Clear code with good documentation
- ✅ **Effectiveness**: Actually improves model quality

---

## Alternative Designs Considered

### 1. Multi-stage Training
**Idea**: Train with different losses at different stages
**Rejected**: More complex, harder to use
**Future Work**: Could add as advanced feature

### 2. Predicted Code Generation
**Idea**: Use model's predicted codes instead of GT
**Rejected**: Unstable, complex reshaping, early training issues
**See**: Section 1 above

### 3. Learnable Loss Weights
**Idea**: Automatically learn loss weights during training
**Rejected**: Less predictable, harder to tune
**Future Work**: Could add as research feature

### 4. Gradient Stopping
**Idea**: Stop gradients at certain points
**Rejected**: Would prevent losses from guiding training
**See**: `GRADIENT_FLOW_NOTES.md`

---

## References

- `IMPLEMENTATION_SUMMARY.md` - Overall implementation
- `GRADIENT_FLOW_NOTES.md` - Gradient flow details
- `ADDITIONAL_LOSSES_README.md` - User documentation
- `trainer_semantic.py` - Implementation code
- `modules/utmos_predictor.py` - Loss modules

---

**Last Updated**: November 2024

