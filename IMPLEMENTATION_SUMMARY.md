# Implementation Summary: Additional Loss Functions for MQ-TTS

## What Was Implemented

This implementation adds two additional loss functions to directly optimize audio quality metrics during TTS training:

### 1. Speaker Embedding Cosine Similarity Loss
- **Purpose**: Ensures generated audio maintains the same speaker characteristics as the reference
- **Method**: Extracts Pyannote speaker embeddings from both generated and reference audio, then maximizes cosine similarity
- **Benefit**: Better speaker consistency and voice cloning quality

### 2. UTMOS MOS Score Loss
- **Purpose**: Directly optimizes Mean Opinion Score (MOS) using a pretrained quality predictor
- **Method**: Uses a frozen UTMOS model to predict quality scores, then optimizes to maximize predicted MOS
- **Benefit**: Higher perceptual quality as measured by standard MOS metrics

## Files Modified

### New Files Created:
1. **`modules/utmos_predictor.py`** (285 lines)
   - `UTMOSPredictor`: Wrapper for frozen UTMOS model
   - `SpeakerEmbeddingSimilarityLoss`: Speaker embedding comparison

2. **`ADDITIONAL_LOSSES_README.md`** (279 lines)
   - Comprehensive documentation
   - Usage examples
   - Troubleshooting guide

3. **`test_additional_losses.py`** (220 lines)
   - Test suite for validation
   - Integration checks

4. **`IMPLEMENTATION_SUMMARY.md`** (This file)

### Modified Files:
1. **`trainer_semantic.py`**
   - Added import for new loss modules
   - Initialize loss modules in `__init__()`
   - Enhanced `training_step()` to:
     - Generate audio from predicted codes
     - Compute speaker similarity loss
     - Compute UTMOS MOS loss
     - Add weighted losses to total loss
     - Enhanced verbose logging

2. **`train_semantic.py`**
   - Added 4 new command-line arguments:
     - `--speaker_similarity_weight`
     - `--utmos_weight`
     - `--utmos_ckpt_path`
     - `--target_mos`

## How It Works

### Training Flow with Additional Losses

```
1. Normal Training Step:
   ├─ Text → Phone Features → Speaker Embedding
   ├─ TTS Transformer → Predicted Acoustic Codes
   └─ Cross-Entropy Loss (base loss)

2. Additional Quality Optimization (NEW):
   ├─ Predicted Codes → Vocoder → Generated Audio
   ├─ Speaker Similarity:
   │  ├─ Extract embedding from generated audio
   │  ├─ Extract embedding from reference audio
   │  └─ Compute: loss = 1 - cosine_similarity
   ├─ UTMOS MOS Score:
   │  ├─ UTMOS predicts MOS of generated audio
   │  └─ Compute: loss = MSE(predicted_MOS, target_MOS)
   └─ Total Loss = base_loss + α*speaker_loss + β*utmos_loss
```

### Key Design Decisions

1. **Frozen Pretrained Models**: UTMOS weights are frozen to use it purely as a loss function
2. **On-the-fly Audio Generation**: Audio is generated during training from predicted codes
3. **Flexible Weighting**: Loss weights can be set to 0 to disable specific losses
4. **Error Handling**: Graceful degradation if models fail to load
5. **Logging**: Comprehensive metrics logged to TensorBoard

## Usage Examples

### Basic Training (No Additional Losses):
```bash
python train_semantic.py \
  --vocoder_config_path ./quantizer/config.json \
  --vocoder_ckpt_path ./quantizer/checkpoints/g_00600000.ckpt \
  --datadir /path/to/data \
  --metapath ./datasets/training.txt \
  --val_metapath ./datasets/validation.txt
```

### With Speaker Similarity Loss:
```bash
python train_semantic.py \
  [... base args ...] \
  --speaker_similarity_weight 0.1
```

### With UTMOS Loss:
```bash
python train_semantic.py \
  [... base args ...] \
  --utmos_weight 0.05 \
  --utmos_ckpt_path ./deps/UTMOS22/checkpoints/utmos_strong.ckpt \
  --target_mos 4.5
```

### With Both Losses:
```bash
python train_semantic.py \
  [... base args ...] \
  --speaker_similarity_weight 0.1 \
  --utmos_weight 0.05 \
  --utmos_ckpt_path ./deps/UTMOS22/checkpoints/utmos_strong.ckpt
```

## Testing

Run the test suite to verify implementation:

```bash
# Test speaker similarity loss only
python test_additional_losses.py

# Test with UTMOS (requires checkpoint)
python test_additional_losses.py --utmos_ckpt_path /path/to/utmos.ckpt
```

## Expected Results

### Metrics to Monitor:

1. **`train/loss`**: Total loss (should still converge)
2. **`train/acc`**: Accuracy (should remain high)
3. **`train/speaker_similarity_score`**: Should increase over time (target: > 0.9)
4. **`train/predicted_mos`**: Should increase over time (target: > 4.0)
5. **`train/speaker_similarity_loss`**: Should decrease over time
6. **`train/utmos_loss`**: Should decrease over time

### Expected Training Impact:

- **Training Time**: +15-30% slower per step due to audio generation
- **Memory**: +500-700MB GPU memory for additional models
- **Convergence**: May require more epochs but should achieve better quality
- **Quality Improvement**: 
  - Better speaker consistency (measured by embedding similarity)
  - Higher perceived quality (measured by MOS)

## Hyperparameter Tuning

### Recommended Starting Points:

| Loss Type | Weight Range | Recommended Start | Notes |
|-----------|--------------|-------------------|-------|
| Speaker Similarity | 0.05 - 0.2 | 0.1 | Higher = more speaker consistency |
| UTMOS | 0.01 - 0.1 | 0.05 | Can be noisy early in training |

### Tuning Strategy:

1. **Start Conservative**: Begin with low weights (0.05 for both)
2. **Monitor Convergence**: Ensure base accuracy doesn't degrade
3. **Increase Gradually**: If stable, increase weights by 0.05
4. **Balance Losses**: Keep speaker_weight ≈ 2× utmos_weight typically
5. **Target Metrics**: 
   - Speaker similarity score > 0.85
   - Predicted MOS > 4.0

## Performance Considerations

### Computational Cost:
- **Speaker Similarity**: ~50-100ms per batch (Pyannote inference)
- **UTMOS**: ~100-200ms per batch (WavLM + LSTM)
- **Total Overhead**: ~20-30% slower training

### Memory Requirements:
- **Base Training**: ~8GB GPU memory
- **With Additional Losses**: ~10-12GB GPU memory
- **Recommended**: 16GB+ GPU for batch size ≥ 32

### Optimization Tips:
1. Use mixed precision: `--precision 16-mixed`
2. Reduce batch size if OOM
3. Use gradient accumulation for effective larger batches
4. Consider applying losses every N steps instead of every step

## Troubleshooting

### Common Issues:

1. **UTMOS checkpoint not loading**
   - Verify path is correct
   - Ensure checkpoint is compatible version
   - Check UTMOS dependencies are installed

2. **Speaker similarity is NaN**
   - Audio may be too short (need >1 second)
   - Check if Pyannote model loaded correctly
   - Ensure sample rate is 16kHz

3. **Training is unstable**
   - Reduce loss weights
   - Ensure base model converges first
   - Consider warming up (start with weights=0, gradually increase)

4. **OOM (Out of Memory)**
   - Reduce batch size
   - Use gradient accumulation
   - Use mixed precision training
   - Disable one of the losses

## Future Improvements

Potential enhancements:

1. **Adaptive Weighting**: Automatically adjust weights based on loss magnitudes
2. **Scheduled Losses**: Apply losses only after N steps when audio quality improves
3. **Efficient Sampling**: Apply losses to subset of batch for speed
4. **Multi-Stage Training**: Train with different loss combinations per stage
5. **Alternative Metrics**: Add PESQ, STOI, or other objective metrics

## Validation

To validate the implementation works:

1. **Run Test Suite**:
   ```bash
   python test_additional_losses.py --utmos_ckpt_path /path/to/checkpoint.ckpt
   ```

2. **Short Training Run**:
   ```bash
   python train_semantic.py \
     [... args ...] \
     --speaker_similarity_weight 0.1 \
     --max_epochs 1 \
     --val_check_interval 100
   ```

3. **Check Logs**:
   - Verify new metrics appear in TensorBoard
   - Confirm losses are computed without errors
   - Check verbose log file for additional loss values

## References

- **UTMOS Paper**: "UTMOS: UTokyo-SaruLab System for VoiceMOS Challenge 2022"
- **Pyannote**: https://github.com/pyannote/pyannote-audio
- **MQ-TTS**: Original repository and paper
- **StyleTTS2**: For style encoder implementation

## Contact & Support

For issues or questions:
1. Check ADDITIONAL_LOSSES_README.md for detailed documentation
2. Run test_additional_losses.py to diagnose problems
3. Review console output for error messages
4. Check TensorBoard for metric visualization

---

**Implementation Date**: November 2024  
**Status**: ✅ Complete and Ready for Testing

