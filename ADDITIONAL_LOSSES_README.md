# Additional Loss Functions for MQ-TTS Semantic Training

This document describes the new loss functions added to optimize speaker similarity and MOS (Mean Opinion Score) directly during training.

## Overview

Two additional loss functions have been integrated into the semantic training pipeline:

1. **Speaker Embedding Cosine Similarity Loss**: Optimizes the speaker similarity between generated and reference audio using Pyannote speaker embeddings.

2. **UTMOS MOS Score Loss**: Directly optimizes Mean Opinion Score using a pretrained frozen UTMOS model to predict and maximize audio quality.

## Architecture

### New Modules

#### `modules/utmos_predictor.py`

Contains two main classes:

1. **`UTMOSPredictor`**: 
   - Loads a pretrained UTMOS model with frozen weights
   - Provides MOS score prediction for audio samples
   - Computes loss that encourages higher MOS scores

2. **`SpeakerEmbeddingSimilarityLoss`**:
   - Extracts speaker embeddings using Pyannote
   - Computes cosine similarity between generated and reference audio
   - Returns loss as `1 - similarity` (minimize to maximize similarity)

### Integration in Trainer

The losses are integrated into `trainer_semantic.py`:

- Models are initialized in `__init__()` with frozen weights
- During `training_step()`:
  1. Predicted acoustic codes are generated from the TTS transformer
  2. Audio is synthesized using the vocoder
  3. Additional losses are computed on the generated audio
  4. Losses are weighted and added to the base cross-entropy loss

## Usage

### Command Line Arguments

New arguments added to `train_semantic.py`:

```bash
python train_semantic.py \
  --speaker_similarity_weight 0.1 \
  --utmos_weight 0.05 \
  --utmos_ckpt_path /path/to/utmos_checkpoint.ckpt \
  --target_mos 5.0 \
  [... other arguments ...]
```

#### Arguments:

- `--speaker_similarity_weight`: Weight for speaker embedding similarity loss (default: 0.0 = disabled)
- `--utmos_weight`: Weight for UTMOS MOS loss (default: 0.0 = disabled)
- `--utmos_ckpt_path`: Path to pretrained UTMOS checkpoint (required if utmos_weight > 0)
- `--target_mos`: Target MOS score to optimize towards (default: 5.0, range: 1-5)

### Example Training Commands

#### Enable both losses:
```bash
python train_semantic.py \
  --vocoder_config_path ./quantizer/config.json \
  --vocoder_ckpt_path ./quantizer/checkpoints/g_00600000.ckpt \
  --datadir /path/to/data \
  --metapath ./datasets/training.txt \
  --val_metapath ./datasets/validation.txt \
  --speaker_similarity_weight 0.1 \
  --utmos_weight 0.05 \
  --utmos_ckpt_path ./deps/UTMOS22/checkpoints/utmos_strong.ckpt \
  --target_mos 4.5 \
  --batch_size 32 \
  --lr 1e-4
```

#### Enable only speaker similarity:
```bash
python train_semantic.py \
  [... other arguments ...] \
  --speaker_similarity_weight 0.15
```

#### Enable only UTMOS loss:
```bash
python train_semantic.py \
  [... other arguments ...] \
  --utmos_weight 0.1 \
  --utmos_ckpt_path ./deps/UTMOS22/checkpoints/utmos_strong.ckpt
```

### Recommended Weight Values

Based on typical loss magnitudes:

- **Speaker Similarity Weight**: 0.05 - 0.2
  - Start with 0.1 and adjust based on convergence
  - Higher values emphasize speaker consistency
  
- **UTMOS Weight**: 0.01 - 0.1
  - Start with 0.05
  - UTMOS loss can be noisy early in training; consider ramping up weight
  - Monitor the `train/predicted_mos` metric

## UTMOS Setup

### Obtaining UTMOS Checkpoint

1. Navigate to the UTMOS directory:
```bash
cd deps/UTMOS22/fairseq_checkpoints
```

2. Download the pretrained checkpoint:
```bash
bash download_strong_checkpoints.sh
```

3. The checkpoint will be downloaded to an appropriate location. Note the path for use with `--utmos_ckpt_path`.

### Alternative: Train Your Own UTMOS Model

Refer to `deps/UTMOS22/strong/README.md` for instructions on training UTMOS from scratch.

## Monitoring Training

### TensorBoard Metrics

The following new metrics are logged:

#### Per-step metrics:
- `train/speaker_similarity_loss`: Unweighted speaker similarity loss (0 = perfect match)
- `train/speaker_similarity_score`: Cosine similarity score (1.0 = perfect match)
- `train/utmos_loss`: Unweighted UTMOS loss
- `train/predicted_mos`: Predicted MOS score from UTMOS (1-5 scale)

#### Verbose logging:
Every `--verbose_step` steps, additional losses are printed to console and written to `--verbose_file`.

### Example TensorBoard Command:
```bash
tensorboard --logdir ./logs
```

## Implementation Details

### Audio Generation for Loss Computation

The losses require audio generation during training:

1. **Predicted Code Extraction**: The TTS transformer's logits are used to get predicted acoustic codes
2. **Vocoder Synthesis**: Codes are passed through the vocoder to generate audio
3. **Length Matching**: Generated and reference audio are trimmed to matching lengths
4. **Loss Computation**: Speaker embeddings and MOS scores are computed on the generated audio

### Computational Cost

- **Speaker Similarity Loss**: ~50-100ms per batch (Pyannote inference)
- **UTMOS Loss**: ~100-200ms per batch (WavLM + LSTM inference)
- Total overhead: ~150-300ms per training step

For faster training, consider:
- Using a lower computation frequency (e.g., apply losses every N steps)
- Reducing batch size if GPU memory is constrained
- Using mixed precision training (`--precision 16-mixed`)

### Memory Requirements

Additional GPU memory is required for:
- UTMOS model (~500MB)
- Pyannote embedding model (~200MB)
- Intermediate audio tensors during synthesis

Ensure sufficient GPU memory (recommend 16GB+ for batch sizes of 32).

## Troubleshooting

### Issue: UTMOS checkpoint not found
**Solution**: Verify the path with `--utmos_ckpt_path` is correct. The checkpoint should be a `.ckpt` file from UTMOS training.

### Issue: Speaker similarity loss is NaN
**Solution**: This can occur if:
- Audio is too short (< 1 second)
- Pyannote model failed to load
- Check console for warning messages

### Issue: UTMOS predictions are constant
**Solution**: 
- Ensure the UTMOS checkpoint is properly trained
- Check that audio sample rate matches (default: 16kHz)
- Early in training, generated audio may be very poor quality

### Issue: Training is very slow
**Solution**:
- Reduce loss computation frequency
- Disable one of the losses if not needed
- Use gradient accumulation to reduce effective batch size
- Use mixed precision training

## Future Enhancements

Potential improvements:

1. **Adaptive Loss Weighting**: Automatically adjust loss weights based on convergence
2. **Scheduled Loss Application**: Apply losses only after N steps when audio quality improves
3. **Batch-wise Sampling**: Compute losses on a subset of batch items for speed
4. **Alternative Metrics**: Add other quality metrics (PESQ, STOI, etc.)

## References

- **UTMOS**: Saeki et al., "UTMOS: UTokyo-SaruLab System for VoiceMOS Challenge 2022"
- **Pyannote**: Bredin et al., "Pyannote.audio: neural building blocks for speaker diarization"
- **MQ-TTS**: Original MQ-TTS paper and implementation

## License

This implementation follows the same license as the base MQ-TTS project.

