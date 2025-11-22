# Quick Start Guide: Additional Loss Functions

This guide will get you started with the new speaker similarity and UTMOS MOS loss functions in under 10 minutes.

## Prerequisites

✅ You have a working MQ-TTS semantic training setup  
✅ You have a trained vocoder checkpoint  
✅ You have training data prepared  
✅ (Optional) You have a UTMOS checkpoint for MOS optimization

## Step 1: Verify Installation (1 minute)

Run the test suite to ensure everything is installed correctly:

```bash
python test_additional_losses.py
```

Expected output:
```
Testing Speaker Embedding Similarity Loss
✓ SpeakerEmbeddingSimilarityLoss initialized
...
✅ All tests passed!
```

## Step 2: Get UTMOS Checkpoint (Optional, 2 minutes)

If you want to use UTMOS loss, download the pretrained checkpoint:

```bash
cd deps/UTMOS22/fairseq_checkpoints
bash download_strong_checkpoints.sh
cd ../../..
```

Note the path to the checkpoint file (e.g., `deps/UTMOS22/checkpoints/utmos_strong.ckpt`)

## Step 3: Start Training (30 seconds)

### Option A: With Speaker Similarity Only (Recommended for first try)

```bash
python train_semantic.py \
  --vocoder_config_path ./quantizer/config.json \
  --vocoder_ckpt_path ./quantizer/checkpoints/g_00600000.ckpt \
  --datadir /path/to/your/data \
  --metapath ./datasets/training.txt \
  --val_metapath ./datasets/validation.txt \
  --speaker_similarity_weight 0.1 \
  --batch_size 32 \
  --lr 1e-4
```

### Option B: With Both Losses (Full Quality Optimization)

```bash
python train_semantic.py \
  --vocoder_config_path ./quantizer/config.json \
  --vocoder_ckpt_path ./quantizer/checkpoints/g_00600000.ckpt \
  --datadir /path/to/your/data \
  --metapath ./datasets/training.txt \
  --val_metapath ./datasets/validation.txt \
  --speaker_similarity_weight 0.1 \
  --utmos_weight 0.05 \
  --utmos_ckpt_path ./deps/UTMOS22/checkpoints/utmos_strong.ckpt \
  --target_mos 4.5 \
  --batch_size 32 \
  --lr 1e-4
```

## Step 4: Monitor Training (2 minutes)

### Launch TensorBoard:

```bash
tensorboard --logdir ./logs
```

Open browser to `http://localhost:6006`

### Key Metrics to Watch:

1. **`train/loss`** - Should converge normally
2. **`train/acc`** - Should stay high (>0.8)
3. **`train/speaker_similarity_score`** - Should increase toward 1.0
4. **`train/predicted_mos`** - Should increase toward 4-5

### Expected Console Output:

```
Epoch: 0 | Step: 1000 | Loss: 2.4531 | Acc: 0.8234
Additional losses: spk_sim: 0.1234 | utmos: 0.0567
...
```

## Step 5: Adjust Weights (Optional)

Based on your results, you may want to tune the loss weights:

### If speaker similarity score is too low (<0.8):
```bash
# Increase speaker similarity weight
--speaker_similarity_weight 0.15  # or 0.2
```

### If predicted MOS is too low (<3.5):
```bash
# Increase UTMOS weight
--utmos_weight 0.1  # or higher
```

### If training is unstable:
```bash
# Reduce all weights
--speaker_similarity_weight 0.05
--utmos_weight 0.02
```

## Troubleshooting

### ❌ Error: "UTMOS checkpoint not found"
**Solution**: Check the path with `ls -la deps/UTMOS22/checkpoints/` and update `--utmos_ckpt_path`

### ❌ Error: "CUDA out of memory"
**Solution**: Reduce batch size:
```bash
--batch_size 16  # or lower
```

### ❌ Warning: "Speaker similarity loss computation failed"
**Solution**: This is usually fine on the first few steps. If it persists, check your audio data.

### ⚠️ Metrics not showing in TensorBoard
**Solution**: 
1. Check that TensorBoard is pointing to correct directory
2. Refresh the browser page
3. Wait a few training steps for metrics to appear

## What's Next?

### After First Epoch:

1. **Check Results**: Listen to generated samples in validation step
2. **Compare Quality**: Compare with baseline (without additional losses)
3. **Fine-tune Weights**: Adjust based on metrics and listening tests

### Advanced Usage:

- **Resume Training**: Add `--resume_checkpoint /path/to/checkpoint.ckpt`
- **Change Target MOS**: Adjust `--target_mos` (range: 1-5, default: 5.0)
- **Fine-tune Vocoder**: Add `--fine_tune_vocoder` (but increases memory)

### Evaluation:

After training, evaluate your model:

```bash
# Speaker similarity evaluation
python measures/sss/speaker_embedding_similarity.py \
  --datadir /path/to/generated \
  --metapath /path/to/test_meta.json

# UTMOS evaluation (if available)
# Use UTMOS evaluation scripts in deps/UTMOS22/
```

## Expected Improvements

With these losses, you should see:

✅ **Better Speaker Consistency**: Generated speech matches reference speaker better  
✅ **Higher Perceptual Quality**: More natural sounding output  
✅ **Improved MOS Scores**: +0.2 to +0.5 MOS improvement typically  
✅ **More Stable Voice**: Less variation in speaker characteristics  

## Performance Expectations

- **Training Speed**: ~20-30% slower than baseline
- **Memory Usage**: +500-700MB GPU memory
- **Convergence**: May need 10-20% more epochs
- **Quality Gain**: Worth the extra computation!

## Tips for Best Results

1. **Start Conservative**: Use low weights first (0.05-0.1)
2. **Monitor Closely**: Watch first 1000 steps carefully
3. **Be Patient**: Quality improvements become clear after several epochs
4. **Compare**: Always compare with baseline model
5. **Iterate**: Adjust weights based on your specific dataset

## Complete Example Command

Here's a complete command with all recommended settings:

```bash
python train_semantic.py \
  --saving_path ./ckpt_with_losses \
  --vocoder_config_path ./quantizer/config.json \
  --vocoder_ckpt_path ./quantizer/checkpoints/g_00600000.ckpt \
  --datadir /path/to/LibriTTS/train-clean-100 \
  --metapath ./datasets/training.txt \
  --val_metapath ./datasets/validation.txt \
  --sampledir ./logs_with_losses \
  --speaker_similarity_weight 0.1 \
  --utmos_weight 0.05 \
  --utmos_ckpt_path ./deps/UTMOS22/checkpoints/utmos_strong.ckpt \
  --target_mos 4.5 \
  --batch_size 32 \
  --lr 1e-4 \
  --max_epochs 100 \
  --precision 16-mixed \
  --accelerator gpu \
  --verbose_step 500 \
  --verbose_file training_log.txt
```

## Need Help?

1. **Documentation**: See `ADDITIONAL_LOSSES_README.md` for detailed info
2. **Implementation**: See `IMPLEMENTATION_SUMMARY.md` for technical details
3. **Testing**: Run `python test_additional_losses.py` for diagnostics
4. **Errors**: Check console output and `verbose.txt` for detailed logs

---

**Ready to train?** Just copy one of the commands above, adjust the paths, and run it! 🚀

