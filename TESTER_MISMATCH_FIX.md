# Critical Issue: Tester/Checkpoint Mismatch

## Problem Discovered

Your checkpoint `style_pyannote_epoch=119-step=142486.ckpt` was trained with the **SEMANTIC model architecture**, but you were trying to use it with the **VANILLA tester.py**, which causes wildly wrong outputs!

## Why This Happens

### Two Different Model Architectures:

1. **Vanilla Model (`wildttstransformer.py`):**
   ```python
   from modules.wildttstransformer import TTSDecoder
   
   # Inference signature:
   def inference_topkp_sampling_batch(self, phone, spkr, phone_mask, prior=None, ...)
   ```
   - Uses ONLY: phone features + speaker embedding
   - No semantic conditioning

2. **Semantic Model (`wildttstransformer_semantic.py`):**
   ```python
   from modules.wildttstransformer_semantic import TTSDecoder
   
   # Inference signature:
   def inference_topkp_sampling_batch(self, phone, spkr, semantic, phone_mask, prior=None, ...)
   ```
   - Uses: phone features + speaker embedding + **semantic embedding (SBERT)**
   - Has additional layers:
     - `TTSdecoder.semantic_linear` (384 → hidden_size)
     - `TTSdecoder.layer_norm_semantic`

## What Went Wrong

When you used `tester.py` (vanilla) with a semantic checkpoint:

1. ✅ Model loads with `strict=False`
2. ❌ **Semantic layers are skipped/ignored** (not in vanilla architecture)
3. ❌ Model runs WITHOUT semantic conditioning
4. ❌ **Output is wildly wrong** because the model was trained expecting semantic input!

It's like training a car with 4 wheels, then trying to drive it with only 3 wheels - it technically "runs" but performs terribly.

## Checkpoint Analysis

Your checkpoint contains:
```
Semantic-related keys:
- TTSdecoder.semantic_linear.weight
- TTSdecoder.semantic_linear.bias  
- TTSdecoder.layer_norm_semantic.weight
- TTSdecoder.layer_norm_semantic.bias
- semantic_encoder.model.* (entire SBERT model)
```

**Conclusion:** This is a SEMANTIC checkpoint and MUST use semantic inference scripts!

## Solution

### Use the Correct Scripts:

| Checkpoint Type | Training Script | Inference Script | Tester Script |
|----------------|----------------|------------------|---------------|
| **Vanilla** | `train.py` | `infer.py` | `tester.py` |
| **Semantic** (your case) | `train_semantic.py` | `infer_semantic.py` | `tester_semantic.py` |

### Correct Command for Your Checkpoint:

```bash
python infer_semantic.py \
    --phonemizer_dict_path ./en_us_cmudict_forward.pt \
    --outputdir ./Infer_Semv0 \
    --model_path ./style_pyannote_epoch=119-step=142486.ckpt \
    --input_path ./speaker_to_text_test.json \
    --config_path ./ckpt_stylepyannote/config.json \
    --batch_size 1 \
    --vocoder_ckpt_path ./quantizer/checkpoints/g_style \
    --vocoder_config_path ./quantizer/config.json
```

## Validation Added

I've added automatic checks to prevent this issue:

### In `tester.py` (vanilla):
```python
# CRITICAL: Check if checkpoint was trained with semantic encoder
has_semantic = any('semantic' in k.lower() for k in state_dict.keys())
if has_semantic:
    raise ValueError("Checkpoint/Tester mismatch: Use tester_semantic.py for semantic checkpoints")
```

### In `tester_semantic.py`:
```python
# CRITICAL: Check if checkpoint was trained with semantic encoder
has_semantic = any('semantic' in k.lower() for k in state_dict.keys())
if not has_semantic:
    print("[WARNING] This checkpoint does NOT contain semantic encoder weights!")
    print("[WARNING] If this is a vanilla checkpoint, use tester.py instead.")
```

Now the scripts will automatically detect and warn/error if there's a mismatch!

## Summary

- ❌ **NEVER use `tester.py` / `infer.py` with semantic checkpoints**
- ✅ **ALWAYS use `tester_semantic.py` / `infer_semantic.py` for semantic checkpoints**
- ✅ **Scripts now auto-detect and prevent mismatches**

The wildly wrong outputs you were seeing were because the model was running without its semantic conditioning, which it desperately needs since it was trained with it!


