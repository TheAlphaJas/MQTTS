# Inference Scripts Fixes - Summary

## Issues Found and Fixed

### 1. **Missing Vocoder Arguments (CRITICAL)**
**Files affected:** `infer.py`, `infer_semantic.py`

**Problem:**
- `--vocoder_config_path` and `--vocoder_ckpt_path` were NOT defined in the argument parser
- Even though users passed these on command line, they were silently ignored
- This caused the vocoder to initialize with **random weights** instead of loading pretrained checkpoint
- Led to the error message: "Initializing Vocoder with random weights (expecting load later)"

**Fix:**
```python
parser.add_argument('--vocoder_config_path', type=str, required=True)
parser.add_argument('--vocoder_ckpt_path', type=str, required=True)
```

### 2. **Config File Overriding Command-Line Arguments**
**Files affected:** `infer.py`, `infer_semantic.py`

**Problem:**
- Old code: `argdict.update(args.__dict__)` then `args.__dict__ = argdict`
- This meant config.json values would OVERRIDE user's command-line arguments
- If config had `"vocoder_ckpt_path": null`, it would ignore your CLI argument

**Fix:**
```python
# Store command-line args before they get overwritten
cmdline_args = args.__dict__.copy()
# Update with config
args.__dict__.update(argdict)
# Override with command-line args (command-line takes precedence)
for key, value in cmdline_args.items():
    if key in ['config_path', 'model_path', 'input_path', 'outputdir', 'phonemizer_dict_path', 
               'vocoder_config_path', 'vocoder_ckpt_path', 'spkr_embedding_path', 'batch_size']:
        args.__dict__[key] = value
```

### 3. **Style Encoder Detection from Config Instead of Checkpoint**
**Files affected:** `tester.py`, `tester_semantic.py`

**Problem:**
- Code checked `hp.style_encoder_type` from config to decide model architecture
- But config might not match what the checkpoint was actually trained with
- Led to shape mismatch: `RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x512 and 640x768)`
- 512 = Pyannote only, 640 = StyleTTS2 + Pyannote

**Fix:**
```python
# Check the actual checkpoint to see what was used during training
checkpoint = torch.load(self.hp.model_path, map_location='cpu')
state_dict = checkpoint['state_dict']
has_style_encoder = any('style_encoder' in k for k in state_dict.keys())
has_style_to_vocoder = any('style_to_vocoder' in k for k in state_dict.keys())

if has_style_encoder or has_style_to_vocoder:
    print("[INFO] Checkpoint was trained WITH style encoder...")
    spkr_embed_dim = 128 + 512  # StyleTTS2 + Pyannote
    self.style_encoder = StyleEncoder()
    self.style_to_vocoder = nn.Linear(128 + 512, 512)
else:
    print("[INFO] Checkpoint was trained WITHOUT style encoder...")
    spkr_embed_dim = 512  # Pyannote only
    self.style_encoder = None
    self.style_to_vocoder = None
```

### 4. **Deprecated Tensor Constructor**
**Files affected:** `tester.py`, `tester_semantic.py`

**Problem:**
```python
speaker_embeddings = torch.cuda.FloatTensor(np.array(speaker_embeddings))
```
- Deprecated in PyTorch 2.x
- Generates UserWarning

**Fix:**
```python
speaker_embeddings = torch.tensor(np.array(speaker_embeddings), dtype=torch.float32, device='cuda')
```

### 5. **No Validation for Pre-computed Embeddings with Style Encoder**
**Files affected:** `infer.py`, `infer_semantic.py`, `tester.py`, `tester_semantic.py`

**Problem:**
- If model was trained with StyleTTS2 style encoder, it needs BOTH:
  - Pyannote embeddings (512-dim)
  - StyleTTS2 style embeddings (128-dim) extracted from raw audio
- Pre-computed embeddings (--spkr_embedding_path) only contain Pyannote
- Would cause silent failures or incorrect results

**Fix:**
Added validation in both inference and tester scripts:
```python
if args.spkr_embedding_path and getattr(args, 'style_encoder_type', None) == 'style_tts2':
    raise ValueError("Cannot use pre-computed speaker embeddings with style encoder model")
```

### 6. **Missing Debug Output**
**Files affected:** `infer.py`, `infer_semantic.py`

**Problem:**
- Hard to diagnose issues like missing vocoder checkpoint
- No visibility into what paths are being loaded

**Fix:**
Added debug prints:
```python
print(f"[INFO] Model path: {args.model_path}")
print(f"[INFO] Vocoder config: {args.vocoder_config_path}")
print(f"[INFO] Vocoder checkpoint: {args.vocoder_ckpt_path}")
print(f"[INFO] Style encoder type: {getattr(args, 'style_encoder_type', None)}")
```

## Memory Leak Analysis

**Conclusion:** No significant memory leaks found in the inference loop.

Both `infer.py` and `infer_semantic.py` properly clear their lists after each batch:
```python
i_wavs, i_phones = [], []  # infer.py
i_wavs, i_phones, i_texts = [], [], []  # infer_semantic.py
```

The torch tensors are properly scoped within `with torch.no_grad():` context, and outputs are converted to numpy before being written to files.

## Corrected Usage Commands

### For Vanilla Model (no semantic encoder):
```bash
python infer.py \
    --phonemizer_dict_path ./en_us_cmudict_forward.pt \
    --outputdir ./Infer_Output \
    --model_path ./style_pyannote_epoch=119-step=142486.ckpt \
    --input_path ./speaker_to_text_test.json \
    --config_path ./ckpt_stylepyannote/config.json \
    --batch_size 1 \
    --vocoder_ckpt_path ./quantizer/checkpoints/g_style \
    --vocoder_config_path ./quantizer/config.json
```

### For Semantic Model:
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

**Note:** Do NOT use `--spkr_embedding_path` if your model was trained with StyleTTS2 style encoder!

## Files Modified
1. `infer.py` - Fixed vocoder args, config merge, added validation & debug
2. `infer_semantic.py` - Fixed vocoder args, config merge, added validation & debug
3. `tester.py` - Fixed checkpoint detection, deprecated tensor, added validation
4. `tester_semantic.py` - Fixed checkpoint detection, deprecated tensor, added validation


