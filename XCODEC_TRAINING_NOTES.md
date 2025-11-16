# XCodec Training and Weight Loading

## Important Clarification

**The XCodec vocoder is NOT trained in this codebase.** It uses **pre-trained weights** loaded from HuggingFace.

## Weight Loading Flow

### 1. Pre-trained Weights Source
- **Location**: HuggingFace model hub
- **Model Name**: Default is `facebook/xcodec-base` (configurable via `--xcodec_model_name`)
- **When Loaded**: During `XCodecWrapper` initialization
- **Code Location**: `modules/xcodec_wrapper.py`, line 35
  ```python
  self.xcodec_model = XCodecModel.from_pretrained(model_name)
  ```

### 2. Weight Loading Process

```
XCodecWrapper.__init__()
  └─> XCodecModel.from_pretrained("facebook/xcodec-base")
      └─> Downloads/loads pre-trained weights from HuggingFace
          └─> Weights are pre-trained on large datasets (not trained here)
```

### 3. During Training

In `trainer.py`:
- XCodec vocoder is initialized with pre-trained weights
- **By default, all vocoder parameters are FROZEN** (same as original MQTTS)
- The transformer learns to generate XCodec codes from text
- XCodec itself is used only for encoding/decoding (not trained)

```python
# In trainer.py
self.vocoder.eval()
for param in self.vocoder.parameters():
    param.requires_grad = False  # Frozen by default
```

## Comparison with Original MQTTS

| Aspect | Original MQTTS | MQTTS-XCodec |
|--------|----------------|--------------|
| **Quantizer Training** | Trained in `quantizer/train.py` | **Not needed** (pre-trained) |
| **Weight Source** | Trained locally | HuggingFace (pre-trained) |
| **Training Script** | `quantizer/train.py` | **None** (uses pre-trained) |
| **During Transformer Training** | Frozen | Frozen (by default) |
| **Fine-tuning** | Not applicable | Optional (see below) |

## Fine-tuning XCodec (Optional)

If you want to fine-tune XCodec on your dataset:

### Option 1: Fine-tune during Transformer Training

```bash
python train.py \
    --use_xcodec \
    --fine_tune_xcodec \
    --xcodec_model_name facebook/xcodec-base \
    # ... other args
```

This will:
- Load pre-trained XCodec weights
- Unfreeze XCodec parameters
- Fine-tune XCodec along with the transformer

**Note**: This requires more GPU memory and longer training time.

### Option 2: Fine-tune XCodec Separately (Recommended)

Create a separate training script (similar to `quantizer/train.py`) to fine-tune XCodec:

```python
# Example: train_xcodec.py (not included, but you can create)
from modules.xcodec_wrapper import XCodecWrapper
import torch

# Load pre-trained XCodec
xcodec = XCodecWrapper(
    model_name="facebook/xcodec-base",
    n_code_groups=4,
    freeze_encoder=False  # Allow training
)

# Fine-tune on your dataset
# ... training loop ...
```

## Why Pre-trained XCodec?

1. **Large-scale Pre-training**: XCodec is pre-trained on vast datasets (similar to how LLMs are pre-trained)
2. **Better Representations**: Pre-trained models capture rich semantic and acoustic features
3. **No Training Needed**: Just load and use (like using a pre-trained BERT)
4. **Transfer Learning**: Benefits from knowledge learned on large datasets

## Summary

- ✅ **XCodec weights**: Loaded from HuggingFace (pre-trained)
- ✅ **Loading location**: `modules/xcodec_wrapper.py` → `XCodecModel.from_pretrained()`
- ✅ **Training**: XCodec is NOT trained (uses pre-trained weights)
- ✅ **During training**: XCodec is frozen (like original vocoder)
- ✅ **Fine-tuning**: Optional, can be enabled with `--fine_tune_xcodec`

The main `train.py` script trains the **transformer** that generates XCodec codes from text. The XCodec vocoder itself uses pre-trained weights and is frozen during training (unless you enable fine-tuning).

