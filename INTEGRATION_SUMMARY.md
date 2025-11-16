# MQTTS-XCodec Integration Summary

## Overview

This integration replaces the original MQTTS quantizer (Encoder + Quantizer) with XCodec, a state-of-the-art neural audio codec that integrates semantic information from pre-trained models. This enhancement improves semantic accuracy and audio quality in text-to-speech synthesis.

## Files Created

1. **`modules/xcodec_wrapper.py`**: Core wrapper that interfaces XCodec with MQTTS
   - Handles encoding/decoding with XCodec
   - Maps XCodec's codebook structure to MQTTS's expected format
   - Provides `encode()`, `forward()`, `embed()`, and `decode()` methods

2. **`modules/vocoder_xcodec.py`**: XCodec-based vocoder
   - Wraps XCodecWrapper for vocoder interface
   - Handles speaker conditioning (optional)
   - Provides `forward()` and `encode()` methods compatible with MQTTS

3. **`quantizer/get_labels_xcodec.py`**: Preprocessing script using XCodec
   - Replaces original `get_labels.py` for XCodec preprocessing
   - Generates quantization codes compatible with MQTTS training

4. **`XCODEC_INTEGRATION.md`**: Comprehensive documentation
   - Usage instructions
   - Architecture details
   - Troubleshooting guide

## Files Modified

1. **`modules/vocoder.py`**: 
   - Added support for both original and XCodec vocoders
   - Backward compatible with existing code

2. **`trainer.py`**: 
   - Updated to initialize XCodec vocoder when `use_xcodec=True`
   - Handles both vocoder types gracefully

3. **`tester.py`**: 
   - Updated inference to support XCodec
   - Handles frame-to-sample ratio differences

4. **`train.py`**: 
   - Added XCodec-related command-line arguments
   - `--use_xcodec`: Enable XCodec integration
   - `--xcodec_model_name`: Specify XCodec model
   - `--frame_to_sample_ratio`: Adjust for different vocoders

5. **`infer.py`**: 
   - Added XCodec support flags
   - Handles both vocoder types

6. **`requirements.txt`**: 
   - Added `transformers>=4.40.0`
   - Added `accelerate>=0.20.0`

## Key Design Decisions

### 1. Codebook Mapping
- XCodec typically has 8 codebooks with 1024 codes each
- MQTTS expects 4 code groups with 160 codes each
- **Solution**: Use first 4 XCodec codebooks, map to MQTTS format
- XCodec's larger codebooks provide richer representations

### 2. Backward Compatibility
- Original MQTTS quantizer still supported
- Can switch between original and XCodec via `use_xcodec` flag
- No breaking changes to existing code

### 3. Pre-trained Models
- XCodec models are pre-trained on large datasets
- No need to train quantizer from scratch
- Can fine-tune XCodec if needed (optional)

### 4. Error Handling
- Robust error handling for XCodec API variations
- Fallback mechanisms for different XCodec versions
- Clear error messages for debugging

## Usage Workflow

### Step 1: Preprocess with XCodec
```bash
python quantizer/get_labels_xcodec.py \
    --input_json datasets/train.json \
    --input_wav_dir datasets/audios \
    --output_json datasets/train_q.json \
    --xcodec_model_name facebook/xcodec-base \
    --n_code_groups 4 \
    --sample_rate 16000
```

### Step 2: Train MQTTS with XCodec
```bash
python train.py \
    --use_xcodec \
    --xcodec_model_name facebook/xcodec-base \
    --datadir datasets/audios \
    --metapath datasets/train_q.json \
    --val_metapath datasets/dev_q.json \
    # ... other training arguments
```

### Step 3: Inference
```bash
python infer.py \
    --model_path ckpt/last.ckpt \
    --config_path ckpt/config.json \
    # ... other inference arguments
```

## Technical Details

### Code Format
- **Input to Transformer**: Codes in format `(B, T, n_code_groups)` where `n_code_groups=4`
- **XCodec Output**: Codes in format `(B, T', n_codebooks)` where `n_codebooks=8`
- **Mapping**: Take first 4 codebooks, reshape to match MQTTS format

### Frame Rate
- Original MQTTS: 256 samples per frame
- XCodec: May differ (typically 320 samples per frame)
- **Solution**: Configurable `frame_to_sample_ratio` parameter

### Speaker Conditioning
- XCodec doesn't have built-in speaker conditioning
- **Solution**: Added optional speaker projection layer in `VocoderXCodec`
- Can be enhanced with more sophisticated conditioning if needed

## Benefits

1. **Better Semantic Accuracy**: XCodec integrates semantic features from pre-trained models
2. **Larger Codebooks**: 1024 codes per codebook vs 160, richer representations
3. **Pre-trained**: No need to train quantizer from scratch
4. **State-of-the-Art**: Uses latest neural audio codec technology

## Limitations & Future Work

1. **API Compatibility**: XCodec API may vary across versions - wrapper handles this
2. **Speaker Conditioning**: Basic implementation, can be enhanced
3. **Fine-tuning**: XCodec fine-tuning not fully tested, may need adjustments
4. **Memory**: XCodec models are larger, may need GPU memory adjustments

## Testing Recommendations

1. Test with small dataset first
2. Verify code dimensions match expectations
3. Check audio quality vs original MQTTS
4. Monitor training stability
5. Validate inference speed and quality

## References

- XCodec Paper: https://arxiv.org/pdf/2408.17175
- XCodec GitHub: https://github.com/zhenye234/xcodec
- XCodec HuggingFace: https://huggingface.co/docs/transformers/en/model_doc/xcodec
- MQTTS Paper: https://arxiv.org/pdf/2302.04215

