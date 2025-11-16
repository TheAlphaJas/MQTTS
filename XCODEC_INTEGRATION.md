# XCodec Integration for MQTTS

This document describes the integration of XCodec (AAAI 2025) into the MQTTS pipeline, replacing the original quantizer architecture.

## Overview

XCodec is a unified semantic and acoustic codec that integrates semantic features from pre-trained models (like HuBERT) before the Residual Vector Quantization (RVQ) stage. This integration enhances semantic integrity in audio generation tasks.

## Key Changes

### 1. New Modules

- **`modules/xcodec_wrapper.py`**: Wrapper for XCodec that provides interface compatible with MQTTS quantizer
- **`modules/vocoder_xcodec.py`**: XCodec-based vocoder for encoding/decoding
- **`quantizer/get_labels_xcodec.py`**: Preprocessing script using XCodec instead of original quantizer

### 2. Modified Modules

- **`modules/vocoder.py`**: Updated to support both original and XCodec vocoders
- **`trainer.py`**: Updated to initialize XCodec vocoder when `use_xcodec=True`
- **`tester.py`**: Updated to support XCodec inference
- **`train.py`**: Added XCodec-related command-line arguments

## Installation

1. Install additional dependencies:
```bash
pip install transformers>=4.40.0 accelerate>=0.20.0
```

2. The XCodec model will be automatically downloaded from HuggingFace on first use.

## Usage

### Preprocessing with XCodec

Instead of training the original quantizer, use XCodec for preprocessing:

```bash
python quantizer/get_labels_xcodec.py \
    --input_json datasets/train.json \
    --input_wav_dir datasets/audios \
    --output_json datasets/train_q.json \
    --xcodec_model_name facebook/xcodec-base \
    --n_code_groups 4 \
    --sample_rate 16000
```

### Training with XCodec

Train the MQTTS transformer using XCodec:

```bash
python train.py \
    --distributed \
    --saving_path ckpt/ \
    --sampledir logs/ \
    --use_xcodec \
    --xcodec_model_name facebook/xcodec-base \
    --datadir datasets/audios \
    --metapath datasets/train_q.json \
    --val_metapath datasets/dev_q.json \
    --use_repetition_token \
    --ar_layer 4 \
    --ar_ffd_size 1024 \
    --ar_hidden_size 256 \
    --ar_nheads 4 \
    --speaker_embed_dropout 0.05 \
    --enc_nlayers 6 \
    --dec_nlayers 6 \
    --ffd_size 3072 \
    --hidden_size 768 \
    --nheads 12 \
    --batch_size 200 \
    --precision bf16-mixed \
    --training_step 800000 \
    --layer_norm_eps 1e-05 \
    --n_cluster_groups 4 \
    --sample_rate 16000
```

### Inference with XCodec

For inference, ensure the config includes `use_xcodec`:

```bash
CUDA_VISIBLE_DEVICES=0 python infer.py \
    --phonemizer_dict_path en_us_cmudict_forward.pt \
    --model_path ckpt/last.ckpt \
    --config_path ckpt/config.json \
    --input_path speaker_to_text.json \
    --outputdir infer_samples \
    --batch_size 32 \
    --top_p 0.8 \
    --min_top_k 2 \
    --max_output_length 15000 \
    --phone_context_window 3 \
    --clean_speech_prior
```

Note: The config.json should include `"use_xcodec": true` and `"xcodec_model_name": "facebook/xcodec-base"`.

## Architecture Details

### Code Structure Mapping

- **Original MQTTS**: Uses 4 code groups, each with 160 codes
- **XCodec**: Uses multiple RVQ codebooks (typically 8), each with 1024 codes
- **Mapping**: We use the first 4 XCodec codebooks to match MQTTS's 4 code groups

### Key Differences

1. **No Quantizer Training**: XCodec is pre-trained, so we skip the quantizer training step
2. **Different Codebook Sizes**: XCodec uses larger codebooks (1024 vs 160), providing richer representations
3. **Semantic Integration**: XCodec includes semantic features from pre-trained models, improving semantic accuracy

## Fine-Tuning

XCodec can be fine-tuned on your dataset. By default, XCodec weights are frozen during MQTTS training. To enable fine-tuning:

1. Set `freeze_encoder=False` in `XCodecWrapper` initialization
2. Ensure XCodec parameters have `requires_grad=True` in the vocoder

## Troubleshooting

### Model Loading Issues

If XCodec model fails to load:
- Check internet connection (model downloads from HuggingFace)
- Verify transformers version: `pip install --upgrade transformers`
- Try alternative model names from HuggingFace

### Code Dimension Mismatches

If you encounter dimension errors:
- Ensure `n_cluster_groups` matches between preprocessing and training
- Check that XCodec has enough codebooks (should have at least `n_cluster_groups`)

### Performance Issues

- XCodec models are larger than original quantizer - ensure sufficient GPU memory
- Consider using smaller XCodec models or reducing batch size
- XCodec encoding/decoding may be slower - adjust training parameters accordingly

## References

- XCodec Paper: https://arxiv.org/pdf/2408.17175
- XCodec GitHub: https://github.com/zhenye234/xcodec
- XCodec HuggingFace: https://huggingface.co/docs/transformers/en/model_doc/xcodec
- MQTTS Paper: https://arxiv.org/pdf/2302.04215

