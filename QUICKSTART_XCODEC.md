# Quick Start Guide: MQTTS with XCodec

## Prerequisites

```bash
# Install additional dependencies
pip install transformers>=4.40.0 accelerate>=0.20.0
```

## Quick Start (3 Steps)

### 1. Preprocess Dataset with XCodec

```bash
python quantizer/get_labels_xcodec.py \
    --input_json datasets/train.json \
    --input_wav_dir datasets/audios \
    --output_json datasets/train_q.json \
    --xcodec_model_name facebook/xcodec-base \
    --n_code_groups 4 \
    --sample_rate 16000
```

**Note**: This replaces the original quantizer training step. XCodec is pre-trained, so no training needed!

### 2. Train MQTTS Transformer

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

**Key Flag**: `--use_xcodec` enables XCodec integration

### 3. Run Inference

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

## Key Differences from Original MQTTS

| Aspect | Original MQTTS | MQTTS-XCodec |
|--------|---------------|--------------|
| Quantizer Training | Required | Not needed (pre-trained) |
| Codebook Size | 160 codes | 1024 codes |
| Codebooks | 4 groups | 8 codebooks (use first 4) |
| Semantic Features | No | Yes (from pre-trained models) |
| Preprocessing | `get_labels.py` | `get_labels_xcodec.py` |
| Training Flag | None | `--use_xcodec` |

## Troubleshooting

### Model Not Found
```bash
# Ensure transformers is up to date
pip install --upgrade transformers

# Try alternative model names
--xcodec_model_name facebook/xcodec-base
```

### Dimension Mismatch
- Ensure `--n_code_groups 4` matches in preprocessing and training
- Check that XCodec has at least 4 codebooks

### Out of Memory
- Reduce batch size: `--batch_size 100`
- Use gradient accumulation
- Use smaller XCodec model if available

## What Changed?

✅ **New**: XCodec wrapper and vocoder modules  
✅ **New**: XCodec preprocessing script  
✅ **Modified**: Vocoder supports both original and XCodec  
✅ **Modified**: Training/inference scripts support XCodec flag  
✅ **Backward Compatible**: Original MQTTS still works  

## Next Steps

1. Test with small dataset first
2. Compare audio quality vs original MQTTS
3. Fine-tune XCodec if needed (set `freeze_encoder=False`)
4. Adjust hyperparameters based on results

For detailed information, see `XCODEC_INTEGRATION.md` and `INTEGRATION_SUMMARY.md`.

