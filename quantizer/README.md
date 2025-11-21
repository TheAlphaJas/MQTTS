# MQTTS Quantizer (VQ-VAE) Training

This directory contains the code for training the first stage of MQTTS: the Vector Quantized Variational Autoencoder (VQ-VAE), often referred to as the **Quantizer** or **Vocoder**.

## Overview

The Quantizer learns to compress audio into discrete codes and reconstruct it back to audio. It is essentially a HiFi-GAN generator conditioned on Vector Quantized embeddings.

We support two training modes:
1.  **Standard MQTTS**: Trains the VQ-VAE using Pyannote speaker embeddings.
2.  **Joint Training (StyleTTS2)**: Trains the VQ-VAE jointly with a Style Encoder (derived from StyleTTS2) and Pyannote embeddings. This allows the model to capture prosody and style better.

## Prerequisites

Ensure you have processed your dataset (e.g., GigaSpeech) and have the `training.txt` and `validation.txt` file lists.

## 1. Training (From Scratch)

### Standard Training
To train the original MQTTS quantizer (using pre-computed Pyannote embeddings if available, or computing them on fly):

```bash
python train.py \
    --config config.json \
    --input_wavs_dir ../datasets/audios \
    --input_training_file ../datasets/training.txt \
    --input_validation_file ../datasets/validation.txt \
    --checkpoint_path checkpoints \
    --fine_tuning False
```

### Joint Training (Recommended)
To train with the Style Encoder (StyleTTS2) + Pyannote integration. This is required if you plan to use the semantic/style-enhanced TTS transformer later.

```bash
python train_joint.py \
    --config config.json \
    --input_wavs_dir ../datasets/audios \
    --input_training_file ../datasets/training.txt \
    --input_validation_file ../datasets/validation.txt \
    --checkpoint_path checkpoints_joint \
    --style_encoder_ckpt ../ckpt/epoch_2nd_00020.pth \
    --fine_tuning False
```
*   `--style_encoder_ckpt`: Path to a pre-trained StyleTTS2 checkpoint to initialize the style encoder.

## 2. Fine-Tuning

If you have pre-trained checkpoints and want to fine-tune on a specific dataset.

1.  **Pre-extract Mel Spectrograms (Optional but Recommended for Speed)**:
    If you set `--fine_tuning True`, the dataloader expects pre-computed Mel spectrograms in `.npy` format.

    ```bash
    python preprocess_mels.py \
        --input_wavs_dir ../datasets/audios \
        --input_training_file ../datasets/training.txt \
        --input_validation_file ../datasets/validation.txt \
        --output_dir ../datasets/mels \
        --config config.json
    ```

2.  **Run Fine-Tuning**:
    ```bash
    python train_joint.py \
        --config config.json \
        --input_wavs_dir ../datasets/audios \
        --input_mels_dir ../datasets/mels \
        --checkpoint_path checkpoints_finetuned \
        --checkpoint_file checkpoints_joint/g_00100000 \
        --fine_tuning True
    ```

## 3. Checkpoints

*   **Standard**: Saved in `checkpoints/`.
*   **Joint**: Saved in `checkpoints_joint/`.

The generator checkpoint (e.g., `g_00100000`) is what you will pass to the TTS Transformer training script (`train.py` or `train_semantic.py`) as `--vocoder_ckpt_path`.

