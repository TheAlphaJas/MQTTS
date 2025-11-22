"""
Test script for additional loss functions (Speaker Similarity & UTMOS).

This script verifies that the loss modules can be loaded and function correctly.
"""

import torch
import argparse
import sys
import os

def test_speaker_similarity_loss():
    """Test speaker embedding similarity loss."""
    print("\n" + "="*60)
    print("Testing Speaker Embedding Similarity Loss")
    print("="*60)
    
    try:
        from modules.utmos_predictor import SpeakerEmbeddingSimilarityLoss
        
        # Initialize loss
        spk_loss = SpeakerEmbeddingSimilarityLoss()
        print("✓ SpeakerEmbeddingSimilarityLoss initialized")
        
        # Create dummy audio (1 second at 16kHz)
        batch_size = 2
        sample_rate = 16000
        duration = 1.0
        
        audio_gen = torch.randn(batch_size, int(sample_rate * duration), requires_grad=True)
        audio_ref = torch.randn(batch_size, int(sample_rate * duration))
        
        print(f"✓ Created dummy audio tensors: {audio_gen.shape}")
        
        # Move to CUDA if available
        if torch.cuda.is_available():
            audio_gen = audio_gen.cuda()
            audio_ref = audio_ref.cuda()
            print("✓ Moved tensors to CUDA")
        
        # Ensure generated audio requires gradients (simulates training scenario)
        if not audio_gen.requires_grad:
            audio_gen.requires_grad_(True)
            print("✓ Enabled gradients on generated audio")
        
        # Compute loss
        print("\nComputing speaker similarity loss...")
        loss, similarity_score = spk_loss(audio_gen, audio_ref, sample_rate)
        
        print(f"✓ Loss computed: {loss.item():.4f}")
        print(f"✓ Similarity score: {similarity_score:.4f}")
        
        # Verify loss properties
        assert loss.requires_grad, "Loss should require gradients"
        assert 0 <= loss.item() <= 2, f"Loss should be in [0, 2], got {loss.item()}"
        assert -1 <= similarity_score <= 1, f"Similarity should be in [-1, 1], got {similarity_score}"
        
        print("\n✅ Speaker Similarity Loss: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Speaker Similarity Loss: FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_utmos_loss(ckpt_path):
    """Test UTMOS MOS score loss."""
    print("\n" + "="*60)
    print("Testing UTMOS MOS Score Loss")
    print("="*60)
    
    if not ckpt_path:
        print("⚠️  Skipping UTMOS test (no checkpoint path provided)")
        print("   Use --utmos_ckpt_path to test UTMOS")
        return None
    
    if not os.path.exists(ckpt_path):
        print(f"⚠️  Skipping UTMOS test (checkpoint not found: {ckpt_path})")
        return None
    
    try:
        from modules.utmos_predictor import UTMOSPredictor
        
        # Initialize predictor
        print(f"Loading UTMOS from: {ckpt_path}")
        utmos = UTMOSPredictor(ckpt_path, device='cuda' if torch.cuda.is_available() else 'cpu')
        print("✓ UTMOSPredictor initialized")
        
        # Create dummy audio (2 seconds at 16kHz)
        batch_size = 2
        sample_rate = 16000
        duration = 2.0
        
        audio = torch.randn(batch_size, int(sample_rate * duration), requires_grad=True)
        
        print(f"✓ Created dummy audio tensor: {audio.shape}")
        
        # Move to CUDA if available
        if torch.cuda.is_available():
            audio = audio.cuda()
            print("✓ Moved tensors to CUDA")
        
        # Ensure audio requires gradients
        if not audio.requires_grad:
            audio.requires_grad_(True)
            print("✓ Enabled gradients on audio")
        
        # Predict MOS
        print("\nPredicting MOS scores...")
        mos_scores = utmos(audio, sample_rate)
        
        print(f"✓ MOS scores computed: {mos_scores}")
        print(f"  Mean MOS: {mos_scores.mean().item():.4f}")
        
        # Compute loss
        print("\nComputing MOS loss...")
        loss, mean_mos = utmos.compute_mos_loss(audio, sample_rate, target_mos=5.0)
        
        print(f"✓ Loss computed: {loss.item():.4f}")
        print(f"✓ Mean predicted MOS: {mean_mos:.4f}")
        
        # Verify loss properties
        assert loss.requires_grad, "Loss should require gradients"
        assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item()}"
        assert 1 <= mean_mos <= 5, f"MOS should be in [1, 5], got {mean_mos}"
        
        print("\n✅ UTMOS Loss: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ UTMOS Loss: FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test integration with trainer."""
    print("\n" + "="*60)
    print("Testing Integration with Trainer")
    print("="*60)
    
    try:
        # Check if modules can be imported by trainer
        from trainer_semantic import Wav2TTS
        print("✓ trainer_semantic imports successfully")
        
        # Check if new imports are present
        import trainer_semantic
        assert hasattr(trainer_semantic, 'UTMOSPredictor'), "UTMOSPredictor not imported"
        assert hasattr(trainer_semantic, 'SpeakerEmbeddingSimilarityLoss'), "SpeakerEmbeddingSimilarityLoss not imported"
        print("✓ New loss modules are imported in trainer_semantic")
        
        print("\n✅ Integration Test: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Integration Test: FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test additional loss functions")
    parser.add_argument('--utmos_ckpt_path', type=str, default=None,
                        help='Path to UTMOS checkpoint for testing')
    parser.add_argument('--skip_integration', action='store_true',
                        help='Skip integration test')
    args = parser.parse_args()
    
    print("="*60)
    print("Additional Loss Functions Test Suite")
    print("="*60)
    
    results = {}
    
    # Test 1: Speaker Similarity Loss
    results['speaker_similarity'] = test_speaker_similarity_loss()
    
    # Test 2: UTMOS Loss
    results['utmos'] = test_utmos_loss(args.utmos_ckpt_path)
    
    # Test 3: Integration
    if not args.skip_integration:
        results['integration'] = test_integration()
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, result in results.items():
        if result is True:
            status = "✅ PASSED"
        elif result is False:
            status = "❌ FAILED"
        else:
            status = "⚠️  SKIPPED"
        print(f"{test_name:20s}: {status}")
    
    # Overall result
    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)
    
    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed > 0:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
        sys.exit(1)
    else:
        print("\n✅ All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()

