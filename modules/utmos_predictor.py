"""
UTMOS predictor wrapper for MOS score optimization.
This module loads a pretrained UTMOS model and provides frozen inference for computing MOS scores.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import importlib.util


class UTMOSPredictor(nn.Module):
    """
    Wrapper for UTMOS model to predict MOS scores.
    The model weights are frozen for use as a loss function.
    """
    
    def __init__(self, ckpt_path, device='cuda'):
        """
        Initialize UTMOS predictor.
        
        Args:
            ckpt_path: Path to UTMOS checkpoint
            device: Device to load model on
        """
        super().__init__()
        self.device = device
        
        # Add UTMOS directory to sys.path for loading, then remove immediately
        utmos_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'deps', 'UTMOS22', 'strong')
        path_was_added = False
        modules_stashed = False
        stashed_modules = None
        
        try:
            # CRITICAL: Temporarily hide our 'modules' package so UTMOS can import its 'modules.py'
            # Save reference to our modules package
            if 'modules' in sys.modules:
                stashed_modules = sys.modules['modules']
                # Temporarily remove it from sys.modules
                del sys.modules['modules']
                modules_stashed = True
                print("Temporarily hid our 'modules' package to allow UTMOS import")
            
            # Temporarily add UTMOS directory to sys.path
            if utmos_dir not in sys.path:
                sys.path.insert(0, utmos_dir)
                path_was_added = True
                print(f"Temporarily added UTMOS directory to sys.path: {utmos_dir}")
            
            # Import UTMOS lightning module (now it will find UTMOS's modules.py)
            import lightning_module as utmos_lightning  # type: ignore
            
            # Load the pretrained model
            print(f"Loading UTMOS model from {ckpt_path}")
            self.utmos_model = utmos_lightning.UTMOSLightningModule.load_from_checkpoint(
                ckpt_path,
                map_location=device
            )
            
            # Freeze all parameters (model weights won't be trained)
            # But gradients can still flow through the audio input for loss computation
            self.utmos_model.eval()
            for param in self.utmos_model.parameters():
                param.requires_grad = False
            
            print("UTMOS model loaded and frozen successfully")
            
        except ModuleNotFoundError as e:
            # Handle missing dependencies
            missing_module = str(e).split("'")[1] if "'" in str(e) else "unknown"
            print("="*80)
            print("ERROR: Missing required dependency for UTMOS!")
            print(f"Missing module: {missing_module}")
            print("="*80)
            if 'fairseq' in str(e).lower():
                print("UTMOS requires 'fairseq' to be installed.")
                print("")
                print("To install fairseq, run:")
                print("  pip install fairseq")
                print("")
                print("Or for development installation:")
                print("  git clone https://github.com/pytorch/fairseq")
                print("  cd fairseq")
                print("  pip install --editable ./")
                print("")
                print("Alternatively, you can train without UTMOS loss:")
                print("  Set --utmos_weight 0.0 or omit UTMOS arguments")
                print("="*80)
            else:
                print(f"Please install the missing module: {missing_module}")
                print("Or set --utmos_weight 0.0 to disable UTMOS loss")
                print("="*80)
            self.utmos_model = None
        except ImportError as e:
            # Handle import errors (often version incompatibilities)
            error_msg = str(e)
            print("="*80)
            print("ERROR: UTMOS import failed - likely version incompatibility!")
            print(f"Error: {error_msg}")
            print("="*80)
            
            if 'AdamW' in error_msg or 'transformers' in error_msg.lower():
                print("UTMOS is using an outdated transformers API.")
                print("")
                print("The issue: UTMOS expects 'AdamW' from 'transformers',")
                print("but newer transformers versions moved AdamW to 'torch.optim'.")
                print("")
                print("SOLUTIONS:")
                print("1. Use an older transformers version compatible with UTMOS:")
                print("   pip install 'transformers<4.21'")
                print("")
                print("2. Train WITHOUT UTMOS loss (RECOMMENDED):")
                print("   Set --utmos_weight 0.0 or omit UTMOS arguments")
                print("   Speaker similarity loss alone works great!")
                print("")
                print("3. Patch UTMOS code (advanced):")
                print("   Edit deps/UTMOS22/strong/lightning_module.py")
                print("   Change: from transformers import AdamW")
                print("   To:     from torch.optim import AdamW")
                print("="*80)
            else:
                print("This appears to be a dependency version mismatch.")
                print("")
                print("UTMOS requires specific versions of:")
                print("  - transformers (likely <4.21)")
                print("  - fairseq")
                print("  - pytorch-lightning")
                print("")
                print("RECOMMENDED: Train without UTMOS loss")
                print("  Set --utmos_weight 0.0 or omit UTMOS arguments")
                print("="*80)
            self.utmos_model = None
        except Exception as e:
            print("="*80)
            print("ERROR: Failed to load UTMOS model!")
            print(f"Error: {e}")
            print(f"UTMOS directory attempted: {utmos_dir}")
            print("="*80)
            print("TROUBLESHOOTING:")
            print("1. UTMOS has a 'modules.py' file that conflicts with this project's 'modules/' directory")
            print("2. UTMOS requires 'fairseq' - install with: pip install fairseq")
            print("3. To disable UTMOS loss, set: --utmos_weight 0.0")
            print("4. Or you can continue training - UTMOS loss will be automatically disabled")
            print("="*80)
            import traceback
            traceback.print_exc()
            print("="*80)
            print("Continuing training WITHOUT UTMOS loss...")
            print("="*80)
            self.utmos_model = None
            
        finally:
            # Remove UTMOS path from sys.path first
            if path_was_added and utmos_dir in sys.path:
                sys.path.remove(utmos_dir)
            
            # CRITICAL: Remove any UTMOS 'modules' that got imported
            # Check if 'modules' is from UTMOS and remove it
            if 'modules' in sys.modules:
                try:
                    module_file = getattr(sys.modules['modules'], '__file__', None)
                    if module_file and 'UTMOS22' in str(module_file):
                        # This is UTMOS's modules.py, remove it
                        del sys.modules['modules']
                        print("Removed UTMOS's 'modules.py' from sys.modules")
                except:
                    pass
            
            # Restore our 'modules' package
            if modules_stashed and stashed_modules is not None:
                sys.modules['modules'] = stashed_modules
                print("Restored our 'modules' package")
            
            # Clean up other UTMOS-related modules (but not the main model objects)
            # We keep lightning_module and related imports cached since they're needed
            # Just remove the conflicting 'modules' entry which we already handled above
    
    def forward(self, audio, sample_rate=16000, requires_grad=True):
        """
        Predict MOS scores for audio samples.
        
        Args:
            audio: Audio tensor of shape (batch_size, samples) or (batch_size, 1, samples)
            sample_rate: Sample rate of audio
            requires_grad: If True, enable gradients for loss computation; if False, use no_grad for inference
            
        Returns:
            MOS scores of shape (batch_size,)
        """
        if self.utmos_model is None:
            # Return dummy scores if model failed to load
            batch_size = audio.shape[0]
            dummy = torch.ones(batch_size, device=audio.device) * 3.0
            if requires_grad and audio.requires_grad:
                dummy.requires_grad_(True)
            return dummy
        
        # Ensure audio is (B, samples)
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        
        batch_size = audio.shape[0]
        
        # UTMOS expects specific input format
        # Create batch dict as expected by UTMOS
        batch = {
            'wav': audio.unsqueeze(1) if audio.dim() == 2 else audio,  # (B, 1, samples)
            'judge_id': torch.zeros(batch_size, dtype=torch.long, device=audio.device),
            'domains': torch.zeros(batch_size, dtype=torch.long, device=audio.device),
            'phonemes': None,
            'phoneme_lens': None,
            'reference': None,
            'reference_lens': None
        }
        
        try:
            # Ensure model is on the same device as audio
            model_device = next(self.utmos_model.parameters()).device
            if audio.device != model_device:
                print(f"[WARNING] Moving UTMOS model from {model_device} to {audio.device}")
                self.utmos_model = self.utmos_model.to(audio.device)
            
            # Enable gradients if needed (UTMOS params are frozen, but audio input needs gradients)
            if requires_grad and audio.requires_grad:
                # Keep gradients enabled for audio input
                # UTMOS model parameters are frozen, so they won't update
                outputs = self.utmos_model(batch)
            else:
                # Use no_grad for inference (faster)
                with torch.no_grad():
                    outputs = self.utmos_model(batch)
            
            # UTMOS outputs are typically in range [-1, 1] and need to be scaled
            # to [1, 5] MOS range: output * 2 + 3
            if outputs.dim() > 1:
                outputs = outputs.mean(dim=1).squeeze(-1)
            
            # Scale to MOS range [1, 5]
            mos_scores = outputs * 2.0 + 3.0
            
            return mos_scores
        
        except Exception as e:
            print(f"Error during UTMOS forward pass: {e}")
            # Return neutral MOS score
            dummy = torch.ones(batch_size, device=audio.device) * 3.0
            if requires_grad and audio.requires_grad:
                dummy.requires_grad_(True)
            return dummy
    
    def compute_mos_loss(self, audio, sample_rate=16000, target_mos=5.0):
        """
        Compute loss that encourages higher MOS scores.
        
        Args:
            audio: Generated audio tensor (should have requires_grad=True for training)
            sample_rate: Sample rate of audio
            target_mos: Target MOS score to optimize towards (default: 5.0 for maximum quality)
            
        Returns:
            Loss value (lower MOS = higher loss), mean MOS score
        """
        # Enable gradients for loss computation
        predicted_mos = self.forward(audio, sample_rate, requires_grad=True)
        
        # We want to maximize MOS, so use MSE loss with target
        # This encourages predicted_mos to approach target_mos
        loss = F.mse_loss(predicted_mos, torch.full_like(predicted_mos, target_mos))
        
        return loss, predicted_mos.mean().item()


class SpeakerEmbeddingSimilarityLoss(nn.Module):
    """
    Computes cosine similarity loss between reference and generated audio speaker embeddings.
    """
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.inference = None
    
    def initialize_model(self, device='cuda'):
        """Lazy initialization of Pyannote embedding model."""
        if self.model is None:
            try:
                from pyannote.audio import Model
                from pyannote.audio import Inference
                
                # Load the actual model for gradient computation
                self.model = Model.from_pretrained("pyannote/embedding")
                self.model.eval()  # Set to eval mode but keep gradients enabled
                
                # Move model to correct device
                if torch.cuda.is_available() and device == 'cuda':
                    self.model = self.model.cuda()
                    print(f"Pyannote speaker embedding model initialized on CUDA")
                else:
                    print(f"Pyannote speaker embedding model initialized on CPU")
                
                # Also keep inference wrapper for non-gradient extraction if needed
                self.inference = Inference("pyannote/embedding", window="whole")
                
            except Exception as e:
                print(f"Failed to initialize Pyannote model: {e}")
                self.model = None
                self.inference = None
    
    def extract_embedding(self, audio, sample_rate=16000, requires_grad=True):
        """
        Extract speaker embedding from audio.
        
        Args:
            audio: Audio tensor of shape (batch_size, samples) or (batch_size, 1, samples)
            sample_rate: Sample rate
            requires_grad: If True, use model that preserves gradients; if False, use inference wrapper
            
        Returns:
            Speaker embeddings of shape (batch_size, 512)
        """
        if self.model is None:
            # Initialize model on the same device as the audio
            device = 'cuda' if audio.is_cuda else 'cpu'
            self.initialize_model(device=device)
        
        if self.model is None:
            # Return dummy embeddings if model failed to load
            batch_size = audio.shape[0] if audio.dim() > 1 else 1
            dummy = torch.zeros(batch_size, 512, device=audio.device)
            if requires_grad and audio.requires_grad:
                dummy.requires_grad_(True)
            return dummy
        
        # Ensure audio is (B, 1, samples) for pyannote model
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        elif audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        
        batch_size = audio.shape[0]
        
        try:
            if requires_grad and audio.requires_grad:
                # Use the actual model for gradient computation
                # Ensure model is on the same device as audio
                if audio.is_cuda and not next(self.model.parameters()).is_cuda:
                    self.model = self.model.cuda()
                elif not audio.is_cuda and next(self.model.parameters()).is_cuda:
                    self.model = self.model.cpu()
                
                # Pyannote model expects (batch, channels, samples)
                with torch.set_grad_enabled(True):
                    embeddings = self.model(audio)
                    # The model returns embeddings of shape (batch, time_steps, embed_dim)
                    # We take the mean over time steps
                    if embeddings.dim() == 3:
                        embeddings = embeddings.mean(dim=1)  # (batch, embed_dim)
                    return embeddings
            else:
                # Use inference wrapper (faster but no gradients)
                embeddings = []
                for i in range(batch_size):
                    wav = audio[i].cpu()  # (1, samples)
                    
                    # Pyannote expects dict format
                    emb = self.inference({
                        'waveform': wav,
                        'sample_rate': sample_rate
                    })
                    
                    embeddings.append(torch.FloatTensor(emb).to(audio.device))
                
                return torch.stack(embeddings, dim=0)
        
        except Exception as e:
            print(f"Error extracting embeddings: {e}")
            # Return dummy embeddings
            dummy = torch.zeros(batch_size, 512, device=audio.device)
            if requires_grad and audio.requires_grad:
                dummy.requires_grad_(True)
            return dummy
    
    def forward(self, generated_audio, reference_audio, sample_rate=16000):
        """
        Compute cosine similarity loss between generated and reference audio.
        
        Args:
            generated_audio: Generated audio tensor (should have requires_grad=True for training)
            reference_audio: Reference (ground truth) audio tensor
            sample_rate: Sample rate
            
        Returns:
            Loss value (higher similarity = lower loss), mean similarity score
        """
        # Determine if we need gradients based on input
        needs_grad = generated_audio.requires_grad
        
        # Extract embeddings
        # Only generated audio needs gradients (reference is ground truth)
        gen_emb = self.extract_embedding(generated_audio, sample_rate, requires_grad=needs_grad)
        ref_emb = self.extract_embedding(reference_audio, sample_rate, requires_grad=False)
        
        # Normalize embeddings
        gen_emb_norm = F.normalize(gen_emb, dim=-1)
        ref_emb_norm = F.normalize(ref_emb, dim=-1)
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(gen_emb_norm, ref_emb_norm, dim=-1)
        
        # Loss is 1 - similarity (we want to minimize this, i.e., maximize similarity)
        loss = (1.0 - similarity).mean()
        
        return loss, similarity.mean().item()

