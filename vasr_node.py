"""
ComfyUI node wrapper for AudioSR (Versatile Audio Super Resolution)

This node wraps the AudioSR model to perform audio super-resolution
within the ComfyUI workflow.
"""

import os
import sys
import random
import tempfile
import gc
from pathlib import Path

import torch
import numpy as np
import soundfile as sf
import yaml
import io

# Add parent directory to path so we can import versatile_audio_super_resolution
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

# Try to import ComfyUI's folder_paths for model directory
try:
    import folder_paths
    HAS_FOLDER_PATHS = True
except ImportError:
    HAS_FOLDER_PATHS = False

# ComfyUI interrupt handling
try:
    from comfy import model_management
    HAS_COMFY = True
except ImportError:
    HAS_COMFY = False

# VASR imports - use local module
from versatile_audio_super_resolution.audiosr.pipeline import (
    make_batch_for_super_resolution,
    seed_everything
)
from versatile_audio_super_resolution.audiosr.latent_diffusion.models.ddpm import LatentDiffusion
from versatile_audio_super_resolution.audiosr.latent_diffusion.modules.attention import set_attention_backend


def check_interrupted():
    """Check if ComfyUI processing was interrupted."""
    if HAS_COMFY:
        model_management.throw_exception_if_processing_interrupted()
    return False


def update_progress(current, total, node_prefix="AudioSR"):
    """Update ComfyUI progress bar if available."""
    if HAS_COMFY:
        try:
            state = model_management.get_progress_state()
            if state is not None:
                # Update the global progress state
                import comfy
                if hasattr(comfy, 'model_management'):
                    comfy.model_management.update_progress(
                        current / total if total > 0 else 0,
                        f"{node_prefix}: Processing {current}/{total}"
                    )
        except Exception:
            pass  # Progress update failed, continue processing


# Global model cache to avoid reloading
_model_cache = None
_model_device = None
_model_path = None


def get_vasr_model_path():
    """Get the AudioSR models directory path."""
    if HAS_FOLDER_PATHS:
        # Fallback: use models directory with AudioSR subfolder
        try:
            models_dir = folder_paths.models_dir
            audio_sr_path = str(Path(models_dir) / "AudioSR")
            print(f"[AudioSR] Checking path: {audio_sr_path}")
            return audio_sr_path
        except (AttributeError, TypeError) as e:
            print(f"[AudioSR] folder_paths.models_dir failed: {e}")
            pass

    # Final fallback: relative path
    fallback_path = str(Path(__file__).parent.parent / "models" / "AudioSR")
    print(f"[AudioSR] Using fallback path: {fallback_path}")
    return fallback_path


def load_vasr_model(ckpt_path, device="cuda"):
    """Load VASR model from checkpoint (supports .bin/.pth and .safetensors)."""
    # Get default config from VASR
    from versatile_audio_super_resolution.audiosr.utils import default_audioldm_config

    config = default_audioldm_config("basic")
    config["model"]["params"]["device"] = device

    latent_diffusion = LatentDiffusion(**config["model"]["params"])

    # Load checkpoint based on file extension
    if ckpt_path.endswith(('.safetensors', '.sft')):
        # Load safetensors format
        try:
            from safetensors.torch import load_file
            # Load to CPU first to detect dtype
            state_dict = load_file(ckpt_path, device="cpu")

            # Detect dtype from first floating-point tensor in state dict
            # Skip int64 tensors (indices, counters) - only look for actual weights
            model_dtype = torch.float32  # default
            for key, tensor in state_dict.items():
                if tensor.dtype in (torch.float32, torch.float16, torch.bfloat16,
                                    torch.float8_e4m3fn, torch.float8_e5m2):
                    model_dtype = tensor.dtype
                    print(f"[AudioSR] Detected dtype {model_dtype} from parameter '{key}'")
                    break

            # For FP8, we need to handle specially since model params don't support FP8 directly
            # Convert FP8 tensors to FP16 for model compatibility
            if model_dtype == torch.float8_e4m3fn or model_dtype == torch.float8_e5m2:
                print(f"[AudioSR] Converting FP8 weights to FP16 for model compatibility")
                state_dict = {k: v.to(torch.float16) for k, v in state_dict.items()}
                model_dtype = torch.float16

            # Convert model to detected dtype BEFORE loading state dict
            if model_dtype != torch.float32:
                print(f"[AudioSR] Converting model to {model_dtype} to match quantized weights")
                latent_diffusion = latent_diffusion.to(model_dtype)

        except ImportError:
            raise ImportError(
                "safetensors library not found. Install it with: pip install safetensors"
            )
    else:
        # Load PyTorch format (.bin, .pth, .ckpt)
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)

    latent_diffusion.load_state_dict(state_dict, strict=False)

    latent_diffusion.eval()
    latent_diffusion = latent_diffusion.to(device)

    return latent_diffusion


def generate_spectrogram_comparison(audio_before, sr_before, audio_after, sr_after=48000):
    """
    Generate a side-by-side spectrogram comparison image.

    Args:
        audio_before: Input audio (numpy array or tensor)
        sr_before: Input sample rate
        audio_after: Output audio (numpy array or tensor)
        sr_after: Output sample rate (always 48kHz for VASR)

    Returns:
        PIL Image: Side-by-side spectrogram comparison
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        from PIL import Image

        # Use non-interactive backend
        matplotlib.use('Agg')

        # Convert tensors to numpy
        if isinstance(audio_before, torch.Tensor):
            audio_before = audio_before.cpu().numpy()
        if isinstance(audio_after, torch.Tensor):
            audio_after = audio_after.cpu().numpy()

        # Ensure mono for spectrogram
        if audio_before.ndim > 1:
            audio_before = np.mean(audio_before, axis=0) if audio_before.shape[0] > 1 else audio_before[0]
        else:
            audio_before = audio_before.flatten()

        if audio_after.ndim > 1:
            audio_after = np.mean(audio_after, axis=0) if audio_after.shape[0] > 1 else audio_after[0]
        else:
            audio_after = audio_after.flatten()

        # Create figure with dark background
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        fig.patch.set_facecolor('#1a1a1a')

        # Compute spectrograms
        import librosa
        import librosa.display

        # Before spectrogram
        D_before = librosa.amplitude_to_db(np.abs(librosa.stft(audio_before)), ref=np.max)
        img1 = librosa.display.specshow(
            D_before,
            sr=sr_before,
            hop_length=512,
            x_axis='time',
            y_axis='hz',
            cmap='magma',
            ax=ax1
        )
        ax1.set_title('Before (Input)', color='white', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Frequency (Hz)', color='white', fontsize=11)
        ax1.tick_params(axis='both', colors='white', labelsize=9)
        ax1.spines['bottom'].set_color('white')
        ax1.spines['top'].set_color('white')
        ax1.spines['left'].set_color('white')
        ax1.spines['right'].set_color('white')
        ax1.xaxis.label.set_color('white')
        ax1.yaxis.label.set_color('white')

        # After spectrogram
        D_after = librosa.amplitude_to_db(np.abs(librosa.stft(audio_after)), ref=np.max)
        img2 = librosa.display.specshow(
            D_after,
            sr=sr_after,
            hop_length=512,
            x_axis='time',
            y_axis='hz',
            cmap='magma',
            ax=ax2
        )
        ax2.set_title(f'After (AudioSR {sr_after//1000}kHz)', color='white', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time (seconds)', color='white', fontsize=11)
        ax2.set_ylabel('Frequency (Hz)', color='white', fontsize=11)
        ax2.tick_params(axis='both', colors='white', labelsize=9)
        ax2.spines['bottom'].set_color('white')
        ax2.spines['top'].set_color('white')
        ax2.spines['left'].set_color('white')
        ax2.spines['right'].set_color('white')
        ax2.xaxis.label.set_color('white')
        ax2.yaxis.label.set_color('white')

        # Add colorbar
        cbar = fig.colorbar(img2, ax=[ax1, ax2], fraction=0.02, pad=0.04)
        cbar.set_label('dB', color='white', fontsize=10)
        cbar.ax.yaxis.set_tick_params(color='white', labelsize=9)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

        # Add main title
        fig.suptitle('Audio Super Resolution Spectrogram Comparison',
                     color='white', fontsize=16, fontweight='bold')

        # Use constrained_layout instead of tight_layout to avoid warnings
        try:
            fig.set_layout_engine('constrained')
        except (AttributeError, TypeError):
            # Fallback for older matplotlib versions
            plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor='#1a1a1a', dpi=100)
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)

        return img

    except ImportError as e:
        print(f"[AudioSR] Warning: Could not import matplotlib for spectrogram: {e}")
        return None
    except Exception as e:
        print(f"[AudioSR] Warning: Could not generate spectrogram: {e}")
        return None


class VASRNode:
    """
    Versatile Audio Super Resolution node for ComfyUI.

    Upscales audio to 48kHz using the AudioSR latent diffusion model.
    """

    DESCRIPTION = "Upscale audio to 48kHz using Versatile Audio Super Resolution (AudioSR)"

    @classmethod
    def INPUT_TYPES(cls):
        # Get available model files from VASR models directory
        model_dir = get_vasr_model_path()
        model_files = []

        print(f"[AudioSR] Looking for models in: {model_dir}")

        if os.path.exists(model_dir):
            print(f"[AudioSR] Model directory exists!")
            for f in os.listdir(model_dir):
                if f.endswith(('.bin', '.pth', '.ckpt', '.safetensors')):
                    model_files.append(f)
                    print(f"[AudioSR] Found model: {f}")
            print(f"[AudioSR] Total models found: {len(model_files)}")
        else:
            print(f"[AudioSR] Model directory does NOT exist!")

        # Add default option if no models found
        if not model_files:
            model_files = ["basic (download required)", "speech (download required)"]
            print(f"[AudioSR] No models found, using default options")

        return {
            "required": {
                "audio": ("AUDIO", {}),
                "ddim_steps": ("INT", {
                    "default": 50,
                    "min": 10,
                    "max": 500,
                    "step": 1,
                    "tooltip": "Number of denoising steps (higher = better quality, slower)"
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 3.5,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.1,
                    "tooltip": "Classifier-free guidance scale (higher = more faithful to input)"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFF,
                    "tooltip": "Random seed (0 = random)"
                }),
            },
            "optional": {
                "model": (model_files, {
                    "default": model_files[0] if model_files else "basic (download required)",
                    "tooltip": "Model checkpoint file (place in ComfyUI/models/AudioSR/)"
                }),
                "chunk_size": ("FLOAT", {
                    "default": 15.0,
                    "min": 2.56,
                    "max": 30.0,
                    "step": 0.01,
                    "tooltip": "Chunk duration in seconds for processing long audio (default: 15s from main repo)"
                }),
                "overlap": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Overlap duration in seconds between chunks. Helps smooth transitions between audio chunks. Higher values = smoother but slower processing. (0.0 = no overlap, 2.0-3.0 recommended for long audio)"
                }),
                "unload_model": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Unload model from memory after generation (frees VRAM, but slower next run)"
                }),
                "show_spectrogram": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Generate before/after spectrogram comparison image"
                }),
                "attention_backend": (["sdpa", "eager"], {
                    "default": "sdpa",
                    "tooltip": "Attention computation backend: sdpa (PyTorch native, fastest & recommended), eager (einsum-based, most compatible)"
                }),
            }
        }

    RETURN_TYPES = ("AUDIO", "IMAGE")
    RETURN_NAMES = ("audio", "spectrogram")
    OUTPUT_NODE = False
    FUNCTION = "upscale_audio"
    CATEGORY = "audio"

    def upscale_audio(
        self,
        audio: tuple,
        ddim_steps: int,
        guidance_scale: float,
        seed: int,
        model: str = "basic (download required)",
        chunk_size: float = 5.12,
        overlap: float = 0.04,
        unload_model: bool = False,
        show_spectrogram: bool = True,
        attention_backend: str = "sdpa"
    ):
        """
        Main processing function for audio super-resolution.

        Args:
            audio: ComfyUI audio tuple (audio_tensor, sample_rate)
            ddim_steps: Number of inference steps (denoising steps)
            guidance_scale: CFG scale (classifier-free guidance)
            seed: Random seed (0 = random)
            model: Model checkpoint file name
            chunk_size: Chunk duration in seconds (default: 15s from main repo)
            overlap: Overlap duration in seconds (default: 2s from main repo)
            unload_model: Unload model from GPU memory after generation
            show_spectrogram: Generate before/after spectrogram comparison
            attention_backend: Attention backend to use (sdpa, eager, flash_attention, sageattn)

        Returns:
            tuple: (audio, spectrogram) - ComfyUI audio format at 48kHz and optional spectrogram image
        """
        global _model_cache, _model_device, _model_path

        # Unpack ComfyUI audio format: can be dict {'waveform': tensor, 'sample_rate': int}
        # or tuple (waveform, sample_rate) for backwards compatibility
        if isinstance(audio, dict):
            audio_waveform = audio['waveform']
            sr = audio['sample_rate']
        else:
            audio_waveform, sr = audio

        # Validate audio input
        if isinstance(audio_waveform, str):
            raise ValueError(
                f"Audio input is a filename string ('{audio_waveform}'), not audio data. "
                f"Please connect the Load Audio node output to the AudioSR audio input."
            )

        # Store original sample rate for logging
        original_sr = sr
        needs_resample = sr != 48000

        # Convert to numpy if tensor
        if isinstance(audio_waveform, torch.Tensor):
            audio_waveform = audio_waveform.cpu().numpy()
        elif not isinstance(audio_waveform, np.ndarray):
            raise TypeError(
                f"Audio waveform must be a torch.Tensor or numpy array, got {type(audio_waveform)}"
            )

        # Ensure 2D shape [channels, samples]
        if audio_waveform.ndim == 1:
            audio_waveform = audio_waveform[np.newaxis, :]
        elif audio_waveform.ndim == 3:
            audio_waveform = audio_waveform.squeeze(0)

        # Store original audio for spectrogram comparison (before resampling)
        original_audio_for_spec = audio_waveform.copy()
        original_sr_for_spec = sr

        # Resample to 48000 Hz if needed (VASR expects 48kHz input)
        if needs_resample:
            import librosa
            # Process each channel independently
            resampled_channels = []
            for ch in range(audio_waveform.shape[0]):
                channel = audio_waveform[ch]
                resampled = librosa.resample(channel, orig_sr=sr, target_sr=48000)
                resampled_channels.append(resampled)
            audio_waveform = np.stack(resampled_channels, axis=0)
            sr = 48000

        # Calculate duration
        num_samples = audio_waveform.shape[1]
        duration_sec = num_samples / sr

        print(f"[AudioSR] Processing audio: {duration_sec:.2f}s at {sr}Hz, {'stereo' if audio_waveform.shape[0] > 1 else 'mono'}")

        # Set seed (0 = random)
        if seed == 0:
            seed = random.randint(0, 2**32 - 1)
        seed_everything(seed)

        # Get device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Handle model selection
        if "download required" in model:
            raise ValueError(
                f"Model not found. Please download the AudioSR model and place it in:\n"
                f"{get_vasr_model_path()}\n\n"
                f"Download from:\n"
                f"Basic: https://huggingface.co/haoheliu/audiosr_basic\n"
                f"Speech: https://huggingface.co/haoheliu/audiosr_speech"
            )

        # Load or reuse model
        ckpt_path = os.path.join(get_vasr_model_path(), model)
        if not os.path.exists(ckpt_path):
            raise ValueError(f"Model file not found: {ckpt_path}")

        if _model_cache is None or _model_device != device or _model_path != ckpt_path:
            print(f"[AudioSR] Loading model '{model}' on {device}...")
            _model_cache = load_vasr_model(ckpt_path, device)
            _model_device = device
            _model_path = ckpt_path
        else:
            print(f"[AudioSR] Using cached model on {device}")

        # Get ComfyUI model_management for VRAM operations
        mm = model_management if HAS_COMFY else None

        # Set attention backend before processing
        set_attention_backend(attention_backend)
        print(f"[AudioSR] Settings: steps={ddim_steps}, guidance={guidance_scale}, seed={seed}, attention={attention_backend}")

        # Wrap processing in interrupt exception handler
        try:
            with torch.no_grad():
                # Process each channel for stereo support
                is_stereo = audio_waveform.shape[0] > 1
                channels = [audio_waveform[0], audio_waveform[1]] if is_stereo else [audio_waveform[0]]

                processed_channels = []
                total_chunks = 0

                # Calculate total number of chunks for progress tracking
                if duration_sec > 10.24:
                    chunk_samples = int(chunk_size * sr)
                    overlap_samples = int(overlap * sr)  # overlap is now in seconds
                    num_chunks_per_channel = 0
                    start = 0
                    while start < num_samples:
                        num_chunks_per_channel += 1
                        start += chunk_samples - overlap_samples
                    total_chunks = num_chunks_per_channel * len(channels)
                    print(f"[AudioSR] Long audio detected: {duration_sec:.2f}s, splitting into {num_chunks_per_channel} chunks per channel ({total_chunks} total)")
                else:
                    total_chunks = len(channels)

                current_chunk = 0

                for ch_idx, channel in enumerate(channels):
                    channel_name = "Left" if ch_idx == 0 else "Right"

                    # Check for interrupt before processing channel
                    check_interrupted()

                    # For short audio (<= 10.24s), process directly
                    if duration_sec <= 10.24:
                        # Prepare batch for single chunk
                        batch, _ = make_batch_for_super_resolution(
                            None,
                            waveform=torch.from_numpy(channel).unsqueeze(0).numpy()
                        )

                        output_waveform = _model_cache.generate_batch(
                            batch,
                            unconditional_guidance_scale=guidance_scale,
                            ddim_steps=ddim_steps,
                            duration=duration_sec
                        )

                        if isinstance(output_waveform, np.ndarray):
                            output_waveform = torch.from_numpy(output_waveform)

                        # Ensure output is at least 2D [1, samples]
                        if output_waveform.ndim == 1:
                            output_waveform = output_waveform.unsqueeze(0)

                        # Trim to correct length (output is at 48kHz)
                        target_samples = int(duration_sec * 48000)
                        if output_waveform.shape[-1] > target_samples:
                            output_waveform = output_waveform[..., :target_samples]

                        processed_channels.append(output_waveform)

                        # Report progress for this channel
                        current_chunk += 1
                        update_progress(current_chunk, total_chunks)

                    else:
                        # Process long audio in chunks
                        chunk_samples = int(chunk_size * sr)
                        overlap_samples = int(overlap * sr)  # overlap is now in seconds
                        output_chunk_samples = int(chunk_size * 48000)
                        output_overlap_samples = int(overlap * 48000)  # overlap at 48kHz output

                        chunks = []
                        start = 0
                        while start < num_samples:
                            end = min(start + chunk_samples, num_samples)
                            chunk = channel[start:end]
                            chunks.append((chunk, start, end))
                            start += chunk_samples - overlap_samples

                        # Process chunks with overlap
                        reconstructed = np.zeros(int(duration_sec * 48000))
                        weight_sum = np.zeros(int(duration_sec * 48000))

                        for i, (chunk, orig_start, orig_end) in enumerate(chunks):
                            # Check for interrupt before each chunk
                            check_interrupted()

                            print(f"[AudioSR] {channel_name} chunk {i+1}/{len(chunks)} ({current_chunk+1}/{total_chunks} overall)")

                            chunk_duration = (orig_end - orig_start) / sr
                            batch, _ = make_batch_for_super_resolution(
                                None,
                                waveform=torch.from_numpy(chunk).unsqueeze(0).numpy()
                            )

                            output_chunk = _model_cache.generate_batch(
                                batch,
                                unconditional_guidance_scale=guidance_scale,
                                ddim_steps=ddim_steps,
                                duration=chunk_duration
                            )

                            if isinstance(output_chunk, np.ndarray):
                                output_chunk = torch.from_numpy(output_chunk)

                            # Ensure output is at least 2D before squeeze
                            if output_chunk.ndim == 1:
                                output_chunk = output_chunk.unsqueeze(0)
                            elif output_chunk.ndim > 2:
                                output_chunk = output_chunk.squeeze()

                            # Convert to numpy and ensure 1D array
                            output_chunk_np = output_chunk.squeeze(0).cpu().numpy()
                            if output_chunk_np.ndim != 1:
                                # If still multi-dimensional, flatten
                                output_chunk_np = output_chunk_np.flatten()

                            orig_output_len = output_chunk_np.shape[0]

                            # Apply fade window for overlap regions
                            if overlap > 0 and i < len(chunks) - 1:
                                fade_len = min(output_overlap_samples, orig_output_len)
                                fade_out = np.linspace(1., 0., fade_len)
                                output_chunk_np[:fade_len] *= fade_out

                            if overlap > 0 and i > 0:
                                fade_len = min(output_overlap_samples, orig_output_len)
                                fade_in = np.linspace(0., 1., fade_len)
                                output_chunk_np[:fade_len] *= fade_in

                            # Calculate output position (time-scaled)
                            out_start = int(orig_start / sr * 48000)
                            out_end = min(out_start + orig_output_len, reconstructed.shape[0])

                            # Truncate output_chunk if needed to match slice size
                            slice_len = out_end - out_start
                            if output_chunk_np.shape[0] > slice_len:
                                output_chunk_np = output_chunk_np[:slice_len]
                            elif output_chunk_np.shape[0] < slice_len:
                                # Pad with zeros if output is too short
                                padded = np.zeros(slice_len)
                                padded[:output_chunk_np.shape[0]] = output_chunk_np
                                output_chunk_np = padded

                            # Add to reconstructed array
                            reconstructed[out_start:out_end] += output_chunk_np
                            weight_sum[out_start:out_end] += 1.0

                            # Report progress after each chunk
                            current_chunk += 1
                            update_progress(current_chunk, total_chunks)

                        # Normalize by weight sum
                        weight_sum[weight_sum == 0] = 1.0
                        processed_channels.append(torch.from_numpy(reconstructed / weight_sum).unsqueeze(0))

                # Combine channels
                if is_stereo:
                    output_waveform = torch.cat(processed_channels, dim=0)
                else:
                    output_waveform = processed_channels[0]

                # CRITICAL: Ensure output is always 2D [channels, samples] for ComfyUI
                if output_waveform.ndim == 1:
                    output_waveform = output_waveform.unsqueeze(0)
                    print("[AudioSR] Warning: output was 1D, fixed to 2D [1, samples]")
                elif output_waveform.ndim > 2:
                    # Flatten extra dimensions, keeping only last one
                    output_waveform = output_waveform.reshape(-1, output_waveform.shape[-1])
                    print(f"[AudioSR] Warning: output was {output_waveform.ndim}D, reshaped to 2D [channels, samples]")

                # Final validation - must be [channels, samples]
                assert output_waveform.ndim == 2, f"Output waveform must be 2D [channels, samples], got shape {output_waveform.shape}"

                # CRITICAL: Add batch dimension for ComfyUI format [batch, channels, samples]
                output_waveform = output_waveform.unsqueeze(0)
                print(f"[AudioSR] Processing complete! Output: {output_waveform.shape[-1]/48000:.2f}s at 48kHz, shape: {output_waveform.shape}")

                # Unload model from memory if requested
                if unload_model:
                    print("[AudioSR] Unloading model from VRAM...")
                    if _model_cache is not None:
                        del _model_cache
                        _model_cache = None
                        _model_device = None
                        _model_path = None
                    gc.collect()
                    if mm is not None:
                        mm.soft_empty_cache()
                    elif torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print("[AudioSR] Model unloaded, VRAM freed")

                # Generate spectrogram comparison if requested
                spectrogram_image = None
                if show_spectrogram:
                    # Remove batch dimension for spectrogram generation (expects [channels, samples])
                    waveform_for_spec = output_waveform.squeeze(0)
                    spectrogram_image = generate_spectrogram_comparison(
                        original_audio_for_spec,
                        original_sr_for_spec,
                        waveform_for_spec,
                        48000
                    )

                # Convert PIL image to tensor for ComfyUI
                if spectrogram_image is not None:
                    # Ensure the image is in RGB mode and proper format
                    spectrogram_image = spectrogram_image.convert('RGB')

                    # Convert to numpy array first, then to tensor
                    img_array = np.array(spectrogram_image).astype(np.float32) / 255.0

                    # Convert to tensor: ComfyUI expects [B, H, W, C]
                    spectrogram_tensor = torch.from_numpy(img_array)  # [H, W, C]
                    spectrogram_tensor = spectrogram_tensor.unsqueeze(0)  # [1, H, W, C]

                    # Ensure proper format [1, H, W, C] with values in [0, 1]
                    spectrogram_tensor = torch.clamp(spectrogram_tensor, 0.0, 1.0)
                else:
                    # Return empty tensor if no spectrogram
                    spectrogram_tensor = torch.zeros((1, 256, 256, 3))

                # ComfyUI 0.8.0+ format: dict with waveform and sample_rate
                audio_output = {"waveform": output_waveform, "sample_rate": 48000}
                return (audio_output, spectrogram_tensor)

        except Exception as e:
            # Check if this was an interrupt exception
            if HAS_COMFY:
                try:
                    from comfy.model_management import InterruptProcessingException
                    if isinstance(e, InterruptProcessingException):
                        # Cleanup on interrupt
                        if unload_model and _model_cache is not None:
                            del _model_cache
                            _model_cache = None
                            _model_device = None
                            _model_path = None
                        gc.collect()
                        if mm is not None:
                            mm.soft_empty_cache()
                        raise  # Re-raise the interrupt exception
                except ImportError:
                    pass
            raise

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Force re-execution when seed or model changes
        seed = kwargs.get("seed", 0)
        model = kwargs.get("model", "")
        return float(f"{hash(model + str(seed))}")


# Register AudioSR models directory with ComfyUI
# This makes the models folder appear in ComfyUI's model management
def register_folder_paths():
    """Register the AudioSR models folder with ComfyUI."""
    try:
        import folder_paths
        # Add AudioSR as a model directory - this creates models/AudioSR/
        folder_paths.add_model_folder_path("audiosr", "AudioSR")
    except Exception:
        # folder_paths not available (older ComfyUI versions)
        pass


# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "AudioSR": VASRNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioSR": "AudioSR"
}

# Register folder paths on import
register_folder_paths()
