<h1 align="center">ComfyUI AudioSR (Versatile Audio Super Resolution)</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Audio-Upscaling-orange.svg">
  <img src="https://img.shields.io/badge/Output-48kHz-green.svg">
</p>

<div align="center">
    <a href="https://arxiv.org/abs/2309.07314"><img src="https://img.shields.io/static/v1?label=Paper&message=AudioSR&color=blue"></a> &ensp;
    <a href="https://github.com/haoheliu/versatile_audio_super_resolution"><img src="https://img.shields.io/static/v1?label=Original&message=Repository&color=orange"></a> &ensp;
    <a href="https://audioldm.github.io/audiosr/"><img src="https://img.shields.io/static/v1?label=Project&message=Page&color=purple"></a> &ensp;
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg"></a> &ensp;
    <a href="https://huggingface.co/drbaph/AudioSR/tree/main/AudioSR"><img src="https://img.shields.io/static/v1?label=Models&message=HuggingFace&color=yellow"></a> &ensp;
    <a href="https://github.com/comfyanonymous/ComfyUI"><img src="https://img.shields.io/badge/ComfyUI-Native-green.svg"></a> &ensp;
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
</div>

<br>

Native ComfyUI node for **AudioSR (Versatile Audio Super Resolution)** - Upscale any audio to 48kHz using state-of-the-art latent diffusion.

**Based on the original AudioSR implementation by [Haohe Liu](https://github.com/haoheliu) et al.**

## 🎯 Key Features

- **🎧 Audio Super Resolution**: Upsample low-quality audio to 48kHz with enhanced high frequencies
- **🎛️ Native ComfyUI Integration**: Works seamlessly with Load Audio, Preview Audio, and Save Audio nodes
- **📊 Built-in Spectrogram Visualization**: Before/after comparison with time and frequency axes
- **🔄 Automatic Sample Rate Handling**: Accepts any input sample rate (8kHz - 48kHz)
- **🧩 Stereo Support**: Processes both mono and stereo audio with independent channel handling
- **📏 Long Audio Support**: Smart chunking with overlap for unlimited audio length
- **⚡ Model Caching**: Model stays in memory for fast subsequent generations
- **🎛️ VRAM Management**: Optional model unloading to free GPU memory between runs
- **⏸️ Interruptible**: Cancel processing mid-run through ComfyUI's interrupt button
- **📈 Progress Reporting**: Real-time progress bar shows chunk processing status

---

## Requirements

All Python dependencies are installed automatically. No external tools required.

**Minimum: 6GB VRAM, 12GB RAM recommended**

---

<img width="1605" height="913" alt="image" src="https://github.com/user-attachments/assets/c84d723b-bfde-4998-926b-0bc38bea4b26" />

<br>

<img width="1200" height="800" alt="ComfyUI_temp_bildo_00002_" src="https://github.com/user-attachments/assets/786f3b0a-706c-4852-b8c0-1ab5786806ce" />

## Audio Examples


| Original | AudioSR |
|----------|----------|
| [speech_up_4.wav](https://huggingface.co/drbaph/AudioSR/resolve/main/samples/speech_up_4.wav) | [speech_audiosr_4.wav](https://huggingface.co/drbaph/AudioSR/resolve/main/samples/speech_audiosr_4.wav) |
| [event_up_2.wav](https://huggingface.co/drbaph/AudioSR/resolve/main/samples/event_up_2.wav) | [event_audiosr_2.wav](https://huggingface.co/drbaph/AudioSR/resolve/main/samples/event_audiosr_2.wav) |


---

## 📦 Installation

### Method 1: ComfyUI Manager (Recommended)

1. Open ComfyUI Manager
2. Search for **"AudioSR"**
3. Click **Install**
4. Restart ComfyUI

That's it! All dependencies are installed automatically.

### Method 2: Manual Installation

**Standard Python:**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Saganaki22/ComfyUI-AudioSR.git
cd ComfyUI-AudioSR
pip install -r requirements.txt
```

**ComfyUI Portable (Windows with embedded Python):**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Saganaki22/ComfyUI-AudioSR
cd ComfyUI-AudioSR
..\python_embeded\python.exe -s -m pip install -r requirements.txt
```

### 📥 Download Models

**Important:** Models must be placed in `ComfyUI/models/AudioSR/`

**Download from HuggingFace:**
https://huggingface.co/drbaph/AudioSR/tree/main/AudioSR

Download one or both models and place them in your ComfyUI models directory:

**Required Folder Structure:**
```
ComfyUI/
└── models/
    └── AudioSR/
        ├── audiosr_basic_fp32.safetensors (for general audio)
        └── audiosr_speech_fp32.safetensors (for voice content)
```

**Available Models (FP32):**
- `audiosr_basic_fp32.safetensors` - General purpose (music, sound effects, etc.)
- `audiosr_speech_fp32.safetensors` - Optimized for voice/speech

---

## 💻 VRAM Requirements

| Configuration | VRAM Usage | Recommended For |
|---------------|------------|-----------------|
| Standard | ~6GB | RTX 3060+ (6GB+) |
| With unload_model enabled | ~0.5GB (when idle) | Systems with limited VRAM |

**Minimum: 6GB VRAM required, 8GB+ recommended**

---

## 🎚️ AudioSR Node

<details>
<summary><b>📖 Click to expand: Overview & Parameters</b></summary>

### Overview

Upscale audio to 48kHz using the AudioSR latent diffusion model. The model analyzes low-quality audio and generates enhanced high-frequency details for a cleaner, fuller sound.

**What it does:**
- Resamples audio to 48kHz (if needed)
- Enhances high frequencies and adds clarity
- Reduces artifacts from low-bitrate compression
- Works on any audio: music, speech, sound effects

**Use Cases:**
- Upsample old/low-quality audio recordings
- Enhance compressed audio (MP3, low-bitrate streams)
- Improve audio for video production
- Restore archived audio content

### Parameters

#### Required Inputs

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| **audio** | AUDIO | - | - | Audio input from Load Audio node |
| **ddim_steps** | INT | 10-500 | 50 | Number of denoising steps (higher = better quality, slower) |
| **guidance_scale** | FLOAT | 1.0-20.0 | 3.5 | CFG scale - higher = more faithful to input |
| **seed** | INT | 0-4.29B | 0 | Random seed (0 = random) |

#### Optional Inputs

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| **model** | COMBO | - | basic | Model file from `ComfyUI/models/AudioSR/` (supports .bin, .safetensors) |
| **chunk_size** | FLOAT | 2.56-30.0 | 15.0 | Chunk duration in seconds (for audio >10.24s) |
| **overlap** | FLOAT | 0.0-5.0 | 0.0 | Overlap in seconds between chunks (helps smooth transitions, 2.0-3.0 recommended for long audio) |
| **unload_model** | BOOLEAN | - | False | Free VRAM after generation (slower next run) |
| **show_spectrogram** | BOOLEAN | - | True | Generate before/after spectrogram image |
| **attention_backend** | COMBO | - | sdpa | Attention backend: sdpa (PyTorch native, fastest), eager (einsum-based) |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| **audio** | AUDIO | Upscaled audio at 48kHz (connect to Preview/Save) |
| **spectrogram** | IMAGE | Before/after spectrogram comparison (optional) |

</details>

---

## 🎨 Workflow Examples

<details>
<summary><b>📖 Click to expand: Workflow Examples</b></summary>

### Basic Audio Upscaling

```
Load Audio → AudioSR → Preview Audio / Save Audio
```

1. Add **Load Audio** node and select your audio file
2. Add **AudioSR** node
3. Connect audio output to AudioSR input
4. Set `ddim_steps: 50` (default)
5. Set `guidance_scale: 3.5` (default)
6. Connect AudioSR audio output to **Preview Audio** or **Save Audio**
7. Queue and generate!

### High Quality Upscaling

```
Settings:
- ddim_steps: 100
- guidance_scale: 5.0
- model: speech (for voice content)
- show_spectrogram: True
```

### Low VRAM Mode

```
Settings:
- unload_model: True
- ddim_steps: 50 (default)
```

This frees VRAM after each generation but makes subsequent runs slower.

### Long Audio Processing

For audio longer than ~10 seconds, the node automatically:
- Splits audio into chunks
- Processes each chunk
- Crossfades overlap regions
- Stitches into seamless output

**Defaults from main repo** (recommended):
```
- chunk_size: 15.0 (seconds per chunk)
- overlap: 2.0 (seconds overlap between chunks)
```

For faster processing with more VRAM:
```
- chunk_size: 20.0-30.0 (fewer chunks = faster)
- overlap: 2.0-3.0 (smoother transitions)
```

</details>

---

## ⚙️ Parameter Guide

<details>
<summary><b>📖 Click to expand: Detailed Parameter Guide</b></summary>

### ddim_steps (10-500)

Number of denoising steps during generation.

| Value | Quality | Speed | Use Case |
|-------|---------|-------|----------|
| 10-30 | Lower | Fast | Quick previews |
| 50 | Good | Medium | **Default recommendation** |
| 100 | Better | Slow | High-quality output |
| 200+ | Best | Very Slow | Maximum quality |

### guidance_scale (1.0-20.0)

Classifier-free guidance scale. Controls how closely the output follows the input.

| Value | Effect |
|-------|--------|
| 1.0-2.0 | More creative/variant |
| 3.0-4.0 | **Balanced** (default) |
| 5.0-8.0 | More faithful to input |
| 10.0+ | Very conservative (may sound artificial) |

### chunk_size (2.56-30.0)

Chunk duration in seconds for long audio processing.

| Value | Effect |
|-------|--------|
| 2.56-5.0 | More chunks, slower, less memory per chunk |
| 15.0 | **Default** (from main repo, balanced) |
| 20.0-30.0 | Fewer chunks, faster, more memory per chunk |

### overlap (0.0-5.0)

Overlap duration in seconds between chunks.

| Value | Effect |
|-------|--------|
| 0.0 | No overlap (may have seams) |
| 2.0 | **Default** (from main repo) |
| 3.0-5.0 | Smoother stitching, slower processing |

</details>

---

## 🐛 Troubleshooting

<details>
<summary><b>📖 Click to expand: Troubleshooting</b></summary>

### Model Not Found Error

**Symptom**: "Model not found. Please download the AudioSR model..."

**Solution**:
1. Download model from: https://huggingface.co/drbaph/AudioSR/tree/main/AudioSR
2. Place in `ComfyUI/models/AudioSR/`
3. Restart ComfyUI

### CUDA Out of Memory

**Symptom**: RuntimeError: CUDA out of memory

**Solutions**:
1. Enable `unload_model: True`
2. Close other GPU-intensive applications
3. Reduce `chunk_size` (try 2.56)
4. Use a smaller audio file

### Poor Audio Quality

**Symptom**: Output sounds distorted or artificial

**Solutions**:
1. Lower `guidance_scale` (try 2.5-3.0)
2. Reduce `ddim_steps` (too high can sound artificial)
3. Try the other model variant (basic ↔ speech)
4. Ensure input audio is reasonably clean

### No Output Audio

**Symptom**: Node completes but no audio is produced

**Solutions**:
1. Check ComfyUI console for error messages
2. Verify audio input is properly connected
3. Ensure output is connected to Preview/Save Audio node
4. Try a different audio file

### Spectrogram Not Showing

**Symptom**: Spectrogram output is empty or shows noise

**Solutions**:
1. Ensure `show_spectrogram: True`
2. Install matplotlib: `pip install matplotlib`
3. Check ComfyUI console for errors

### Slow Processing

**Symptom**: Node takes very long to process

**Solutions**:
1. Disable `unload_model` (keeps model cached)
2. Increase `chunk_size` (fewer chunks = faster)
3. Ensure GPU is being used (not CPU)
4. Try `ddim_steps: 30` for faster processing

</details>

---

## 📊 Spectrogram Visualization

<details>
<summary><b>📖 Click to expand: Spectrogram Details</b></summary>

The node generates a side-by-side spectrogram comparison when `show_spectrogram: True`:

**Top panel**: Input audio (before) - Shows limited high frequencies
**Bottom panel**: Output audio (after) - Shows enhanced frequency content

The spectrogram uses the **magma** colormap:
- **Purple/Black**: Low energy (silence/quiet)
- **Red/Orange**: Medium energy
- **Yellow**: High energy (loud frequencies)

**Axes**:
- **X-axis**: Time in seconds
- **Y-axis**: Frequency in Hz (0-24kHz visible range)

</details>

---

## 🔗 Links

- [AudioSR Paper (arXiv)](https://arxiv.org/abs/2309.07314)
- [Project Page](https://audioldm.github.io/audiosr/)
- [Original Repository](https://github.com/haoheliu/versatile_audio_super_resolution)
- [Models (HuggingFace)](https://huggingface.co/datasets/drbaph/AudioSR/tree/main/AudioSR)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

---

## 💡 Tips for Best Results

- **Input Quality**: The model can't create what isn't there - extremely low-quality audio may still sound artificial
- **Guidance Scale**: Start at 3.5 and adjust based on results
- **Steps**: 50 steps is usually sufficient; 100+ for critical applications
- **Speech Audio**: Use the `audiosr_speech_fp32.safetensors` model for voice content
- **Music/General**: Use the `audiosr_basic_fp32.safetensors` model for music and sound effects
- **Long Audio**: Let the auto-chunking handle files >10 seconds
- **VRAM**: Enable `unload_model` if you need GPU memory for other tasks

---

## 📚 Credits & License

**Original Research**: [AudioSR: Versatile Audio Super-Resolution](https://arxiv.org/abs/2309.07314) by [Haohe Liu](https://github.com/haoheliu) et al.

**Original Implementation**: [versatile_audio_super_resolution](https://github.com/haoheliu/versatile_audio_super_resolution) by Haohe Liu

**ComfyUI Integration**: This custom node implementation

**License**: MIT (same as original AudioSR project)

---

## 📝 Changelog

### Version 1.0.1

- ✅ Native ComfyUI AUDIO type support
- ✅ Automatic sample rate conversion (any input rate → 48kHz)
- ✅ Stereo audio processing
- ✅ Longer audio support with smart chunking
- ✅ Before/after spectrogram visualization
- ✅ Progress reporting and interrupt support
- ✅ Model caching and optional VRAM unloading
- ✅ Time and frequency axes on spectrograms
