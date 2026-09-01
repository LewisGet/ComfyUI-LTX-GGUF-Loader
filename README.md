# ComfyUI-LTX-GGUF-Loader

Dedicated ComfyUI custom node for loading **LTX-Video 2.3 (LTXAV)** and other diffusion models in GGUF / Safetensors formats.

## Features

- **LTX 2.3 Architecture Auto-Detection**: Accurately infers `cross_attention_adaln`, `caption_proj_before_connector`, and connector configurations even when JSON metadata is absent from GGUF conversions.
- **Dtype Safety**: Prevents `uint8` storage types from leaking into inference dtype calculation, resolving PyTorch floating-point requirements on modules like RMSNorm and Linear layers.
- **GPU Accelerated GGUF Inference**: Seamlessly integrates with `ComfyUI-GGUF` (`GGMLOps`) for dynamic dequantization on CUDA.

## Node

- **`LTX / GGUF Diffusion Model Loader`** (`LTX23_GGUF_Diffusion_Loader`)
  - **Category**: `loaders/ltx_gguf`
  - **Inputs**:
    - `unet_name`: Select `.gguf` or `.safetensors` model file (from `models/diffusion_models/` or `models/unet/`)
    - `dequant_dtype`: `default` / `target` / `float32` / `float16` / `bfloat16`
    - `patch_dtype`: `default` / `target` / `float32` / `float16` / `bfloat16`
    - `patch_on_device`: `false` (default)
  - **Output**: `MODEL`

## Workflow Example

```
[ LTX / GGUF Diffusion Model Loader ] ──> MODEL ──> [ KSampler ]
  - 10Eros_v1.5-Q3_K_S.gguf

[ CLIPLoader (GGUF) / CLIPLoader ] ────> CLIP  ──> [ CLIPTextEncode ]
  - gemma-3-12b (type: ltxv)

[ Load VAE ] ─────────────────────────> VAE   ──> [ VAEDecode ]
  - LTX23_video_vae_bf16.safetensors
```
