import os
import json
import torch
import importlib
import folder_paths
import comfy.sd
import comfy.utils
import comfy.model_management as model_management
import comfy.model_detection as model_detection
from comfy.model_detection import count_blocks

try:
    gguf_ops = importlib.import_module("custom_nodes.ComfyUI-GGUF.ops")
    GGMLOps = gguf_ops.GGMLOps
    gguf_nodes = importlib.import_module("custom_nodes.ComfyUI-GGUF.nodes")
    GGUFModelPatcher = gguf_nodes.GGUFModelPatcher
    gguf_loader = importlib.import_module("custom_nodes.ComfyUI-GGUF.loader")
except Exception:
    GGMLOps = None
    GGUFModelPatcher = None
    gguf_loader = None


def get_diffusion_filenames():
    files = set()
    for folder in ["diffusion_models", "unet", "checkpoints", "unet_gguf"]:
        try:
            for f in folder_paths.get_filename_list(folder):
                files.add(f)
        except Exception:
            pass
    return sorted(list(files))


def resolve_model_path(filename):
    for folder in ["diffusion_models", "unet", "checkpoints"]:
        try:
            p = folder_paths.get_full_path(folder, filename)
            if p and os.path.exists(p):
                return p
        except Exception:
            pass
    if os.path.exists(filename):
        return filename
    raise FileNotFoundError(f"Model file not found: {filename}")


def parse_op_dtype(dtype_name):
    if dtype_name in ("default", None):
        return None
    if dtype_name == "target":
        return "target"
    return getattr(torch, dtype_name, None)


def patch_ltx_config(unet_config, sd, state_dict_keys, metadata, prefix=""):
    if f"{prefix}adaln_single.emb.timestep_embedder.linear_1.bias" not in state_dict_keys:
        return unet_config

    if metadata and "config" in metadata:
        try:
            unet_config.update(json.loads(metadata["config"]).get("transformer", {}))
        except Exception:
            pass

    if (
        f"{prefix}prompt_adaln_single.linear.weight" in state_dict_keys
        or f"{prefix}audio_prompt_adaln_single.linear.weight" in state_dict_keys
        or f"{prefix}transformer_blocks.0.prompt_scale_shift_table" in state_dict_keys
        or (
            f"{prefix}transformer_blocks.0.scale_shift_table" in state_dict_keys
            and sd[f"{prefix}transformer_blocks.0.scale_shift_table"].shape[0] == 9
        )
    ):
        unet_config.setdefault("cross_attention_adaln", True)

    if f"{prefix}caption_projection.linear.weight" in state_dict_keys:
        unet_config.setdefault("caption_proj_before_connector", True)
        unet_config.setdefault("caption_projection_first_linear", True)
    elif f"{prefix}caption_projection.linear_1.weight" in state_dict_keys:
        unet_config.setdefault("caption_proj_before_connector", False)
        unet_config.setdefault("caption_projection_first_linear", False)
    elif f"{prefix}video_embeddings_connector.transformer_1d_blocks.0.attn1.to_q.weight" in state_dict_keys:
        v_conn_dim = sd[f"{prefix}video_embeddings_connector.transformer_1d_blocks.0.attn1.to_q.weight"].shape[0]
        if v_conn_dim == unet_config.get("cross_attention_dim", 4096):
            unet_config.setdefault("caption_proj_before_connector", True)
            unet_config.setdefault("caption_projection_first_linear", False)

    if f"{prefix}video_embeddings_connector.transformer_1d_blocks.0.attn1.to_q.weight" in state_dict_keys:
        v_conn_dim = sd[f"{prefix}video_embeddings_connector.transformer_1d_blocks.0.attn1.to_q.weight"].shape[0]
        unet_config.setdefault(
            "connector_num_attention_heads",
            v_conn_dim // unet_config.get("connector_attention_head_dim", 128),
        )
        unet_config.setdefault(
            "connector_num_layers",
            count_blocks(state_dict_keys, f"{prefix}video_embeddings_connector.transformer_1d_blocks." + "{}."),
        )

    if f"{prefix}audio_embeddings_connector.transformer_1d_blocks.0.attn1.to_q.weight" in state_dict_keys:
        a_conn_dim = sd[f"{prefix}audio_embeddings_connector.transformer_1d_blocks.0.attn1.to_q.weight"].shape[0]
        if f"{prefix}audio_embeddings_connector.transformer_1d_blocks.0.attn1.to_gate_logits.bias" in state_dict_keys:
            num_heads = sd[f"{prefix}audio_embeddings_connector.transformer_1d_blocks.0.attn1.to_gate_logits.bias"].shape[0]
            unet_config.setdefault("audio_connector_num_attention_heads", num_heads)
            unet_config.setdefault("audio_connector_attention_head_dim", a_conn_dim // num_heads)
        else:
            unet_config.setdefault(
                "audio_connector_num_attention_heads",
                a_conn_dim // unet_config.get("connector_attention_head_dim", 128),
            )

    if f"{prefix}transformer_blocks.0.attn1.to_gate_logits.bias" in state_dict_keys:
        unet_config.setdefault("apply_gated_attention", True)

    if f"{prefix}video_embeddings_connector.transformer_1d_blocks.0.attn1.to_gate_logits.bias" in state_dict_keys:
        unet_config.setdefault("connector_apply_gated_attention", True)

    unet_config["use_keyframes_abs_pos_embedding"] = f"{prefix}keyframes_abs_pos_embedding" in state_dict_keys
    return unet_config


def load_diffusion_gguf(unet_path, dequant_dtype="default", patch_dtype="default", patch_on_device=False):
    if gguf_loader is None or GGMLOps is None:
        raise RuntimeError("ComfyUI-GGUF is required to load GGUF models. Please ensure ComfyUI-GGUF is installed.")

    ops = GGMLOps()
    ops.Linear.dequant_dtype = parse_op_dtype(dequant_dtype)
    ops.Linear.patch_dtype = parse_op_dtype(patch_dtype)

    sd, extra = gguf_loader.gguf_sd_loader(unet_path)
    metadata = extra.get("metadata", {})

    prefix = ""
    state_dict_keys = set(sd.keys())

    unet_config = model_detection.detect_unet_config(sd, prefix, metadata=metadata)
    if unet_config is None:
        raise RuntimeError(f"Could not detect model type of: {unet_path}")

    unet_config = patch_ltx_config(unet_config, sd, state_dict_keys, metadata, prefix)

    model_config = model_detection.model_config_from_unet_config(unet_config, sd, prefix)
    if model_config is None:
        raise RuntimeError(f"Could not initialize model config for: {unet_path}")

    load_device = model_management.get_torch_device()
    offload_device = model_management.unet_offload_device()

    unet_weight_dtype = list(model_config.supported_inference_dtypes)
    unet_dtype = model_management.unet_dtype(
        device=load_device, model_params=0, supported_dtypes=unet_weight_dtype, weight_dtype=None
    )
    manual_cast_dtype = model_management.unet_manual_cast(
        unet_dtype, load_device, model_config.supported_inference_dtypes
    )
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype, device=load_device)
    model_config.custom_operations = ops

    model = model_config.get_model(sd, prefix="")
    model_patcher = GGUFModelPatcher(model, load_device=load_device, offload_device=offload_device)
    if not model_management.is_device_cpu(offload_device):
        model.to(offload_device)
    model.load_model_weights(sd, "")
    model_patcher.patch_on_device = patch_on_device
    return model_patcher


class LTX23_GGUF_Diffusion_Loader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (get_diffusion_filenames(), {"tooltip": "Select GGUF or Safetensors diffusion model"}),
                "dequant_dtype": (["default", "target", "float32", "float16", "bfloat16"], {"default": "default"}),
                "patch_dtype": (["default", "target", "float32", "float16", "bfloat16"], {"default": "default"}),
                "patch_on_device": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_unet"
    CATEGORY = "loaders/ltx_gguf"
    TITLE = "LTX / GGUF Diffusion Model Loader"

    def load_unet(self, unet_name, dequant_dtype="default", patch_dtype="default", patch_on_device=False):
        unet_path = resolve_model_path(unet_name)
        if not unet_path.endswith(".gguf"):
            return (comfy.sd.load_diffusion_model(unet_path),)
        return (load_diffusion_gguf(unet_path, dequant_dtype, patch_dtype, patch_on_device),)
