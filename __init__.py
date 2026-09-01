from .nodes import LTX23_GGUF_Diffusion_Loader

NODE_CLASS_MAPPINGS = {
    "LTX23_GGUF_Diffusion_Loader": LTX23_GGUF_Diffusion_Loader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTX23_GGUF_Diffusion_Loader": "LTX / GGUF Diffusion Model Loader",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
