import torch
from .transformer import TextTransformer
from .model import _build_text_tower, CLIPTextCfg
from transformers import CLIPTokenizerFast


def create_text_model(device, dtype) -> TextTransformer:
    text_cfg = CLIPTextCfg(
        output_tokens=False,
        context_length=77,
        vocab_size=49408,
        width=1280,
        heads=20,
        layers=32,
        mlp_ratio=4.0,
        ls_init_value=None,
        embed_cls=False,
        no_causal_mask=False,
        pad_id=1,
        pool_type="last",
        proj_type="linear",
    )

    text_model = _build_text_tower(
        embed_dim=1280,
        text_cfg=text_cfg,
        quick_gelu=False,
        cast_dtype=torch.float32,
    )
    text_model.to(device=device, dtype=dtype)

    return text_model


def create_tokenizer():
    return CLIPTokenizerFast.from_pretrained("laion/CLIP-ViT-g-14-laion2B-s34B-b88K")
