import argparse
import pathlib
import torch
import sd_mecha
from time_embed_fix_utils import distill_time_embed, model_config


def main_cli():
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
        "fp64": torch.float64,
    }

    parser = argparse.ArgumentParser(
        description="Aggregate & train time‑embedding layers from multiple Stable‑Diffusion models (memory‑safe)."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="PATH",
        required=True,
        help="One or more .safetensors checkpoints to aggregate.",
    )
    parser.add_argument(
        "--fallback_model",
        metavar="PATH",
        required=True,
        help="The .safetensors checkpoint to fix.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help='Torch device to run on (default: "cuda:0").',
    )
    parser.add_argument(
        "--dtype",
        choices=list(dtype_map.keys()),
        default="float32",
        help="Computation datatype (default: float32).",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=4096,
        help="Number of optimization steps (default: 4096).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4).",
    )
    parser.add_argument(
        "--max_timestep",
        type=int,
        default=2048,
        help="Maximum timestep to optimize (default: 2048)"
    )
    parser.add_argument(
        "--out",
        metavar="PATH",
        default=None,
        help="Where to write the trained embeddings (default = original hard‑coded path).",
    )

    args = parser.parse_args()
    main(
        paths=[str(pathlib.Path(p).absolute()) for p in args.models],
        fallback_model=args.fallback_model,
        iters=args.iters,
        max_timestep=args.max_timestep,
        device=args.device,
        dtype=dtype_map[args.dtype],
        out_path=args.out,
    )


def main(paths, fallback_model, iters, max_timestep, device, dtype, out_path):
    recipe = distill_time_embed(
        *(sd_mecha.model(path, config=model_config) for path in paths),
        iters=iters,
        max_timestep=max_timestep,
    )
    recipe.set_cache()
    sd_mecha.merge(
        recipe,
        fallback_model=sd_mecha.model(fallback_model, config=model_config),
        merge_device=device,
        merge_dtype=dtype,
        threads=0,
        output=out_path,
    )


if __name__ == "__main__":
    main_cli()
