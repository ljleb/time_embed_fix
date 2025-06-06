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
        help="Two or more .safetensors checkpoints that will be used as reference to fix the time_embed keys.",
    )
    parser.add_argument(
        "--model_to_fix",
        metavar="PATH",
        required=True,
        help="The .safetensors checkpoint to fix.",
    )
    parser.add_argument(
        "--out",
        required=True,
        metavar="PATH",
        help="Where to write the fixed model.",
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        help="The weights of the input models. Must be either empty or match the number of models.",
    )
    parser.add_argument(
        "--prompt_dataset",
        default=None,
        metavar="PATH",
        help="Path to a directory containing one text file per prompt (each with .txt extension).",
    )
    parser.add_argument(
        "--prompt_dataset_seed",
        default=0,
        type=int,
        help="Seed used to shuffle the dataset. The dataset is reshuffled in the same order for each text encoder.",
    )
    parser.add_argument(
        "--prompt_batch_size",
        default=4,
        type=int,
        help="Batch size used to encode prompts with each text encoder.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help='Torch device to run on (default: "cuda").',
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

    args = parser.parse_args()
    main(
        paths=[str(pathlib.Path(p).absolute()) for p in args.models],
        model_to_fix=args.model_to_fix,
        alphas=args.alphas if args.alphas else None,
        prompt_dataset=args.prompt_dataset,
        prompt_dataset_seed=args.prompt_dataset_seed,
        prompt_batch_size=args.prompt_batch_size,
        iters=args.iters,
        device=args.device,
        dtype=dtype_map[args.dtype],
        out_path=args.out,
    )


def main(paths, model_to_fix, alphas, prompt_dataset, prompt_dataset_seed, prompt_batch_size, iters, device, dtype, out_path):
    recipe = distill_time_embed(
        *(sd_mecha.model(path, config=model_config) for path in paths),
        model_to_fix=sd_mecha.model(model_to_fix, config=model_config),
        alphas=torch.tensor(alphas, device=device, dtype=dtype),
        iters=iters,
        prompt_dataset=prompt_dataset,
        prompt_dataset_shuffle_seed=prompt_dataset_seed,
        prompt_encoding_batch_size=prompt_batch_size,
    )
    sd_mecha.merge(
        recipe,
        merge_device=device,
        merge_dtype=dtype,
        output=out_path,
    )


if __name__ == "__main__":
    main_cli()
