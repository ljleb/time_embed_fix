# Time Embed Fix

This is a method that reduces the "greyness" in generated images from merges between SDXL checkpoints that are slightly too far appart or not completely compatible.

## Install

From CLI:

```sh
git clone https://github.com/ljleb/time_embed_fix.git
cd time_embed_fix
pip install -r requirements.txt
```

Alternatively, you can also use this in ComfyUI using the [comfy-mecha nodepack](https://github.com/ljleb/comfy-mecha):

```sh
cd ComfyUI/custom_nodes/comfy-mecha/mecha_extensions
git clone https://github.com/ljleb/time_embed_fix.git
```

The node is called `Distill Time Embed`. Make sure the version of comfy-mecha is at least 1.2.36. Completely restart ComfyUI if it is already running.

## How to use

### CLI

CLI arguments to `distill_time_embed.py`:

- `--models`: Two or more .safetensors checkpoints that will be used as reference to fix the time_embed keys.
Usually, these are base models. The merge to be fixed is typically NOT included in this list.
- `--fallback_model`: The .safetensors checkpoint to fix.
- `--out`: Where to write the fixed model.
- `--device`: Torch device to run on (default: "cuda").
- `--dtype`: Torch dtype to run on (default: float32).
- `--iters`: Number of optimization steps (default: 4096).
- `--lr`: Learning rate (default: 1e-4).
- `--max_timestep`: Maximum timestep to optimize (default: 2048).

For example, to fix a merge `my_amazing_merge.safetensors` of Animagine 4.0 Zero and NoobAI 1.1 EPS:

```
python distill_time_embed.py \
    --models animagine-xl-4.0-zero.safetensors noobaiXLNAIXL_epsilonPred11Version.safetensors \
    --fallback_model my_amazing_merge.safetensors \
    --out F:\sd\models\Stable-diffusion\functional_106_timestep_train_2048.safetensors
```

This is equivalent to:

```
python distill_time_embed.py \
    --models animagine-xl-4.0-zero.safetensors noobaiXLNAIXL_epsilonPred11Version.safetensors \
    --fallback_model my_amazing_merge.safetensors \
    --out F:\sd\models\Stable-diffusion\functional_106_timestep_train_2048.safetensors \
    --device cuda:0 --dtype fp32 --iters 4096 --max_timestep 2048
```

### ComfyUI

In ComfyUI, this is one way to use the `Distill Time Embed` node:

![ComfyUI workflow illustrating how to use Time Embed Fix](/media/comfyui.png)

Using the `Fallback` node allows the method to be used as part of a larger merge workflow.  
Alternatively, the `Fallback` node can be removed and the `fallback_model` input of the `Merger` node can be used instead.

## Honorable Mention

Big thanks to Velvet Toroyashi and @vvv01d for reproducing my preliminary results and for going as far as experimenting with some alternatives and optimizing the script.
