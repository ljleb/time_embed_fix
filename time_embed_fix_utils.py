import torch
import math
from sd_mecha import merge_method, Parameter, StateDict, Return, StateDictKeyError
from sd_mecha.extensions import model_configs
from torch import Tensor


model_config = model_configs.resolve("sdxl-sgm")
TIME_EMBED_KEYS = {
    key for key in model_config.keys()
    if ".time_embed." in key or "emb_layers" in key
}


@merge_method
def distill_time_embed(
    *models: Parameter(StateDict[Tensor], "weight", model_config),
    iters: Parameter(int) = 8192,
    max_timestep: Parameter(int) = 2048,
    **kwargs,
) -> Return(Tensor, "weight", model_config):
    if len(models) < 2:
        raise RuntimeError("distill_time_embed needs at least 2 model as input")

    key = kwargs["key"]
    if key not in TIME_EMBED_KEYS:
        raise StateDictKeyError(key)

    cache = kwargs.get("cache")
    if cache is None:
        raise RuntimeError("distill_time_embed must be executed with cache")

    init_sd = {}
    dummy_key = models[0][next(iter(TIME_EMBED_KEYS))]
    device, dtype = dummy_key.device, dummy_key.dtype

    if key in cache:
        return cache[key].to(device=device, dtype=dtype)

    with torch.no_grad():
        model = TimeEmbed(device=device, dtype=dtype)
        timesteps = get_timesteps(max_timestep, model.time_channels, device, dtype)

        for i, sd in enumerate(models, start=1):
            update_model(init_sd, 1/i, sd, model, timesteps, device, dtype)

        target = init_sd.pop("target")
        model.load_state_dict(init_sd)

        trained_sd = train(model, timesteps, target, iters)
        for k, v in trained_sd.items():
            cache[k] = v.cpu()

    return cache[key].to(device=device, dtype=dtype)


def get_timesteps(max_timestep, time_channels, device, dtype):
    timesteps = torch.arange(max_timestep+1, device=device, dtype=dtype)
    embeds = timestep_embedding(timesteps, time_channels)
    return embeds


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=timesteps.dtype,
                                             device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def update_model(
    average_sd: dict, weight: float,
    model_sd, model: "TimeEmbed", timesteps: torch.Tensor,
    device, dtype,
) -> None:
    sd = {}
    for key in model_sd.keys():
        if key in TIME_EMBED_KEYS:
            sd_key = ".".join(key.split(".")[2:])
            sd[sd_key] = model_sd[key].to(device=device, dtype=dtype)
            average_sd[sd_key] = average_sd.get(sd_key, 0) + (sd[sd_key] - average_sd.get(sd_key, 0)) * weight

    model.load_state_dict(sd)
    target_i = model(timesteps)
    average_sd["target"] = average_sd.get("target", 0) + (target_i - average_sd.get("target", 0)) * weight


class TimeEmbed(torch.nn.Module):
    def __init__(self, time_channels=320, time_embed_dim=1280, device=None, dtype=None):
        super().__init__()
        self.time_channels = time_channels
        self.time_embed = torch.nn.Sequential(
            torch.nn.Linear(time_channels, time_embed_dim, dtype=dtype, device=device),
            torch.nn.SiLU(),
            torch.nn.Linear(time_embed_dim, time_embed_dim, dtype=dtype, device=device),
        )
        self.input_blocks = torch.nn.ModuleList([
            torch.nn.ModuleList([EmbLayerBlock(320 * 2**(i // 3), dtype=dtype, device=device)])
            if i % 3 != 0
            else torch.nn.Identity()
            for i in range(9)
        ])
        self.middle_block = torch.nn.ModuleList([
            EmbLayerBlock(320*(2**2), dtype=dtype, device=device)
            if i % 2 == 0
            else torch.nn.Identity()
            for i in range(3)
        ])
        self.output_blocks = torch.nn.ModuleList([
            torch.nn.ModuleList([EmbLayerBlock(320 * 2**(2 - i // 3), dtype=dtype, device=device)])
            for i in range(9)
        ])

    def forward(self, x):
        x = self.time_embed(x)
        res = []
        for blocks in (*self.input_blocks, self.middle_block, *self.output_blocks):
            if isinstance(blocks, torch.nn.Identity):
                continue

            for block in blocks:
                res.append(block(x))
        return torch.cat(res, dim=-1)


class EmbLayerBlock(torch.nn.Module):
    def __init__(self, out_dim, in_dim=1280, dtype=None, device=None):
        super().__init__()
        self.emb_layers = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(in_dim, out_dim, dtype=dtype, device=device),
        )

    def forward(self, x):
        return self.emb_layers(x)


@torch.enable_grad()
def train(model, timesteps, target, iters):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), 1e-4)

    iters_num_digits = math.floor(math.log10(iters)+1)
    for i in range(iters):
        emb_c = model(timesteps)
        loss = torch.nn.functional.mse_loss(emb_c, target, reduction="none")
        max_loss = loss.detach().abs().max()
        loss = loss.mean()

        print(f"it {i+1:{iters_num_digits}}, loss: {loss.item():0.8f}, max_loss: {max_loss.item():0.8f}")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    with torch.no_grad():
        return {"model.diffusion_model." + k: v for k, v in model.state_dict().items()}
