import pathlib
import torch
import math
import threading
from sd_mecha import merge_method, Parameter, StateDict, Return
from sd_mecha.extensions import model_configs
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from tqdm import trange, tqdm
from transformers import CLIPTokenizerFast
from clip_g import create_tokenizer, create_text_model, TextTransformer


model_config = model_configs.resolve("sdxl-sgm")
TIME_EMBED_KEYS = sorted([
    key for key in model_config.keys()
    if ".time_embed." in key or "emb_layers" in key or ".label_emb." in key
], key=lambda v: model_config.keys()[v].shape.numel(), reverse=True)
TEXT_ENCODER_KEYS = [
    key for key in model_config.keys()
    if key.startswith("conditioner.embedders.1") and "logit_scale" not in key
]


@merge_method
def distill_time_embed(
    *models: Parameter(StateDict[Tensor], "weight", model_config),
    model_to_fix: Parameter(StateDict[Tensor], "weight", model_config),
    alphas: Parameter(Tensor, "param") = None,
    iters: Parameter(int) = 4096,
    prompt_dataset: Parameter(str) = None,
    prompt_encoding_batch_size: Parameter(int) = 4,
    prompt_dataset_shuffle_seed: Parameter(int) = 0,
    **kwargs,
) -> Return(Tensor, "weight", model_config):
    if len(models) < 2:
        raise RuntimeError("distill_time_embed needs at least 2 model as input")

    key = kwargs["key"]
    if key not in TIME_EMBED_KEYS:
        return model_to_fix[key]

    cache = kwargs.get("cache")
    if cache is None:
        raise RuntimeError("distill_time_embed must be executed with cache")

    prompt_encoding_batch_size = max(prompt_encoding_batch_size, 1)

    with cache.setdefault("lock", threading.Lock()), torch.no_grad():
        init_sd = {}
        dummy_key = models[0][next(iter(TIME_EMBED_KEYS))]
        device, dtype = dummy_key.device, dummy_key.dtype

        if key in cache:
            return cache[key].to(device=device, dtype=dtype)

        if prompt_dataset is not None:
            prompt_dataset_generator = torch.Generator()
            prompt_dataset = pathlib.Path(prompt_dataset)
            if not prompt_dataset.is_dir():
                raise RuntimeError("prompt_dataset needs to be a valid directory.")
            prompt_dataset = DataLoader(
                PromptDirectoryDataset(prompt_dataset),
                batch_size=prompt_encoding_batch_size,
                shuffle=True,
                generator=prompt_dataset_generator,
                num_workers=0,
            )
            tokenizer = create_tokenizer()
            text_model = create_text_model(device=device, dtype=torch.bfloat16)
        else:
            tokenizer = text_model = None

        time_embed_model = TimeEmbed(device=device, dtype=dtype)
        timesteps = get_timesteps(time_embed_model.time_channels, device, dtype)

        if alphas is None:
            alphas = torch.ones(len(models), device=device, dtype=dtype)
        alphas_cumsum = alphas.cumsum(dim=0)

        for i, sd in enumerate(models):
            if prompt_dataset is not None:
                prompt_dataset_generator.manual_seed(prompt_dataset_shuffle_seed)

            update_init_and_target(
                init_sd, alphas[i] / alphas_cumsum[i],
                sd, time_embed_model, timesteps,
                tokenizer, text_model, prompt_dataset,
                device, dtype,
            )

        target = init_sd.pop("target")  # extra key introduced by `update_init_and_target`
        if prompt_dataset is not None:
            prompt_dataset_generator.manual_seed(prompt_dataset_shuffle_seed)

        adm = encode_adm(model_to_fix, text_model, tokenizer, prompt_dataset, device, dtype)
        if prompt_dataset is not None:
            del text_model, tokenizer, prompt_dataset, prompt_dataset_generator
        time_embed_model.load_state_dict(init_sd)

        trained_sd = train(time_embed_model, timesteps, adm, target, iters)
        for k, v in trained_sd.items():
            cache[k] = v.cpu()

        return cache[key].to(device=device, dtype=dtype)


distill_time_embed_create_recipe_original = distill_time_embed.create_recipe
distill_time_embed.create_recipe = lambda *args, **kwargs: distill_time_embed_create_recipe_original(*args, **kwargs).set_cache()


def get_timesteps(time_channels, device, dtype):
    timesteps = torch.arange(1000, device=device, dtype=dtype)
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
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=timesteps.dtype, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def update_init_and_target(
    average_sd: dict, weight: float,
    model_sd, time_embed_model: "TimeEmbed", timesteps: torch.Tensor,
    tokenizer: CLIPTokenizerFast, text_model: TextTransformer, prompt_dataset: "DataLoader",
    device, dtype,
) -> None:
    adm = encode_adm(model_sd, text_model, tokenizer, prompt_dataset, device, dtype)
    time_embed_sd = {}
    for key in model_sd.keys():
        if key in TIME_EMBED_KEYS:
            sd_key = ".".join(key.split(".")[2:])
            time_embed_sd[sd_key] = model_sd[key].to(device=device, dtype=dtype)
            average_sd[sd_key] = average_sd.get(sd_key, 0) + (time_embed_sd[sd_key] - average_sd.get(sd_key, 0)) * weight

    time_embed_model.load_state_dict(time_embed_sd)
    target_i = time_embed_model(timesteps, adm)
    average_sd["target"] = average_sd.get("target", 0) + (target_i - average_sd.get("target", 0)) * weight


class TimeEmbed(torch.nn.Module):
    def __init__(self, time_channels=320, label_channels=2816, embed_dim=1280, device=None, dtype=None):
        super().__init__()
        self.time_channels = time_channels
        self.time_embed = torch.nn.Sequential(
            torch.nn.Linear(time_channels, embed_dim, dtype=dtype, device=device),
            torch.nn.SiLU(),
            torch.nn.Linear(embed_dim, embed_dim, dtype=dtype, device=device),
        )
        self.label_emb = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(label_channels, embed_dim, dtype=dtype, device=device),
                torch.nn.SiLU(),
                torch.nn.Linear(embed_dim, embed_dim, dtype=dtype, device=device),
            ),
        ])
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

    def forward(self, x, y=None):
        x = self.time_embed(x)
        if y is not None:
            x = x + average_adm_to_1k(self.label_emb[0](y))

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


def encode_adm(model_sd, text_model, tokenizer, prompt_dataset, device, dtype):
    if prompt_dataset is None:
        return None

    all_pooled = []
    text_sd = {}
    for key in model_sd.keys():
        if key in TEXT_ENCODER_KEYS:
            sd_key = ".".join(key.split(".")[4:])
            text_sd[sd_key] = model_sd[key].to(device=device, dtype=torch.bfloat16)

    text_model.load_state_dict(text_sd)
    for prompts in tqdm(prompt_dataset, desc="Encoding"):
        tokens = tokenizer(prompts, padding="max_length", truncation=True, return_tensors="pt")
        pooled = text_model(tokens["input_ids"].to(device=device)).to(dtype=dtype)
        all_pooled.append(pooled)

    all_pooled = torch.cat(all_pooled, dim=0)

    width = 1024
    height = 1024
    crop_w = 0
    crop_h = 0
    target_width = width
    target_height = height

    adm_dim = 256
    out = [
        timestep_embedding(torch.tensor([height], device=all_pooled.device, dtype=all_pooled.dtype), adm_dim),
        timestep_embedding(torch.tensor([width], device=all_pooled.device, dtype=all_pooled.dtype), adm_dim),
        timestep_embedding(torch.tensor([crop_h], device=all_pooled.device, dtype=all_pooled.dtype), adm_dim),
        timestep_embedding(torch.tensor([crop_w], device=all_pooled.device, dtype=all_pooled.dtype), adm_dim),
        timestep_embedding(torch.tensor([target_height], device=all_pooled.device, dtype=all_pooled.dtype), adm_dim),
        timestep_embedding(torch.tensor([target_width], device=all_pooled.device, dtype=all_pooled.dtype), adm_dim),
    ]
    flat = torch.flatten(torch.cat(out)).unsqueeze(0).repeat(all_pooled.shape[0], 1)
    return torch.cat((all_pooled.to(flat.device), flat), dim=1)


def average_adm_to_1k(adm: torch.Tensor) -> torch.Tensor:
    """
    Reduce an [N, D] ADM tensor to [1000, D] using the
    “split‑then‑2‑part‑average” strategy the user described.
    """
    n, d = adm.shape
    if n == 1000:
        return adm

    full_blocks, tail_len = divmod(n, 1000)
    L = tail_len if tail_len else 1000

    front_parts = [adm[i*1000:i*1000 + L]
                   for i in range(full_blocks)]
    if tail_len:
        front_parts.append(adm[-tail_len:])

    back_parts = []
    if L < 1000:
        for i in range(full_blocks):
            back_parts.append(adm[i*1000 + L:(i+1)*1000])

    front_avg = torch.stack(front_parts).mean(dim=0)
    if back_parts:
        back_avg = torch.stack(back_parts).mean(dim=0)
        aligned = torch.cat([front_avg, back_avg], dim=0)
    else:
        aligned = front_avg

    return aligned


class PromptDirectoryDataset(Dataset):
    def __init__(self, directory: pathlib.Path):
        self.file_paths = list(f for f in directory.iterdir() if f.is_file() and f.suffix == ".txt")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        with open(file_path, 'r', encoding='utf-8') as f:
            prompt = f.readline().rstrip("\n")
        return prompt


@torch.enable_grad()
def train(model, timesteps, adm, target, iters):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), 1e-4)

    progress_bar = trange(iters, desc="Training", unit="it")
    for _ in progress_bar:
        emb_c = model(timesteps, adm)
        loss = torch.nn.functional.mse_loss(emb_c, target, reduction="none")
        max_loss = loss.detach().abs().max()
        loss = loss.mean()

        progress_bar.set_postfix_str(f"loss: {loss.item():0.8f}, max_loss: {max_loss.item():0.8f}")

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    with torch.no_grad():
        model.eval()
        return {"model.diffusion_model." + k: v for k, v in model.state_dict().items()}
