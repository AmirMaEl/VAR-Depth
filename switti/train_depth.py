import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms as T

from Framework.data.marigold_dl.hypersim_dataset import get_hypersim_trainloader
from Framework.data.utils import save_data_batch
from eval_model import eval_kitti, get_kitti_dl, get_nyu_dl
from models import SwittiPipeline


device = 'cuda:0'

# model_path = "yresearch/Switti-1024"
model_path = "yresearch/Switti"


class RGBConditionAdapter(nn.Module):
    def __init__(
        self,
        seq_len=77,
        context_dim=2048,
        pooled_dim=1280,
        num_scales=10,
        prefix_len=8,
        prefix_rank=128,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_scales = num_scales
        self.prefix_len = prefix_len
        self.context_dim = context_dim
        self.prefix_rank = prefix_rank
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(512, 768, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
        )
        self.seq_pool = nn.AdaptiveAvgPool2d((7, 11))
        self.seq_proj = nn.Sequential(
            nn.Linear(768, 1024),
            nn.SiLU(),
            nn.Linear(1024, context_dim),
        )
        self.seq_norm = nn.LayerNorm(context_dim)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_proj = nn.Sequential(
            nn.Linear(768, 768),
            nn.SiLU(),
            nn.Linear(768, pooled_dim),
        )
        self.global_norm = nn.LayerNorm(pooled_dim)
        self.alpha_logit = nn.Parameter(torch.tensor(0.1))
        self.beta_logit = nn.Parameter(torch.tensor(0.1))
        self.prefix_token_proj = nn.Sequential(
            nn.Linear(768, 512),
            nn.SiLU(),
            nn.Linear(512, num_scales * prefix_len * prefix_rank),
        )
        self.prefix_context_basis = nn.Parameter(0.02 * torch.randn(prefix_rank, context_dim))
        self.prefix_delta_norm = nn.LayerNorm(context_dim)
        self.prefix_delta_logit = nn.Parameter(torch.tensor(-2.0))

    def forward(self, rgb_B3HW):
        feat = self.encoder(rgb_B3HW)

        seq_feat = self.seq_pool(feat).flatten(2).transpose(1, 2)
        seq_cond = self.seq_proj(seq_feat)
        seq_cond = self.seq_norm(seq_cond)

        pooled_feat = self.global_pool(feat).flatten(1)
        pooled_cond = self.global_proj(pooled_feat)
        pooled_cond = self.global_norm(pooled_cond)
        prefix_tokens = self.prefix_token_proj(pooled_feat)
        prefix_tokens = prefix_tokens.view(
            rgb_B3HW.shape[0], self.num_scales, self.prefix_len, self.prefix_rank
        )
        prefix_delta = torch.einsum("bspr,rc->bspc", prefix_tokens, self.prefix_context_basis)
        prefix_delta = self.prefix_delta_norm(prefix_delta)
        return seq_cond, pooled_cond, prefix_delta

    def blend(self, text_context, text_vector, rgb_context, rgb_vector):
        alpha = torch.sigmoid(self.alpha_logit)
        beta = torch.sigmoid(self.beta_logit)
        cond_context = text_context + alpha * rgb_context
        cond_vector = text_vector + beta * rgb_vector
        return cond_context, cond_vector, alpha.item(), beta.item()

    def build_hybrid_prefix(self, static_prefix_scales, conditional_prefix_delta):
        gamma = torch.sigmoid(self.prefix_delta_logit)
        hybrid_prefix = static_prefix_scales.unsqueeze(0) + gamma * conditional_prefix_delta
        batch_prefix_context = hybrid_prefix.view(
            hybrid_prefix.shape[0], self.num_scales * self.prefix_len, self.context_dim
        )
        return batch_prefix_context, gamma.item()

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_adapter_pipeline_bundle(
    adapter,
    prefix_context_scales,
    optimizer,
    step,
    model_path,
    prefix_len,
    prefix_mode="hybrid_conditional",
    train_version="v2",
    save_dir="checkpoints/depth_adapter_pipeline",
):
    root = Path(save_dir)
    root.mkdir(parents=True, exist_ok=True)

    ckpt_file = root / "adapter_prefix.pt"
    payload = {
        "checkpoint_format_version": 2,
        "prefix_mode": prefix_mode,
        "train_version": train_version,
        "step": step,
        "adapter_state_dict": adapter.state_dict(),
        "static_prefix_context_scales": prefix_context_scales.detach().cpu(),
        "prefix_context_scales": prefix_context_scales.detach().cpu(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(payload, ckpt_file)

    config = {
        "base_model": model_path,
        "artifact": "switti-depth-adapter-pipeline",
        "pipeline_format_version": 2,
        "checkpoint_file": "adapter_prefix.pt",
        "prefix_mode": prefix_mode,
        "train_version": train_version,
        "prefix_len": int(prefix_len),
        "num_scales": int(prefix_context_scales.shape[0]),
        "context_dim": int(prefix_context_scales.shape[-1]),
        "conditional_prefix_rank": int(getattr(adapter, "prefix_rank", 128)),
        "uses_conditional_prefix_head": True,
        "adapter_class": "RGBConditionAdapter",
    }
    with open(root / "pipeline_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    latest_file = root / "latest_step.txt"
    latest_file.write_text(str(step), encoding="utf-8")
    print(f"[pipeline] saved upload-ready bundle to {root}")


def load_adapter_pipeline_bundle(
    adapter,
    prefix_context_scales,
    optimizer,
    save_dir="checkpoints/depth_adapter_pipeline",
):
    root = Path(save_dir)
    ckpt_file = root / "adapter_prefix.pt"
    if not ckpt_file.exists():
        print(f"[pipeline] no checkpoint found at {ckpt_file}, starting fresh")
        return 0

    checkpoint = torch.load(ckpt_file, map_location="cpu")
    missing_keys, unexpected_keys = adapter.load_state_dict(
        checkpoint["adapter_state_dict"], strict=False
    )
    if len(missing_keys) > 0 or len(unexpected_keys) > 0:
        print(
            "[pipeline] adapter checkpoint loaded with non-strict key match "
            f"(missing={len(missing_keys)}, unexpected={len(unexpected_keys)})"
        )

    loaded_prefix = checkpoint.get("static_prefix_context_scales", checkpoint.get("prefix_context_scales"))
    if loaded_prefix is None:
        raise ValueError(
            "checkpoint missing both 'static_prefix_context_scales' and 'prefix_context_scales'"
        )
    loaded_prefix = loaded_prefix.to(prefix_context_scales.device)
    if loaded_prefix.shape != prefix_context_scales.shape:
        raise ValueError(
            f"prefix_context_scales shape mismatch: expected {tuple(prefix_context_scales.shape)}, "
            f"got {tuple(loaded_prefix.shape)}"
        )
    with torch.no_grad():
        prefix_context_scales.copy_(loaded_prefix)

    if "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except ValueError as exc:
            print(f"[pipeline] skipping optimizer state restore due to mismatch: {exc}")

    last_step = int(checkpoint.get("step", -1))
    next_step = last_step + 1
    ckpt_version = int(checkpoint.get("checkpoint_format_version", 1))
    prefix_mode = checkpoint.get("prefix_mode", "static")
    print(
        f"[pipeline] resumed from {ckpt_file} at step={last_step} "
        f"(version={ckpt_version}, prefix_mode={prefix_mode})"
    )
    return next_step


def train(
    restart_from_beginning=False,
    eval_interval=-1,
    save_interval=500,
    supervised_samples=25,
    train_version="v2",
):
    # Stage 1: fine-tune the Switti depth prior using RGB-conditioned adapters.
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    pipe = SwittiPipeline.from_pretrained(model_path, device=device, torch_dtype=torch.float32, reso=512)
    BS = 2
    # ld = get_vkitti2_trainloader(batch_size=BS, return_dataset=True)
    ld = get_hypersim_trainloader(batch_size=BS, return_dataset=True)
    if supervised_samples > 0:
        subset_size = min(supervised_samples, len(ld))
        ld = torch.utils.data.Subset(ld, list(range(subset_size)))
        print(f"[data] using supervised subset: {subset_size} samples")
    else:
        print("[data] using full supervised dataset")
    ld = torch.utils.data.DataLoader(ld, batch_size=BS, shuffle=True, num_workers=0, drop_last=True)
    model = pipe.switti
    val_dl = get_kitti_dl(bs=2)['dl']
    sample_val = next(iter(val_dl))['rgb_norm'].to(device)  
    sample_val = T.Resize((512, 512))(sample_val)

    val_dl1 = get_nyu_dl(bs=2)['dl']
    sample_val1 = next(iter(val_dl1))['rgb_norm'].to(device)
    sample_val1 = T.Resize((512, 512))(sample_val1)


    pipe.vae = pipe.vae.to(device)
    model = model.to(device)
    model.eval()
    pipe.switti = model
    vae_device = next(pipe.vae.parameters()).device
    model_device = next(model.parameters()).device

    for param in model.parameters():
        param.requires_grad = False

    train_loss = torch.nn.CrossEntropyLoss(reduction="none")
    V = pipe.vae.vocab_size

    B = BS
    prompts = ["a photo of a street"] * B
    with torch.no_grad():
        (
            prompt_embeds,
            pooled_prompt_embeds,
            prompt_attn_bias,
        ) = pipe.encode_prompt(prompts, encode_null=False)

    prompt_embeds = prompt_embeds.to(model_device)
    pooled_prompt_embeds = pooled_prompt_embeds.to(model_device)
    prompt_attn_bias = prompt_attn_bias.to(model_device)

    prefix_len = 8
    prefix_rank = 128
    context_dim = prompt_embeds.shape[-1]
    num_scales = len(model.patch_nums)
    prefix_context_scales = nn.Parameter(
        0.02 * torch.randn(num_scales, prefix_len, context_dim, device=model_device)
    )

    adapter = RGBConditionAdapter(
        seq_len=prompt_embeds.shape[1],
        context_dim=prompt_embeds.shape[-1],
        pooled_dim=pooled_prompt_embeds.shape[-1],
        num_scales=num_scales,
        prefix_len=prefix_len,
        prefix_rank=prefix_rank,
    ).to(model_device)
    adapter.train()

    if train_version not in {"v1", "v2", "v3"}:
        raise ValueError(f"Unsupported train_version='{train_version}'. Expected one of: v1, v2, v3")

    dynamic_prefix_param_names = {
        "prefix_token_proj.0.weight",
        "prefix_token_proj.0.bias",
        "prefix_token_proj.2.weight",
        "prefix_token_proj.2.bias",
        "prefix_context_basis",
        "prefix_delta_norm.weight",
        "prefix_delta_norm.bias",
        "prefix_delta_logit",
    }

    for name, param in adapter.named_parameters():
        if train_version in {"v1", "v3"} and name in dynamic_prefix_param_names:
            param.requires_grad = False
        else:
            param.requires_grad = True

    if train_version == "v1":
        prefix_mode = "static"
        prefix_context_scales.requires_grad = True
        print("[train] version=v1 | static prefix tuning")
    elif train_version == "v2":
        prefix_mode = "hybrid_conditional"
        prefix_context_scales.requires_grad = False
        print("[train] version=v2 | dynamic prefix tuning")
    else:
        prefix_mode = "none"
        prefix_context_scales.requires_grad = False
        for p in model.blocks[-1].parameters():
            p.requires_grad = True
        model.train()
        print("[train] version=v3 | no prefix tuning, train adapter + last Switti block")

    checkpoint_dir = f"checkpoints/depth_adapter_pipeline/{train_version}"
    print(f"[pipeline] checkpoint dir: {checkpoint_dir}")

    adapter_named_params = dict(adapter.named_parameters())
    gate_param_names = {"alpha_logit", "beta_logit", "prefix_delta_logit"}
    gate_params = [
        p for name, p in adapter_named_params.items()
        if name in gate_param_names and p.requires_grad
    ]
    non_gate_params = [
        p for name, p in adapter_named_params.items()
        if name not in gate_param_names and p.requires_grad
    ]
    prefix_params = [prefix_context_scales]
    switti_last_block_params = [p for p in model.blocks[-1].parameters() if p.requires_grad]

    base_non_gate_lr = 3e-4
    base_gate_lr = 1e-3
    base_prefix_lr = 1e-3
    base_backbone_lr = 3e-5
    optimizer_groups = []
    group_lr_lookup = {}
    if len(non_gate_params) > 0:
        optimizer_groups.append({"params": non_gate_params, "lr": base_non_gate_lr})
        group_lr_lookup[id(non_gate_params[0])] = base_non_gate_lr
    if len(gate_params) > 0:
        optimizer_groups.append({"params": gate_params, "lr": base_gate_lr})
        group_lr_lookup[id(gate_params[0])] = base_gate_lr
    if any(p.requires_grad for p in prefix_params):
        optimizer_groups.append({"params": prefix_params, "lr": base_prefix_lr})
        group_lr_lookup[id(prefix_params[0])] = base_prefix_lr
    if len(switti_last_block_params) > 0:
        optimizer_groups.append({"params": switti_last_block_params, "lr": base_backbone_lr})
        group_lr_lookup[id(switti_last_block_params[0])] = base_backbone_lr

    optimizer = torch.optim.Adam(optimizer_groups)
    if restart_from_beginning:
        start_step = 0
        print("[pipeline] restart requested, starting from step=0 without loading checkpoint")
    else:
        start_step = load_adapter_pipeline_bundle(
            adapter=adapter,
            prefix_context_scales=prefix_context_scales,
            optimizer=optimizer,
            save_dir=checkpoint_dir,
        )
    adapter_trainable = count_trainable_params(adapter)
    prefix_trainable = sum(p.numel() for p in prefix_params if p.requires_grad)
    optim_trainable = sum(
        p.numel()
        for group in optimizer.param_groups
        for p in group["params"]
        if p.requires_grad
    )
    print(
        f"Trainable params | adapter={adapter_trainable:,} "
        f"prefix={prefix_trainable:,} "
        f"last_block={sum(p.numel() for p in switti_last_block_params):,} "
        f"total={optim_trainable:,}"
    )

    @torch.inference_mode()
    def predict_depth(tmpimg):
        if tmpimg.dtype == torch.int32:
            tmpimg = (tmpimg.float() / 127.5 - 1)
        tmpimg = tmpimg.to(model_device)
        batch_size = tmpimg.shape[0]

        rgb_context, rgb_vector, prefix_delta = adapter(tmpimg)
        cond_context, cond_vector, _, _ = adapter.blend(
            prompt_embeds[:batch_size],
            pooled_prompt_embeds[:batch_size],
            rgb_context,
            rgb_vector,
        )

        if prefix_mode == "none":
            cond_context_with_prefix = cond_context
            cond_attn_bias_with_prefix = prompt_attn_bias[:batch_size]
        elif prefix_mode == "static":
            flat_prefix_context = prefix_context_scales.reshape(1, num_scales * prefix_len, context_dim)
            batch_prefix_context = flat_prefix_context.expand(batch_size, -1, -1)
            cond_context_with_prefix = torch.cat([batch_prefix_context, cond_context], dim=1)
            prefix_attn_bias = torch.zeros(
                batch_size,
                num_scales * prefix_len,
                device=model_device,
                dtype=prompt_attn_bias.dtype,
            )
            cond_attn_bias_with_prefix = torch.cat([prefix_attn_bias, prompt_attn_bias[:batch_size]], dim=1)
        else:
            batch_prefix_context, _ = adapter.build_hybrid_prefix(
                static_prefix_scales=prefix_context_scales,
                conditional_prefix_delta=prefix_delta,
            )
            cond_context_with_prefix = torch.cat([batch_prefix_context, cond_context], dim=1)
            prefix_attn_bias = torch.zeros(
                batch_size,
                num_scales * prefix_len,
                device=model_device,
                dtype=prompt_attn_bias.dtype,
            )
            cond_attn_bias_with_prefix = torch.cat([prefix_attn_bias, prompt_attn_bias[:batch_size]], dim=1)
        pipe.switti.eval()
        output = pipe(
            prompts[:batch_size],
            cond_context=cond_context_with_prefix,
            condition_vector=cond_vector,
            cond_attn_bias=cond_attn_bias_with_prefix,
            return_pil=False,
            cfg=0.0,
            turn_off_cfg_start_si=0,
        )
        output = (output * 2) - 1
        return output

    steps = 30000
    use_constant_lr = True
    ld_iter = iter(ld)
    fixed_batch = None
    pbar = tqdm(range(start_step, start_step + steps), desc="Single-batch overfit")
    for step in pbar:
        if use_constant_lr:
            for group in optimizer.param_groups:
                group_params = group["params"]
                if len(group_params) == 0:
                    continue
                group["lr"] = group_lr_lookup.get(id(group_params[0]), group["lr"])

        optimizer.zero_grad()

        try:
            batch = next(ld_iter)
        except StopIteration:
            ld_iter = iter(ld)
            batch = next(ld_iter)

        rgb_inp_B3HW_model = T.Resize((512, 512))(batch['rgb_norm']).to(model_device)
        depth = T.Resize((512, 512))(batch['depth_filled_norm'].repeat(1, 3, 1, 1)).to(vae_device)
        rgb_vis = T.Resize((512, 512))(batch['rgb_norm']).cpu()
        with torch.no_grad():
            gt_idx_Bl = pipe.vae.img_to_idxBl(depth)
            gt_BL = torch.cat(gt_idx_Bl, dim=1).to(model_device)
            x_BLCv_wo_first_l = pipe.vae.quantize.idxBl_to_switti_input(gt_idx_Bl).to(model_device)
        L = gt_BL.shape[1]
        loss_weight = torch.ones(1, L, device=model_device) / L

        rgb_context, rgb_vector, prefix_delta = adapter(rgb_inp_B3HW_model)
        cond_context, cond_vector, alpha, beta = adapter.blend(
            prompt_embeds,
            pooled_prompt_embeds,
            rgb_context,
            rgb_vector,
        )

        if prefix_mode == "none":
            cond_context_with_prefix = cond_context
            cond_attn_bias_with_prefix = prompt_attn_bias
            gamma = 0.0
        elif prefix_mode == "static":
            flat_prefix_context = prefix_context_scales.reshape(1, num_scales * prefix_len, context_dim)
            batch_prefix_context = flat_prefix_context.expand(B, -1, -1)
            cond_context_with_prefix = torch.cat([batch_prefix_context, cond_context], dim=1)
            prefix_attn_bias = torch.zeros(
                B,
                num_scales * prefix_len,
                device=model_device,
                dtype=prompt_attn_bias.dtype,
            )
            cond_attn_bias_with_prefix = torch.cat([prefix_attn_bias, prompt_attn_bias], dim=1)
            gamma = 0.0
        else:
            batch_prefix_context, gamma = adapter.build_hybrid_prefix(
                static_prefix_scales=prefix_context_scales,
                conditional_prefix_delta=prefix_delta,
            )
            cond_context_with_prefix = torch.cat([batch_prefix_context, cond_context], dim=1)
            prefix_attn_bias = torch.zeros(
                B,
                num_scales * prefix_len,
                device=model_device,
                dtype=prompt_attn_bias.dtype,
            )
            cond_attn_bias_with_prefix = torch.cat([prefix_attn_bias, prompt_attn_bias], dim=1)

        logits_BLV = model(
            x_BLCv_wo_first_l,
            prompt_embeds=cond_context_with_prefix,
            pooled_prompt_embeds=cond_vector,
            prompt_attn_bias=cond_attn_bias_with_prefix,
            batch_width=[512] * B,
            batch_height=[512] * B,
        )

        loss = train_loss(
            logits_BLV.view(-1, V),
            gt_BL.reshape(-1),
        ).view(B, -1)
        loss = loss.mul(loss_weight).sum(dim=-1).mean()
        loss.backward()
        trainable_params = [p for p in adapter.parameters() if p.requires_grad]
        if any(p.requires_grad for p in prefix_params):
            trainable_params += prefix_params
        trainable_params += switti_last_block_params
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
        optimizer.step()
        pbar.set_description(
            f"Single-batch overfit | step {step} loss {loss.item():.4f} "
            f"a={alpha:.4f} b={beta:.4f} g={gamma:.4f} lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if step % save_interval == 0 and save_interval > 0:
            save_adapter_pipeline_bundle(
                adapter=adapter,
                prefix_context_scales=prefix_context_scales,
                optimizer=optimizer,
                step=step,
                model_path=model_path,
                prefix_len=prefix_len,
                prefix_mode=prefix_mode,
                train_version=train_version,
                save_dir=checkpoint_dir,
            )

        if step % 20 == 0 or step == steps - 1:
            if eval_interval > 0 and step % eval_interval == 0:
                eval_kitti(predict_depth, hw=(512, 512))
            output = predict_depth(rgb_inp_B3HW_model)
            val_pred = predict_depth(sample_val)
            val_pred1 = predict_depth(sample_val1)


                
            out = torch.cat([rgb_vis, output.cpu(),depth.cpu(), sample_val.cpu(), val_pred.cpu(), sample_val1.cpu(), val_pred1.cpu()], dim=0)
            save_data_batch(out, f"overfit_step.png", grid=[3, 5])

            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train depth adapter for Switti")
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Start from step 0 and skip loading any saved checkpoint",
    )
    parser.add_argument(
        "--eval-iter",
        "--evaliter",
        dest="eval_iter",
        type=int,
        default=-1,
        help="Run eval every N steps. Use -1 (default) to disable periodic eval.",
    )
    parser.add_argument(
        "--save",
        "--saveinterval",
        dest="save_interval",
        type=int,
        default=500,
        help="Save checkpoint every N steps. Use <=0 to disable saving.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=25,
        help="Number of supervised training samples to use. Use <=0 to use full dataset.",
    )
    parser.add_argument(
        "--version",
        choices=["v1", "v2", "v3"],
        default="v2",
        help=(
            "Training version: v1=static prefix tuning, "
            "v2=dynamic prefix tuning, "
            "v3=no prefix tuning (train condition adapter + last Switti block)."
        ),
    )
    args = parser.parse_args()

    train(
        restart_from_beginning=args.restart,
        eval_interval=args.eval_iter,
        save_interval=args.save_interval,
        supervised_samples=args.samples,
        train_version=args.version,
    )