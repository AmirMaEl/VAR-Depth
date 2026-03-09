import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from diffusers import UNet2DModel

from Framework.data.marigold_dl.hypersim_dataset import get_hypersim_trainloader
from Framework.data.marigold_dl.vkitti_dataset import get_vkitti2_trainloader
from Framework.data.utils import save_data_batch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_train_loader(args: argparse.Namespace) -> DataLoader:
    ds_hypersim = get_hypersim_trainloader(
        size=(args.height, args.width),
        batch_size=args.batch_size,
        shuffle=True,
        return_dataset=True,
    )
    ds_vkitti = get_vkitti2_trainloader(
        size=(args.height, args.width),
        batch_size=args.batch_size,
        shuffle=True,
        return_dataset=True,
    )

    dataset = ConcatDataset([ds_hypersim, ds_vkitti])
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader


def make_var_scale_prediction(depth_target: torch.Tensor) -> torch.Tensor:
    b, _, h, w = depth_target.shape
    down_factor = torch.randint(low=2, high=9, size=(1,), device=depth_target.device).item()
    low_h = max(8, h // down_factor)
    low_w = max(8, w // down_factor)

    coarse = F.interpolate(depth_target, size=(low_h, low_w), mode="bilinear", align_corners=False)
    coarse = F.interpolate(coarse, size=(h, w), mode="bilinear", align_corners=False)

    noise = 0.02 * torch.randn_like(coarse)
    bias = 0.03 * torch.randn((b, 1, 1, 1), device=depth_target.device)
    coarse = (coarse + noise + bias).clamp(-1.0, 1.0)
    return coarse


def save_checkpoint(
    out_dir: Path,
    model: UNet2DModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"ckpt_step_{step:07d}.pt"
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        },
        ckpt_path,
    )


def train(args: argparse.Namespace) -> None:
    # Stage 2: train the conditional diffusion upsampler with RGB + coarse depth input.
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_loader = build_train_loader(args)

    model = UNet2DModel(
        sample_size=(args.height, args.width),
        in_channels=4,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 256),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_step = 0
    if args.resume and os.path.exists(args.resume):
        payload = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        start_step = int(payload["step"]) + 1
        print(f"Resumed from {args.resume} at step {start_step}")

    model.train()
    pbar = tqdm(total=args.max_steps, initial=start_step, desc="Train diffusion UNet upsampler")
    data_iter = iter(train_loader)

    step = start_step
    while step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        rgb = batch["rgb_norm"].to(device, non_blocking=True)
        depth_gt = batch["depth_filled_norm"].to(device, non_blocking=True)

        var_scale_pred = make_var_scale_prediction(depth_gt)
        model_input = torch.cat([rgb, var_scale_pred], dim=1)

        timesteps = torch.randint(
            0,
            args.num_train_timesteps,
            (model_input.shape[0],),
            device=device,
            dtype=torch.long,
        )

        pred_depth = model(sample=model_input, timestep=timesteps).sample

        loss_l1 = F.l1_loss(pred_depth, depth_gt)
        loss_mse = F.mse_loss(pred_depth, depth_gt)
        loss = loss_l1 + args.mse_weight * loss_mse

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
        optimizer.step()

        if step % args.log_interval == 0:
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "l1": f"{loss_l1.item():.4f}",
                    "mse": f"{loss_mse.item():.4f}",
                }
            )

        if step % args.vis_interval == 0:
            with torch.no_grad():
                vis = torch.cat(
                    [
                        rgb[:4].detach().cpu(),
                        var_scale_pred[:4].repeat(1, 3, 1, 1).detach().cpu(),
                        pred_depth[:4].repeat(1, 3, 1, 1).detach().cpu(),
                        depth_gt[:4].repeat(1, 3, 1, 1).detach().cpu(),
                    ],
                    dim=0,
                )
                save_data_batch(vis, os.path.join(args.out_dir, "train_preview.png"), grid=[4, 4])

        if step % args.save_interval == 0 and step > start_step:
            save_checkpoint(Path(args.out_dir), model, optimizer, step, args)

        step += 1
        pbar.update(1)

    pbar.close()
    save_checkpoint(Path(args.out_dir), model, optimizer, step - 1, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a diffusion UNet depth upsampler with RGB + var-scale depth input.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--mse_weight", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--vis_interval", type=int, default=200)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--out_dir", type=str, default="checkpoints/diffusion_depth_upsample")
    parser.add_argument("--resume", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    train(args)
