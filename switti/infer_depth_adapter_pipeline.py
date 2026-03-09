import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel
from PIL import Image
from torchvision import transforms as T
from torchvision.utils import save_image

from models import SwittiPipeline


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
        return cond_context, cond_vector

    def build_hybrid_prefix(self, static_prefix_scales, conditional_prefix_delta):
        gamma = torch.sigmoid(self.prefix_delta_logit)
        hybrid_prefix = static_prefix_scales.unsqueeze(0) + gamma * conditional_prefix_delta
        batch_prefix_context = hybrid_prefix.view(
            hybrid_prefix.shape[0], self.num_scales * self.prefix_len, self.context_dim
        )
        return batch_prefix_context


class DepthAdapterPipeline:
    def __init__(self, bundle_dir: Path, device: str = "cuda:0", reso: int = 512):
        self.bundle_dir = bundle_dir
        self.device = torch.device(device)
        self.reso = int(reso)

        config_path = bundle_dir / "pipeline_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing pipeline config: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        checkpoint_file = self.config.get("checkpoint_file", "adapter_prefix.pt")
        ckpt_path = bundle_dir / checkpoint_file
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

        base_model = self.config.get("base_model")
        if not base_model:
            raise ValueError("pipeline_config.json missing 'base_model'")

        self.pipe = SwittiPipeline.from_pretrained(
            base_model,
            device=str(self.device),
            torch_dtype=torch.float32,
            reso=self.reso,
        )
        self.model = self.pipe.switti.to(self.device)
        self.model.eval()
        self.pipe.switti = self.model
        self.pipe.vae = self.pipe.vae.to(self.device)

        for param in self.model.parameters():
            param.requires_grad = False

        self.ckpt = torch.load(ckpt_path, map_location="cpu")

        static_prefix = self.ckpt.get("static_prefix_context_scales", self.ckpt.get("prefix_context_scales"))
        if static_prefix is None:
            raise ValueError("Checkpoint missing prefix tensor")
        self.static_prefix_context_scales = static_prefix.to(self.device)

        self.prefix_len = int(self.config.get("prefix_len", self.static_prefix_context_scales.shape[1]))
        self.num_scales = int(self.config.get("num_scales", self.static_prefix_context_scales.shape[0]))
        self.context_dim = int(self.config.get("context_dim", self.static_prefix_context_scales.shape[-1]))
        self.prefix_rank = int(self.config.get("conditional_prefix_rank", 128))
        self.prefix_mode = self.config.get("prefix_mode", self.ckpt.get("prefix_mode", "static"))

        prompt_embeds, pooled_prompt_embeds, prompt_attn_bias = self.pipe.encode_prompt(
            ["a photo of a street"], encode_null=False
        )
        self.prompt_embeds = prompt_embeds.to(self.device)
        self.pooled_prompt_embeds = pooled_prompt_embeds.to(self.device)
        self.prompt_attn_bias = prompt_attn_bias.to(self.device)

        self.adapter = RGBConditionAdapter(
            seq_len=self.prompt_embeds.shape[1],
            context_dim=self.prompt_embeds.shape[-1],
            pooled_dim=self.pooled_prompt_embeds.shape[-1],
            num_scales=self.num_scales,
            prefix_len=self.prefix_len,
            prefix_rank=self.prefix_rank,
        ).to(self.device)
        self.adapter.eval()

        adapter_state = self.ckpt.get("adapter_state_dict")
        if adapter_state is None:
            raise ValueError("Checkpoint missing 'adapter_state_dict'")
        self.adapter.load_state_dict(adapter_state, strict=False)

    @torch.inference_mode()
    def predict(self, rgb_B3HW: torch.Tensor, prompt: str = "a photo of a street") -> torch.Tensor:
        batch_size = rgb_B3HW.shape[0]
        rgb_B3HW = rgb_B3HW.to(self.device)

        if prompt:
            p_emb, p_pool, p_bias = self.pipe.encode_prompt([prompt] * batch_size, encode_null=False)
            text_context = p_emb.to(self.device)
            text_vector = p_pool.to(self.device)
            text_attn_bias = p_bias.to(self.device)
        else:
            text_context = self.prompt_embeds.expand(batch_size, -1, -1)
            text_vector = self.pooled_prompt_embeds.expand(batch_size, -1)
            text_attn_bias = self.prompt_attn_bias.expand(batch_size, -1)

        rgb_context, rgb_vector, prefix_delta = self.adapter(rgb_B3HW)
        cond_context, cond_vector = self.adapter.blend(
            text_context,
            text_vector,
            rgb_context,
            rgb_vector,
        )

        if self.prefix_mode == "hybrid_conditional":
            batch_prefix_context = self.adapter.build_hybrid_prefix(
                static_prefix_scales=self.static_prefix_context_scales,
                conditional_prefix_delta=prefix_delta,
            )
        else:
            flat_prefix_context = self.static_prefix_context_scales.reshape(
                1, self.num_scales * self.prefix_len, self.context_dim
            )
            batch_prefix_context = flat_prefix_context.expand(batch_size, -1, -1)

        cond_context_with_prefix = torch.cat([batch_prefix_context, cond_context], dim=1)
        prefix_attn_bias = torch.zeros(
            batch_size,
            self.num_scales * self.prefix_len,
            device=self.device,
            dtype=text_attn_bias.dtype,
        )
        cond_attn_bias_with_prefix = torch.cat([prefix_attn_bias, text_attn_bias], dim=1)

        out = self.pipe(
            [prompt] * batch_size,
            cond_context=cond_context_with_prefix,
            condition_vector=cond_vector,
            cond_attn_bias=cond_attn_bias_with_prefix,
            return_pil=False,
            cfg=0.0,
            turn_off_cfg_start_si=0,
        )
        out = (out * 2) - 1
        return out


class DepthUpsampler:
    def __init__(self, checkpoint_path: Path, device: str = "cuda:0", height: int = 384, width: int = 640):
        self.device = torch.device(device)
        self.height = int(height)
        self.width = int(width)

        self.model = UNet2DModel(
            sample_size=(self.height, self.width),
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
        ).to(self.device)

        payload = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, rgb_B3HW: torch.Tensor, coarse_depth_B1HW: torch.Tensor) -> torch.Tensor:
        rgb = rgb_B3HW.to(self.device)
        coarse_depth = coarse_depth_B1HW.to(self.device)

        if rgb.shape[-2:] != (self.height, self.width):
            rgb = F.interpolate(rgb, size=(self.height, self.width), mode="bilinear", align_corners=False)
        if coarse_depth.shape[-2:] != (self.height, self.width):
            coarse_depth = F.interpolate(
                coarse_depth,
                size=(self.height, self.width),
                mode="bilinear",
                align_corners=False,
            )

        model_input = torch.cat([rgb, coarse_depth], dim=1)
        timesteps = torch.zeros((model_input.shape[0],), device=self.device, dtype=torch.long)
        pred_depth = self.model(sample=model_input, timestep=timesteps).sample
        return pred_depth


def load_image_as_model_tensor(image_path: Path, resolution: int) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.Resize((resolution, resolution)),
        T.ToTensor(),
    ])
    tensor = transform(img)
    tensor = tensor * 2.0 - 1.0
    return tensor.unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description="Run Switti depth adapter pipeline bundle inference")
    parser.add_argument("--bundle-dir", type=str, required=True, help="Path containing pipeline_config.json and checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Input RGB image path")
    parser.add_argument("--output", type=str, default="depth_pred.png", help="Output image path")
    parser.add_argument("--prompt", type=str, default="a photo of a street", help="Prompt for text conditioning")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device, e.g. cuda:0 or cpu")
    parser.add_argument("--reso", type=int, default=512, help="Inference resolution")
    parser.add_argument("--upsampler-ckpt", type=str, default="", help="Optional diffusion upsampler checkpoint")
    parser.add_argument("--upsampler-height", type=int, default=384, help="Upsampler height")
    parser.add_argument("--upsampler-width", type=int, default=640, help="Upsampler width")
    parser.add_argument("--save-stage1", type=str, default="", help="Optional path to save stage 1 output")
    args = parser.parse_args()

    bundle_dir = Path(args.bundle_dir)
    image_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    runner = DepthAdapterPipeline(bundle_dir=bundle_dir, device=args.device, reso=args.reso)
    rgb_B3HW = load_image_as_model_tensor(image_path=image_path, resolution=args.reso)
    pred_stage1 = runner.predict(rgb_B3HW=rgb_B3HW, prompt=args.prompt)

    if args.save_stage1:
        stage1_path = Path(args.save_stage1)
        stage1_path.parent.mkdir(parents=True, exist_ok=True)
        stage1_vis = pred_stage1.clamp(-1, 1)
        stage1_vis = (stage1_vis + 1.0) / 2.0
        save_image(stage1_vis, str(stage1_path))

    if args.upsampler_ckpt:
        upsampler = DepthUpsampler(
            checkpoint_path=Path(args.upsampler_ckpt),
            device=args.device,
            height=args.upsampler_height,
            width=args.upsampler_width,
        )
        coarse_depth = pred_stage1.mean(dim=1, keepdim=True)
        pred = upsampler.predict(rgb_B3HW=rgb_B3HW, coarse_depth_B1HW=coarse_depth)
        pred_vis = pred.clamp(-1, 1)
        pred_vis = (pred_vis + 1.0) / 2.0
    else:
        pred_vis = pred_stage1.clamp(-1, 1)
        pred_vis = (pred_vis + 1.0) / 2.0

    save_image(pred_vis, str(output_path))
    print(f"saved prediction to {output_path}")


if __name__ == "__main__":
    main()
