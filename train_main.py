import argparse
from pathlib import Path

from switti import train_depth as stage1
from switti import train_depth_diffusion_upsample as stage2


def run_stage1(args: argparse.Namespace) -> None:
    """Stage 1: fine-tune the Switti depth prior with RGB conditioning."""
    stage1.train(
        restart_from_beginning=args.stage1_restart,
        eval_interval=args.stage1_eval_interval,
        save_interval=args.stage1_save_interval,
        supervised_samples=args.stage1_samples,
        train_version=args.stage1_version,
    )


def run_stage2(args: argparse.Namespace) -> None:
    """Stage 2: conditional upsampling using RGB + coarse depth input."""
    stage2_args = argparse.Namespace(
        device=args.stage2_device,
        seed=args.stage2_seed,
        height=args.stage2_height,
        width=args.stage2_width,
        batch_size=args.stage2_batch_size,
        num_workers=args.stage2_num_workers,
        max_steps=args.stage2_max_steps,
        num_train_timesteps=args.stage2_num_train_timesteps,
        lr=args.stage2_lr,
        weight_decay=args.stage2_weight_decay,
        mse_weight=args.stage2_mse_weight,
        max_grad_norm=args.stage2_max_grad_norm,
        log_interval=args.stage2_log_interval,
        vis_interval=args.stage2_vis_interval,
        save_interval=args.stage2_save_interval,
        out_dir=args.stage2_out_dir,
        resume=args.stage2_resume,
    )
    Path(stage2_args.out_dir).mkdir(parents=True, exist_ok=True)
    stage2.train(stage2_args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Two-stage depth training: stage 1 fine-tunes the Switti depth prior, "
            "stage 2 trains the conditional diffusion upsampler."
        )
    )
    parser.add_argument(
        "--stage",
        choices=["1", "2", "both"],
        default="both",
        help="Which stage to run.",
    )

    # Stage 1 options (Switti depth prior).
    parser.add_argument("--stage1-restart", action="store_true", help="Restart stage 1 from step 0.")
    parser.add_argument("--stage1-eval-interval", type=int, default=-1, help="Eval interval in steps.")
    parser.add_argument("--stage1-save-interval", type=int, default=500, help="Checkpoint save interval.")
    parser.add_argument("--stage1-samples", type=int, default=25, help="Supervised sample count.")
    parser.add_argument(
        "--stage1-version",
        choices=["v1", "v2", "v3"],
        default="v2",
        help="Switti depth prior training mode.",
    )

    # Stage 2 options (conditional diffusion upsampler).
    parser.add_argument("--stage2-device", type=str, default="cuda:0")
    parser.add_argument("--stage2-seed", type=int, default=0)
    parser.add_argument("--stage2-height", type=int, default=384)
    parser.add_argument("--stage2-width", type=int, default=640)
    parser.add_argument("--stage2-batch-size", type=int, default=4)
    parser.add_argument("--stage2-num-workers", type=int, default=4)
    parser.add_argument("--stage2-max-steps", type=int, default=100000)
    parser.add_argument("--stage2-num-train-timesteps", type=int, default=1000)
    parser.add_argument("--stage2-lr", type=float, default=1e-4)
    parser.add_argument("--stage2-weight-decay", type=float, default=1e-4)
    parser.add_argument("--stage2-mse-weight", type=float, default=0.5)
    parser.add_argument("--stage2-max-grad-norm", type=float, default=1.0)
    parser.add_argument("--stage2-log-interval", type=int, default=20)
    parser.add_argument("--stage2-vis-interval", type=int, default=200)
    parser.add_argument("--stage2-save-interval", type=int, default=1000)
    parser.add_argument(
        "--stage2-out-dir",
        type=str,
        default="checkpoints/diffusion_depth_upsample",
    )
    parser.add_argument("--stage2-resume", type=str, default="")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.stage in {"1", "both"}:
        run_stage1(args)
    if args.stage in {"2", "both"}:
        run_stage2(args)


if __name__ == "__main__":
    main()
