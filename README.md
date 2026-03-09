# VAR-Depth
Official codebase for "Visual Autoregressive Modelling for Monocular Depth Estimation".
Paper: https://arxiv.org/pdf/2512.22653

## Training (Two-Stage)
Stage 1 fine-tunes the Switti depth prior with RGB conditioning.
Stage 2 trains the conditional diffusion upsampler with RGB + coarse depth input.

```bash
# Run both stages in order
python train_main.py --stage both

# Run only stage 1
python train_main.py --stage 1

# Run only stage 2
python train_main.py --stage 2
```

### Stage 1: Switti depth prior
```bash
python switti/train_depth.py \
	--version v2 \
	--save 500 \
	--samples 25
```

### Stage 2: Diffusion upsampler
```bash
python switti/train_depth_diffusion_upsample.py \
	--height 384 \
	--width 640 \
	--batch_size 4 \
	--max_steps 100000
```

## Inference
Use the depth adapter pipeline (stage 1) and optionally the diffusion upsampler (stage 2).

```bash
python switti/infer_depth_adapter_pipeline.py \
	--bundle-dir checkpoints/depth_adapter_pipeline/v2 \
	--input path/to/rgb.png \
	--output depth_pred.png \
	--prompt "a photo of a street"
```

```bash
python switti/infer_depth_adapter_pipeline.py \
	--bundle-dir checkpoints/depth_adapter_pipeline/v2 \
	--input path/to/rgb.png \
	--output depth_pred_upsampled.png \
	--upsampler-ckpt checkpoints/diffusion_depth_upsample/ckpt_step_0001000.pt \
	--upsampler-height 384 \
	--upsampler-width 640
```

## Notes
- These scripts depend on the internal Framework dataset loaders used in the original Switti code.
- Ensure the Framework module is available on your PYTHONPATH when running training.

## Next Steps
- Push checkpoints for both Stage 1 (depth prior) and Stage 2 (diffusion upsampler).
