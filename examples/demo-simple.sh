#!/bin/bash
# Simple demo script for ComfyUI client - Single image generation

echo "🎨 ComfyUI Client - Simple Demo"
echo "================================"
echo "Generating a single image with Stable Diffusion 1.5..."
echo

# Generate a single image with SD 1.5 (fastest model)
comfyui-submit ../workflows/api/sd15/sd15-txt2img.json \
  --prompt "a majestic mountain landscape at sunset, snow-capped peaks, golden hour lighting, photorealistic" \
  --negative "blurry, low quality, distorted" \
  --seed 42 \
  --steps 20 \
  --cfg 7.5 \
  --width 512 \
  --height 512 \
  --wait \
  --output ./output

echo
echo "✅ Image generated and saved to ./output/"
echo "Check the output directory for your image!"