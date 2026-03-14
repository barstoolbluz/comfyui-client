#!/bin/bash
# FLUX Fast Demo - Quick demo optimized for speed

echo "⚡ ComfyUI Client - FLUX Fast Demo"
echo "===================================="
echo "Quick FLUX generation with reduced settings for demos"
echo

mkdir -p output/flux-fast

# Simpler prompt for faster generation
PROMPT="a beautiful mountain landscape at sunset, professional photography"

echo "📍 Generating FLUX image (fast settings)..."
echo "📍 Using: 512x512, 5 steps for quick demo"
echo

comfyui-submit ../workflows/api/flux/flux-txt2img.json \
  --prompt "$PROMPT" \
  --seed 42 \
  --steps 5 \
  --cfg 3.5 \
  --width 512 \
  --height 512 \
  --wait \
  --output ./output/flux-fast \
  --prefix "flux_fast"

echo
echo "✅ FLUX fast demo complete!"
echo "📁 Check output/flux-fast/"
echo "💡 For production quality, use more steps and higher resolution"