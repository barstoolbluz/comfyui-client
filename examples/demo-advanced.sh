#!/bin/bash
# Advanced demo script for ComfyUI client - Multiple models comparison

echo "🎨 ComfyUI Client - Advanced Demo"
echo "==================================="
echo "Comparing the same prompt across different models..."
echo

# Create output directories
mkdir -p output/sd15 output/sdxl output/sd35

# The prompt we'll use for all models
PROMPT="a cyberpunk city street at night, neon lights, rain reflections, flying cars, blade runner atmosphere"
NEGATIVE="blurry, low quality, ugly, distorted, malformed"
SEED=1337

echo "📍 Prompt: $PROMPT"
echo "📍 Using seed: $SEED for consistency"
echo

# SD 1.5 - Fast, lower quality
echo "1️⃣ Generating with Stable Diffusion 1.5 (512x512)..."
comfyui-submit ../workflows/api/sd15/sd15-txt2img.json \
  --prompt "$PROMPT" \
  --negative "$NEGATIVE" \
  --seed $SEED \
  --steps 20 \
  --cfg 7.5 \
  --width 512 \
  --height 512 \
  --wait \
  --output ./output/sd15 \
  --prefix "sd15_cyberpunk"

# SDXL - Higher quality, slower
echo "2️⃣ Generating with Stable Diffusion XL (1024x1024)..."
comfyui-submit ../workflows/api/sdxl/sdxl-txt2img.json \
  --prompt "$PROMPT" \
  --negative "$NEGATIVE" \
  --seed $SEED \
  --steps 25 \
  --cfg 7.0 \
  --width 1024 \
  --height 1024 \
  --wait \
  --output ./output/sdxl \
  --prefix "sdxl_cyberpunk"

# SD 3.5 - Latest model
echo "3️⃣ Generating with Stable Diffusion 3.5 (1024x1024)..."
comfyui-submit ../workflows/api/sd35/sd35-txt2img.json \
  --prompt "$PROMPT" \
  --negative "$NEGATIVE" \
  --seed $SEED \
  --steps 28 \
  --cfg 7.0 \
  --width 1024 \
  --height 1024 \
  --wait \
  --output ./output/sd35 \
  --prefix "sd35_cyberpunk"

echo
echo "✅ All images generated!"
echo "📁 Compare results in:"
echo "   - output/sd15/ (fastest, lower quality)"
echo "   - output/sdxl/ (balanced quality/speed)"
echo "   - output/sd35/ (latest model)"