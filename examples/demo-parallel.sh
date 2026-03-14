#!/bin/bash
# Parallel generation demo - Submit multiple jobs at once

echo "🎨 ComfyUI Client - Parallel Generation Demo"
echo "============================================="
echo "Submitting 4 jobs in parallel for faster generation..."
echo

mkdir -p output/parallel

PROMPT="a steampunk airship flying through clouds, brass and copper details, victorian era, detailed mechanical parts"
NEGATIVE="modern, contemporary, blurry, low quality"

echo "📍 Prompt: $PROMPT"
echo "📍 Submitting 4 jobs in parallel..."
echo "⚡ This will use more GPU resources but complete faster overall"
echo

# Submit 4 images in parallel mode
comfyui-submit ../workflows/api/sd15/sd15-txt2img.json \
  --prompt "$PROMPT" \
  --negative "$NEGATIVE" \
  --steps 20 \
  --cfg 7.5 \
  --width 512 \
  --height 512 \
  --count 4 \
  --parallel \
  --wait \
  --output ./output/parallel \
  --prefix "steampunk"

echo
echo "✅ All 4 images completed!"
echo "📁 Results saved to output/parallel/"
echo "💡 Parallel mode is great when your GPU can handle multiple jobs"