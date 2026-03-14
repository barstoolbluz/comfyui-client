#!/bin/bash
# Variations demo - Generate multiple variations with different seeds

echo "🎨 ComfyUI Client - Variations Demo"
echo "======================================"
echo "Generating 5 variations of the same prompt..."
echo

mkdir -p output/variations

PROMPT="a magical forest with glowing mushrooms, fairy lights, mystical atmosphere, fantasy art style"
NEGATIVE="ugly, tiling, poorly drawn, out of frame, mutation, mutated"

echo "📍 Prompt: $PROMPT"
echo "📍 Generating 5 variations with sequential mode..."
echo

# Generate 5 variations with different seeds (sequential mode for stability)
comfyui-submit ../workflows/api/sd15/sd15-txt2img.json \
  --prompt "$PROMPT" \
  --negative "$NEGATIVE" \
  --steps 20 \
  --cfg 8.0 \
  --width 512 \
  --height 512 \
  --count 5 \
  --wait \
  --output ./output/variations \
  --prefix "forest_variation"

echo
echo "✅ Generated 5 variations!"
echo "📁 Check output/variations/ for all images"
echo "💡 Each image has a different seed for variety"