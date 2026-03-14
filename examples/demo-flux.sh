#!/bin/bash
# FLUX demo - Showcase the most powerful model

echo "🚀 ComfyUI Client - FLUX Demo"
echo "================================"
echo "Generating with FLUX - The most powerful model"
echo "⚠️  Note: FLUX is slower but produces the highest quality"
echo

mkdir -p output/flux

# FLUX excels at complex prompts with natural language
PROMPT="A serene moment captured in a cozy bookshop cafe during golden hour, \
warm sunlight streaming through large windows casting long shadows across worn wooden floors, \
steam rising from a ceramic coffee cup on a table covered with open books and handwritten notes, \
vintage leather armchairs, shelves filled with old books reaching the ceiling, \
a cat sleeping on a windowsill, dust particles dancing in the light beams, \
photorealistic, extraordinary detail, cinematic lighting"

NEGATIVE="blurry, low quality, distorted, ugly, malformed"

echo "📍 Generating high-quality FLUX image..."
echo "📍 This may take 1-2 minutes depending on your GPU"
echo

comfyui-submit ../workflows/api/flux/flux-txt2img.json \
  --prompt "$PROMPT" \
  --negative "$NEGATIVE" \
  --seed 42 \
  --steps 10 \
  --cfg 3.5 \
  --width 1024 \
  --height 1024 \
  --wait \
  --output ./output/flux \
  --prefix "flux_masterpiece"

echo
echo "✅ FLUX image generated!"
echo "📁 Check output/flux/ for your masterpiece"
echo "💡 FLUX excels at complex, natural language prompts"