#!/bin/bash
# FLUX Quality Comparison - Compare FLUX with other models

echo "🚀 ComfyUI Client - FLUX Quality Comparison"
echo "============================================="
echo "Comparing FLUX against SD 1.5 and SDXL with the same prompt"
echo

# Create output directories
mkdir -p output/comparison/{sd15,sdxl,flux}

# A prompt that showcases FLUX's superior understanding
PROMPT="A master chef in a professional kitchen preparing an elaborate dish, \
ingredients flying through the air in perfect arcs, dramatic lighting from overhead, \
steam and flames visible, stainless steel surfaces reflecting the action, \
photographic quality, professional food photography style, shallow depth of field"

NEGATIVE="cartoon, illustration, anime, low quality"
SEED=777

echo "📍 Using identical prompt for all models:"
echo "   '$PROMPT'"
echo "📍 Seed: $SEED for consistency"
echo

# SD 1.5 - Baseline
echo "1️⃣ Generating with SD 1.5 (fastest, baseline quality)..."
comfyui-submit ../workflows/api/sd15/sd15-txt2img.json \
  --prompt "$PROMPT" \
  --negative "$NEGATIVE" \
  --seed $SEED \
  --steps 20 \
  --cfg 7.5 \
  --width 512 \
  --height 512 \
  --wait \
  --output ./output/comparison/sd15 \
  --prefix "comparison"

# SDXL - Mid-tier
echo "2️⃣ Generating with SDXL (balanced quality/speed)..."
comfyui-submit ../workflows/api/sdxl/sdxl-txt2img.json \
  --prompt "$PROMPT" \
  --negative "$NEGATIVE" \
  --seed $SEED \
  --steps 25 \
  --cfg 7.0 \
  --width 1024 \
  --height 1024 \
  --wait \
  --output ./output/comparison/sdxl \
  --prefix "comparison"

# FLUX - Highest quality
echo "3️⃣ Generating with FLUX (highest quality, worth the wait)..."
comfyui-submit ../workflows/api/flux/flux-txt2img.json \
  --prompt "$PROMPT" \
  --negative "$NEGATIVE" \
  --seed $SEED \
  --steps 20 \
  --cfg 3.5 \
  --width 1024 \
  --height 1024 \
  --wait \
  --output ./output/comparison/flux \
  --prefix "comparison"

echo
echo "✅ All comparisons complete!"
echo "📊 Compare the results:"
echo "   - output/comparison/sd15/  (baseline)"
echo "   - output/comparison/sdxl/  (good quality)"
echo "   - output/comparison/flux/  (best quality)"
echo
echo "💡 Notice FLUX's superior:"
echo "   - Understanding of complex prompts"
echo "   - Photorealistic quality"
echo "   - Coherent composition"
echo "   - Fine detail rendering"