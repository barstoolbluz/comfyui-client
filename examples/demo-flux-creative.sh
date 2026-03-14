#!/bin/bash
# FLUX Creative Showcase - Demonstrate FLUX's creative capabilities

echo "🎨 ComfyUI Client - FLUX Creative Showcase"
echo "==========================================="
echo "Generating 4 diverse creative pieces with FLUX"
echo "Each showcases different strengths of the model"
echo

mkdir -p output/flux-creative

echo "🎭 Generating 4 masterpieces sequentially..."
echo

# 1. Surreal Art
echo "1️⃣ Surreal Art Piece..."
comfyui-submit ../workflows/api/flux/flux-txt2img.json \
  --prompt "A giant teacup floating in a cloudy sky serving as a swimming pool for tiny businesspeople in suits, \
Salvador Dali inspired surrealism, melting clocks on the rim, \
photorealistic rendering with impossible physics, golden hour lighting" \
  --negative "boring, ordinary, low quality" \
  --seed 100 \
  --steps 20 \
  --cfg 4.0 \
  --width 1024 \
  --height 1024 \
  --wait \
  --output ./output/flux-creative \
  --prefix "flux_surreal"

# 2. Portrait Photography
echo "2️⃣ Professional Portrait..."
comfyui-submit ../workflows/api/flux/flux-txt2img.json \
  --prompt "Close-up portrait of an elderly storyteller with deeply weathered skin telling tales by firelight, \
eyes twinkling with wisdom and mischief, dramatic Rembrandt lighting, \
every wrinkle tells a story, photographic quality, shot on Hasselblad, \
shallow depth of field, warm color grading" \
  --negative "cartoon, painting, illustration" \
  --seed 200 \
  --steps 20 \
  --cfg 3.5 \
  --width 1024 \
  --height 1024 \
  --wait \
  --output ./output/flux-creative \
  --prefix "flux_portrait"

# 3. Architecture Visualization
echo "3️⃣ Architectural Marvel..."
comfyui-submit ../workflows/api/flux/flux-txt2img.json \
  --prompt "Futuristic sustainable treehouse city integrated into a giant ancient redwood forest, \
bio-luminescent pathways, living bridges made of woven branches, \
glass pods as homes, morning mist, rays of sunlight through the canopy, \
architectural photography, ultra detailed, octane render quality" \
  --negative "dark, gloomy, destroyed" \
  --seed 300 \
  --steps 25 \
  --cfg 3.5 \
  --width 1024 \
  --height 1024 \
  --wait \
  --output ./output/flux-creative \
  --prefix "flux_architecture"

# 4. Product Photography
echo "4️⃣ Product Shot..."
comfyui-submit ../workflows/api/flux/flux-txt2img.json \
  --prompt "Luxury watch floating in zero gravity surrounded by orbiting water droplets, \
each droplet perfectly reflecting the watch face, dramatic studio lighting, \
black background with subtle gradient, commercial photography, \
extreme macro detail on watch mechanics visible through sapphire crystal" \
  --negative "cheap, plastic, toy" \
  --seed 400 \
  --steps 20 \
  --cfg 3.5 \
  --width 1024 \
  --height 1024 \
  --wait \
  --output ./output/flux-creative \
  --prefix "flux_product"

echo
echo "✅ FLUX Creative Showcase complete!"
echo "🎨 4 diverse pieces generated:"
echo "   1. Surreal Art - Impossible made possible"
echo "   2. Portrait - Photographic quality with emotion"
echo "   3. Architecture - Complex environmental design"
echo "   4. Product - Commercial photography quality"
echo
echo "📁 All images in output/flux-creative/"
echo "💡 FLUX handles diverse styles with consistent high quality"