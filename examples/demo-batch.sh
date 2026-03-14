#!/bin/bash
# Batch processing demo - Process multiple prompts from JSON file

echo "🎨 ComfyUI Client - Batch Processing Demo"
echo "=========================================="
echo "Processing 8 different prompts from batch-prompts.json..."
echo

mkdir -p output/batch

echo "📄 Loading prompts from batch-prompts.json"
echo "🎯 Each prompt has custom settings (seed, steps, cfg)"
echo

# Process the batch file
comfyui-batch batch-prompts.json \
  --workflow ../workflows/api/sd15/sd15-txt2img.json \
  --output ./output/batch

echo
echo "✅ Batch processing complete!"
echo "📁 All 8 images saved to output/batch/"
echo "💡 Batch files are great for processing many different prompts with custom settings"