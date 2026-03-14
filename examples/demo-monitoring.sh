#!/bin/bash
# Queue monitoring demo - Submit jobs and monitor the queue

echo "🎨 ComfyUI Client - Queue Monitoring Demo"
echo "=========================================="
echo "This demo shows how to monitor job progress"
echo

mkdir -p output/monitoring

# Submit jobs without waiting (so we can monitor them)
echo "📤 Submitting 3 jobs to the queue (without --wait)..."
echo

echo "Job 1: Landscape..."
PROMPT_ID_1=$(comfyui-submit ../workflows/api/sd15/sd15-txt2img.json \
  --prompt "mountain landscape" \
  --seed 111 \
  --output ./output/monitoring \
  --prefix "job1" \
  2>&1 | grep -oP 'Prompt ID: \K[a-f0-9-]+')

echo "Job 2: Portrait..."
PROMPT_ID_2=$(comfyui-submit ../workflows/api/sd15/sd15-txt2img.json \
  --prompt "portrait of a wizard" \
  --seed 222 \
  --output ./output/monitoring \
  --prefix "job2" \
  2>&1 | grep -oP 'Prompt ID: \K[a-f0-9-]+')

echo "Job 3: Abstract art..."
PROMPT_ID_3=$(comfyui-submit ../workflows/api/sd15/sd15-txt2img.json \
  --prompt "abstract colorful patterns" \
  --seed 333 \
  --output ./output/monitoring \
  --prefix "job3" \
  2>&1 | grep -oP 'Prompt ID: \K[a-f0-9-]+')

echo
echo "📊 Checking queue status..."
comfyui-queue

echo
echo "👀 To monitor in real-time, run: comfyui-status --watch"
echo "❌ To cancel a job, run: comfyui-cancel <prompt_id>"
echo "📥 To get results manually, run: comfyui-result <prompt_id> -o ./output/monitoring"
echo
echo "💡 Tip: Open another terminal and run 'comfyui-status --watch' to see live updates!"