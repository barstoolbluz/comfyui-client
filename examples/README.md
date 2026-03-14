# ComfyUI Client - Example Scripts

This directory contains demo scripts showcasing different features of the ComfyUI client. Each script demonstrates a specific use case or feature.

## Prerequisites

Before running these examples:

1. Ensure ComfyUI server is running (default: `localhost:8188`)
2. Activate the Flox environment: `flox activate`

## Note on Workflow Files

The workflow files in `../workflows/api/` don't have `.json` extensions but are valid JSON files.

## Available Demo Scripts

### 1. Simple Demo (`demo-simple.sh`)
- **Purpose**: Quick start with a single image generation
- **Features**: Basic prompt with negative prompt, fixed seed
- **Model**: SD 1.5 (fastest)
- **Output**: Single 512x512 image

```bash
./demo-simple.sh
```

### 2. Advanced Demo (`demo-advanced.sh`)
- **Purpose**: Compare same prompt across different models
- **Features**: Tests SD 1.5, SDXL, and SD 3.5 with same seed
- **Model**: Multiple (SD 1.5, SDXL, SD 3.5)
- **Output**: 3 images in separate directories for comparison

```bash
./demo-advanced.sh
```

### 3. Variations Demo (`demo-variations.sh`)
- **Purpose**: Generate multiple variations with different seeds
- **Features**: Sequential generation of 5 images
- **Model**: SD 1.5
- **Output**: 5 variations with different seeds

```bash
./demo-variations.sh
```

### 4. Parallel Demo (`demo-parallel.sh`)
- **Purpose**: Demonstrate parallel job submission
- **Features**: Submits 4 jobs simultaneously
- **Model**: SD 1.5
- **Output**: 4 images generated in parallel
- **Note**: Uses more GPU resources but completes faster

```bash
./demo-parallel.sh
```

### 5. Batch Processing Demo (`demo-batch.sh`)
- **Purpose**: Process multiple prompts from JSON file
- **Features**: 8 different prompts with custom settings each
- **Model**: SD 1.5
- **Input**: `batch-prompts.json`
- **Output**: 8 different themed images

```bash
./demo-batch.sh
```

### 6. Queue Monitoring Demo (`demo-monitoring.sh`)
- **Purpose**: Show how to monitor and manage jobs
- **Features**: Submit jobs and check queue status
- **Model**: SD 1.5
- **Output**: 3 images (demonstrates monitoring, not waiting)

```bash
./demo-monitoring.sh
# Then in another terminal: comfyui-status --watch
```

## Batch Prompts File

`batch-prompts.json` contains 8 different themed prompts:
1. Japanese garden
2. Fantasy dragon
3. Underwater coral reef
4. Cozy coffee shop
5. Astronaut in space
6. Egyptian temple
7. Cyberpunk robot
8. Lion portrait

Each prompt includes custom settings for seed, steps, CFG scale, and dimensions.

## Output Structure

All demos create organized output directories:
```
output/
├── sd15/       # Model-specific outputs
├── sdxl/
├── sd35/
├── variations/ # Multiple variations
├── parallel/   # Parallel generation results
├── batch/      # Batch processing results
└── monitoring/ # Queue monitoring demo results
```

## Tips for Demos

1. **Start with `demo-simple.sh`** - Fastest way to verify everything works
2. **Use `demo-variations.sh`** - Great for showing seed variation effects
3. **Run `demo-parallel.sh`** - Demonstrates GPU utilization efficiency
4. **Try `demo-batch.sh`** - Shows production-like batch processing
5. **Compare with `demo-advanced.sh`** - Visual quality comparison across models

## Customization

Feel free to modify these scripts:
- Change prompts to match your demo theme
- Adjust steps/CFG for quality vs speed tradeoffs
- Modify output directories
- Add your own batch prompts to `batch-prompts.json`

## Performance Notes

- **SD 1.5**: Fastest, 512x512, good for demos
- **SDXL**: Slower, 1024x1024, higher quality
- **SD 3.5**: Latest model, varies by configuration
- **Parallel mode**: Faster overall but uses more VRAM
- **Sequential mode**: More stable, lower resource usage

## Troubleshooting

If scripts fail:
1. Check ComfyUI server is running: `comfyui-status`
2. Verify models are available: `comfyui-models checkpoints`
3. Check GPU memory: `comfyui-status` (shows VRAM)
4. Reduce image size or batch count if out of memory
5. Use sequential mode instead of parallel for limited VRAM