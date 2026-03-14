# ComfyUI Client - Flox Environment

A powerful command-line interface for interacting with ComfyUI servers, packaged as a reproducible Flox environment. Submit workflows, monitor progress, and manage your AI image generation pipeline from the terminal.

## 🎯 Perfect Companion to ComfyUI-Complete

This client environment perfectly complements the **ComfyUI-Complete** server environment:

```bash
# Terminal 1: Start ComfyUI server with all models
flox pull --copy barstoolbluz/comfyui-complete
flox activate --start-services

# Terminal 2: Use this client to interact with the server
flox pull --copy barstoolbluz/comfyui-client
flox activate
comfyui-submit workflows/api/flux.json -p "amazing artwork" -w
```

The `comfyui-complete` environment provides a fully-configured ComfyUI server with models, while this `comfyui-client` environment provides the CLI tools to interact with it.

## 🚀 Quick Start

```bash
# Clone and activate the environment
git clone https://github.com/barstoolbluz/comfyui-client
cd comfyui-client
flox activate

# Submit your first workflow (assumes ComfyUI running on localhost:8188)
comfyui-submit workflows/api/sd15.json -p "a beautiful sunset" -w
```

## 📋 Prerequisites

- **Flox**: Install from [flox.dev](https://flox.dev)
- **ComfyUI Server**: Running instance (local or remote)
  - Default: `http://localhost:8188`
  - Or set `COMFYUI_HOST` and `COMFYUI_PORT` for remote servers

## 🔧 Installation

### Local Development Environment

```bash
# Initialize from this repository
cd comfyui-client
flox activate

# The environment automatically:
# - Creates a Python virtual environment
# - Installs the CLI tools
# - Sets up workflow symlinks
```

### From FloxHub (Published Version)

```bash
# Pull the environment from FloxHub
flox pull --copy barstoolbluz/comfyui-client

# Activate it
flox activate
```

### Remote Server Connection

```bash
# Connect to a remote ComfyUI instance
COMFYUI_HOST=192.168.1.100 COMFYUI_PORT=8188 flox activate

# These settings persist for your session
comfyui-info  # Will connect to the remote server
```

## 🛠️ Core Features

### CLI Tools Suite

The environment provides 8 specialized command-line tools:

| Tool | Purpose | Example |
|------|---------|---------|
| `comfyui-submit` | Submit workflows with parameters | `comfyui-submit workflow.json -p "cat" -w` |
| `comfyui-batch` | Run multiple jobs from batch file | `comfyui-batch jobs.json -W workflow.json` |
| `comfyui-status` | Monitor queue and job status | `comfyui-status --watch` |
| `comfyui-queue` | Show current queue status | `comfyui-queue` |
| `comfyui-cancel` | Cancel running/queued jobs | `comfyui-cancel <prompt_id>` |
| `comfyui-result` | Get result for a prompt ID | `comfyui-result <prompt_id> -o ./output` |
| `comfyui-models` | List available models | `comfyui-models checkpoints` |
| `comfyui-info` | Show server configuration | `comfyui-info` |

### Bundled Workflows

Pre-configured workflows for popular models:

- `workflows/api/sd15.json` - Stable Diffusion 1.5
- `workflows/api/sdxl.json` - Stable Diffusion XL
- `workflows/api/sd35.json` - Stable Diffusion 3.5
- `workflows/api/flux.json` - FLUX models
- `workflows/api/img2img.json` - Image-to-image workflows

## 📝 Configuration

### Environment Variables

Configure the client behavior through environment variables:

```bash
# Server connection (defaults shown)
export COMFYUI_HOST="localhost"
export COMFYUI_PORT="8188"

# Custom workflow directory (optional)
export COMFYUI_WORKFLOWS="$HOME/my-workflows"

# Activate with custom settings
COMFYUI_HOST=remote.server.com flox activate
```

### Persistent Configuration

Edit `.flox/env/manifest.toml` to change defaults:

```toml
[vars]
# Uncomment to set a custom workflow directory
COMFYUI_WORKFLOWS = "$HOME/comfyui-work/user/default/workflows"
```

### Shell Completions

Enable tab completion for all CLI commands:

```bash
# Bash
comfyui-submit --install-completion

# Or manually add to .bashrc
eval "$(_COMFYUI_SUBMIT_COMPLETE=bash_source comfyui-submit)"

# Zsh
comfyui-submit --install-completion

# Fish
comfyui-submit --install-completion

# After installation, restart your shell or source your config:
source ~/.bashrc  # or ~/.zshrc
```

Completions work for all tools and provide:
- Command and option completion
- File path completion for workflows and images
- Model type suggestions for `comfyui-models`
- Available scheduler/sampler names

## 🎯 Command Reference

### comfyui-submit

Submit workflows to the ComfyUI server with customizable parameters.

```bash
# Basic usage
comfyui-submit workflows/api/sd15.json -p "a majestic mountain"

# Full parameter control
comfyui-submit workflows/api/sdxl.json \
  --prompt "cyberpunk city" \
  --negative "blurry, low quality" \
  --seed 42 \
  --steps 30 \
  --cfg 7.5 \
  --width 1024 \
  --height 1024 \
  --wait \
  --output ./output

# Image-to-image generation
comfyui-submit workflows/api/img2img.json \
  --image input.png \
  --prompt "make it sunset" \
  --denoise 0.5 \
  --wait

# Batch generation with varied seeds
comfyui-submit workflows/api/flux.json \
  --prompt "portrait of a robot" \
  --count 4 \
  --parallel \
  --prefix "robot_" \
  --output ./batch_output
```

#### Batch Generation Modes

The client supports two batch generation strategies:

**Sequential Mode (default):**
```bash
# Submits jobs one at a time, waits for each to complete
comfyui-submit workflow.json -p "landscape" --count 5 --wait
# Advantage: Lower server load, predictable resource usage
# Use when: Server has limited resources or you want controlled generation
```

**Parallel Mode:**
```bash
# Submits all jobs at once to the queue
comfyui-submit workflow.json -p "landscape" --count 5 --parallel --wait
# Advantage: Faster overall completion if server can handle it
# Use when: Server has multiple GPUs or high capacity
```

**How Batch Generation Works:**
1. Takes your base parameters (prompt, steps, cfg, etc.)
2. Generates unique seeds for each image (unless seed is fixed)
3. Creates separate jobs with incremented seeds
4. In sequential mode: Submits job 1, waits, submits job 2, etc.
5. In parallel mode: Submits all jobs immediately to queue
6. With `--wait`: Downloads all images as they complete
7. Files are named: `{prefix}_{seed}_{timestamp}.png`

**Options:**
- `-p, --prompt TEXT`: Positive prompt for generation
- `-n, --negative TEXT`: Negative prompt
- `-s, --seed INTEGER`: Random seed (default: random)
- `--steps INTEGER`: Sampling steps
- `--cfg FLOAT`: CFG scale for guidance
- `-W, --width INTEGER`: Output width
- `-H, --height INTEGER`: Output height
- `-d, --denoise FLOAT`: Denoise strength (0.0-1.0) for img2img
- `--sampler TEXT`: Sampler algorithm (euler, dpmpp_2m, etc.)
- `--scheduler TEXT`: Scheduler type (normal, karras, etc.)
- `-i, --image PATH`: Input image for img2img workflows
- `-w, --wait`: Wait for completion and download results
- `-o, --output PATH`: Output directory for images
- `-c, --count INTEGER`: Generate multiple images with varied seeds
- `--parallel`: Submit all jobs simultaneously
- `--prefix TEXT`: Filename prefix for outputs

### comfyui-status

Monitor the ComfyUI queue and job status.

```bash
# Check current queue
comfyui-status

# Watch queue in real-time (refreshes every 2 seconds)
comfyui-status --watch

# Custom refresh interval
comfyui-status --watch --interval 5
```

### comfyui-cancel

Cancel running or queued generation jobs.

```bash
# Cancel a specific job
comfyui-cancel 12345678-90ab-cdef-1234-567890abcdef

# Cancel all queued jobs
comfyui-cancel --all

# Cancel currently running job
comfyui-cancel --current
```

### comfyui-models

List available models by category.

```bash
# List all model categories
comfyui-models

# List specific model type
comfyui-models checkpoints
comfyui-models loras
comfyui-models vae
comfyui-models embeddings
comfyui-models controlnet

# Filter results
comfyui-models checkpoints | grep SD
```

### comfyui-queue

Display current queue status without continuous monitoring.

```bash
# Show queue status
comfyui-queue

# Shows:
# - Running jobs
# - Pending jobs
# - Queue position for each job
```

### comfyui-result

Download results for a completed generation.

```bash
# Get result by prompt ID
comfyui-result 12345678-90ab-cdef-1234-567890abcdef

# Save to specific directory
comfyui-result <prompt_id> --output ./results

# Get result with original metadata
comfyui-result <prompt_id> --with-metadata
```

### comfyui-batch

Process multiple generation jobs from a JSON batch file.

```bash
# Create a batch file (jobs.json)
cat > jobs.json << 'EOF'
[
  {"prompt": "sunset over mountains", "seed": 42},
  {"prompt": "cyberpunk city", "seed": 123},
  {"prompt": "fantasy dragon", "seed": 456}
]
EOF

# Run the batch
comfyui-batch jobs.json -W workflows/api/sdxl.json

# With additional parameters
comfyui-batch jobs.json -W workflows/api/flux.json --wait --output ./batch_results
```

### comfyui-info

Display server configuration and system information.

```bash
# Show server info
comfyui-info

# Output includes:
# - Server version
# - Python version
# - PyTorch configuration
# - Available devices (CPU/GPU)
# - Memory usage
# - Installed custom nodes
```

## 🗂️ Workflow Management

### Using Bundled Workflows

The environment includes tested workflows in `workflows/api/`:

```bash
# Stable Diffusion 1.5 - fastest, 512x512 default
comfyui-submit workflows/api/sd15.json -p "ancient temple" -w

# SDXL - high quality, 1024x1024 default
comfyui-submit workflows/api/sdxl.json -p "futuristic city" -w

# SD 3.5 - latest model
comfyui-submit workflows/api/sd35.json -p "fantasy landscape" -w

# FLUX - highest quality, slower
comfyui-submit workflows/api/flux.json -p "photorealistic portrait" -w
```

### Using Custom Workflows

```bash
# Set custom workflow directory
export COMFYUI_WORKFLOWS="$HOME/my-comfyui-workflows"
flox activate

# Now workflows symlink points to your directory
ls workflows/  # Shows your custom workflows

# Submit custom workflow
comfyui-submit workflows/my-custom-workflow.json -p "custom prompt" -w
```

### Creating New Workflows

1. Export from ComfyUI web interface (API format)
2. Save to your workflows directory
3. Submit with the CLI:

```bash
# Export from ComfyUI: Settings → Save (API Format)
# Save as workflows/my-new-workflow.json
comfyui-submit workflows/my-new-workflow.json -p "test prompt"
```

## 🔨 Development

### Modifying the Python Client

The Python package source is in `src/comfyui_client/`:

```bash
# Edit the client code
vim src/comfyui_client/cli.py

# Reinstall in the virtual environment
flox activate
pip install -e .

# Test your changes
comfyui-submit --help
```

### Building and Publishing Updates

```bash
# Increment version in manifest.toml
vim .flox/env/manifest.toml

# Build the package
flox build comfyui-scripts

# Publish to FloxHub (requires authentication)
flox publish comfyui-scripts -o myorg
```

### Project Structure

```
comfyui-client/
├── .flox/
│   ├── env/
│   │   └── manifest.toml    # Flox environment definition
│   └── cache/               # Python venv (auto-created)
├── src/
│   └── comfyui_client/      # Python package source
│       ├── client.py        # WebSocket client implementation
│       ├── cli.py           # CLI commands
│       ├── workflow.py      # Workflow manipulation
│       └── __init__.py
├── workflows/
│   └── api/                 # Bundled workflow templates
└── pyproject.toml           # Python package configuration
```

## 🌐 Publishing and Sharing

### Share via FloxHub

```bash
# Push your customized environment
flox push

# Others can pull it
flox pull --copy <your-handle>/comfyui-client
```

### Package Custom Workflows

Include your workflows in the Flox package:

```bash
# Add to .flox/env/manifest.toml build section
[build.my-workflows]
command = '''
  mkdir -p $out/share/workflows
  cp -r workflows/* $out/share/workflows/
'''

# Build and publish
flox build my-workflows
flox publish my-workflows
```

## 🐛 Troubleshooting

### Connection Issues

```bash
# Test server connection
comfyui-info

# If it fails, check:
# 1. Is ComfyUI running?
# 2. Correct host/port?
COMFYUI_HOST=correct-host COMFYUI_PORT=8188 flox activate

# 3. Firewall blocking connection?
curl http://localhost:8188/system_stats
```

### Python Virtual Environment

```bash
# Rebuild the venv if needed
rm -rf $FLOX_ENV_CACHE/venv
flox activate  # Auto-recreates

# Check venv is active
which python  # Should show .flox/cache/venv/bin/python
```

### Workflow Compatibility

```bash
# Check required models are installed
comfyui-models checkpoints

# Verify workflow structure
python -m json.tool < workflows/my-workflow.json

# Test with simple prompt first
comfyui-submit workflows/api/sd15.json -p "test" --steps 1
```

### Performance Issues

```bash
# Monitor server load
comfyui-status --watch

# Reduce parallel submissions
comfyui-submit workflow.json -c 10  # Sequential instead of --parallel

# Lower generation settings
comfyui-submit workflow.json --steps 20 --width 512 --height 512
```

## 🤖 Architecture Notes (For AI Agents)

### Key Components

1. **Flox Environment Layer**:
   - Manages Python 3.13 + uv package manager
   - Provides `comfyui-scripts` package (version 0.6.9)
   - Auto-configures Python virtual environment

2. **Python Client Library**:
   - WebSocket-based async communication
   - Queue management and progress tracking
   - Workflow JSON manipulation

3. **CLI Tools**:
   - Built with Click + Rich for terminal UI
   - Each tool is a separate entry point
   - Shared configuration via environment variables

### Integration Points

- **Server API**: WebSocket at `ws://{host}:{port}/ws`
- **HTTP Endpoints**:
  - `/prompt` - Submit workflows
  - `/queue` - Queue management
  - `/history` - Job history
  - `/system_stats` - Server info
  - `/object_info` - Available nodes/models

### Workflow Format

Workflows are ComfyUI API JSON format with variable substitution:

```json
{
  "6": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "text": "%%PROMPT%%",  // Replaced by CLI
      "clip": ["4", 0]
    }
  }
}
```

Variables replaced by CLI:
- `%%PROMPT%%` - Positive prompt
- `%%NEGATIVE%%` - Negative prompt
- `%%SEED%%` - Random seed
- `%%STEPS%%` - Sampling steps
- `%%CFG%%` - CFG scale
- `%%WIDTH%%` - Image width
- `%%HEIGHT%%` - Image height
- `%%DENOISE%%` - Denoise strength

### Extension Points

- Add new CLI commands in `src/comfyui_client/cli.py`
- Extend workflow manipulation in `workflow.py`
- Custom workflow templates in `workflows/api/`
- Environment modifications in `.flox/env/manifest.toml`

## 📚 Additional Resources

- [ComfyUI Documentation](https://github.com/comfyanonymous/ComfyUI)
- [Flox Documentation](https://flox.dev/docs)
- [FloxHub](https://hub.flox.dev)
- [Issue Tracker](https://github.com/barstoolbluz/comfyui-client/issues)

## 📄 License

This project is packaged and distributed using Flox. See individual component licenses:
- Python client: See `pyproject.toml`
- ComfyUI workflows: Community-contributed
- Flox environment: MIT