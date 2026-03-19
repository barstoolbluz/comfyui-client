{ lib, stdenv, writeShellApplication, writeTextFile, symlinkJoin, runCommand }:

let
  # Model types for workflow scripts
  models = [ "sd15" "sdxl" "sd35" "flux" ];

  operations = [ "txt2img" "img2img" "upscale" "inpaint" ];

  # Generate a single workflow wrapper script
  makeScript = model: op:
    let
      name = "${model}-${op}";
    in
    writeShellApplication {
      inherit name;
      text = ''
        WORKFLOWS_BASE="''${COMFYUI_WORKFLOWS:-''${FLOX_ENV}/share/comfyui-client/workflows}"
        WORKFLOW="$WORKFLOWS_BASE/api/${model}/${model}-${op}.json"

        if [ ! -f "$WORKFLOW" ]; then
          echo "Error: Workflow not found: $WORKFLOW" >&2
          echo "Set COMFYUI_WORKFLOWS to your workflows directory." >&2
          exit 1
        fi

        exec comfyui-submit "$WORKFLOW" --wait "$@"
      '';
    };

  # Generate all workflow scripts
  allScripts = lib.flatten (
    map (model:
      map (op: makeScript model op) operations
    ) models
  );

  # Python source — copied from src/ directory (single source of truth)

  pythonSource = runCommand "comfyui-client-source" {} ''
    mkdir -p $out/share/comfyui-client/src/comfyui_client
    cp ${./../../pyproject.toml} $out/share/comfyui-client/pyproject.toml
    cp ${./../../src/comfyui_client}/*.py $out/share/comfyui-client/src/comfyui_client/

    cat > $out/share/comfyui-client/.flox-build-v4 << 'FLOX_BUILD'
    FLOX_BUILD_RUNTIME_VERSION=4
    description: Source consolidation - Python from src/ directory
    date: 2026-03-19
    change:
      Move all Python source from inline Nix writeTextFile to src/ directory.
      Nix expression now packages src/ files instead of generating them.
      Rename _ws_wait to async_wait_for_completion for Phase 2 API server.
    FLOX_BUILD
  '';

  # Man pages
  manSubmit = writeTextFile {
    name = "comfyui-submit.1";
    text = ''
      .TH COMFYUI-SUBMIT 1 "2026-02-23" "comfyui-client 0.9.0" "ComfyUI Client Manual"
      .SH NAME
      comfyui-submit \- submit a workflow to ComfyUI
      .SH SYNOPSIS
      .B comfyui-submit
      .I workflow_path
      .RI [ OPTIONS ]
      .SH DESCRIPTION
      .B comfyui-submit
      submits a ComfyUI workflow JSON file to a running ComfyUI server for execution.
      The workflow can be modified on-the-fly using command-line options to set
      prompts, seeds, dimensions, and other generation parameters.
      .SH ARGUMENTS
      .TP
      .I workflow_path
      Path to the ComfyUI workflow JSON file. This should be a workflow exported
      in API format (not the standard web UI format).
      .SH OPTIONS
      .TP
      .BR \-p ", " \-\-prompt " " \fITEXT\fR
      Set the positive prompt text. This replaces the text in CLIPTextEncode nodes
      that have "positive" or "prompt" in their title.
      .TP
      .BR \-n ", " \-\-negative " " \fITEXT\fR
      Set the negative prompt text. This replaces the text in CLIPTextEncode nodes
      that have "negative" in their title.
      .TP
      .BR \-s ", " \-\-seed " " \fIINT\fR
      Set the random seed for generation. Applied to KSampler, KSamplerAdvanced,
      and SamplerCustom nodes.
      .TP
      .BR \-\-steps " " \fIINT\fR
      Set the number of sampling steps.
      .TP
      .BR \-\-cfg " " \fIFLOAT\fR
      Set the CFG (Classifier-Free Guidance) scale. Higher values follow the prompt
      more closely. Typical range is 5.0-15.0.
      .TP
      .BR \-W ", " \-\-width " " \fIINT\fR
      Set the image width in pixels. Must be used together with \fB\-\-height\fR.
      .TP
      .BR \-H ", " \-\-height " " \fIINT\fR
      Set the image height in pixels. Must be used together with \fB\-\-width\fR.
      .TP
      .BR \-d ", " \-\-denoise " " \fIFLOAT\fR
      Set the denoise strength for img2img workflows. Range is 0.0 (no change) to
      1.0 (full regeneration). Typical values for img2img are 0.4-0.8.
      .TP
      .BR \-\-sampler " " \fINAME\fR
      Set the sampler algorithm. Common values: euler, euler_ancestral, heun,
      dpmpp_2m, dpmpp_2m_sde, dpmpp_3m_sde, uni_pc, ddim.
      .TP
      .BR \-\-scheduler " " \fINAME\fR
      Set the scheduler type. Values: normal, karras, exponential, sgm_uniform,
      simple, ddim_uniform, beta.
      .TP
      .BR \-i ", " \-\-image " " \fIPATH\fR
      Set the input image path for img2img workflows. This sets the image in
      LoadImage nodes.
      .TP
      .BR \-w ", " \-\-wait
      Wait for the workflow to complete before exiting. Uses a WebSocket connection
      to monitor execution progress in real time, showing which node is running
      and sampling step counts. Without this flag, the command exits immediately
      after submission.
      .TP
      .BR \-o ", " \-\-output " " \fIDIR\fR
      Output directory for downloading generated images. Only effective when used
      with \fB\-\-wait\fR. Images are saved with their original filenames.
      .TP
      .BR \-c ", " \-\-count " " \fIN\fR
      Number of images to generate. The seed is auto-incremented for each
      variation (base, base+1, base+2, ...). The base seed is taken from
      \fB\-\-seed\fR if specified, otherwise from the workflow's current seed.
      Default: 1
      .TP
      .BR \-\-parallel
      When used with \fB\-\-count\fR, submit all jobs at once instead of
      sequentially. Without this flag, each job is submitted, waited on (if
      \fB\-\-wait\fR), and downloaded before the next one starts.
      .TP
      .BR \-\-prefix " " \fITEXT\fR
      Prefix string prepended to output filenames. When set, downloaded images
      are saved as \fIprefix\fR_\fIoriginal_filename\fR instead of just
      \fIoriginal_filename\fR.
      .SH ENVIRONMENT
      .TP
      .B COMFYUI_HOST
      Hostname of the ComfyUI server. Default: localhost
      .TP
      .B COMFYUI_PORT
      Port of the ComfyUI server. Default: 8188
      .SH REMOTE OPERATION
      This command uses HTTP for workflow submission and image retrieval, and
      WebSocket for real-time completion monitoring (with \fB\-\-wait\fR). When using
      \fB\-\-output\fR, generated images are downloaded from the server via the
      ComfyUI \fB/view\fR endpoint. No filesystem access to the ComfyUI server
      is required.
      .PP
      This means you can run the client on a separate machine from ComfyUI:
      .PP
      .RS
      .nf
      COMFYUI_HOST=gpu-server.local comfyui-submit workflow.json \\
          --wait -o ./local-output
      .fi
      .RE
      .PP
      The workflow executes on the remote server, and the resulting images are
      transferred to your local \fB./local-output\fR directory over HTTP.
      .SH EXAMPLES
      Submit a workflow and download results:
      .PP
      .RS
      .nf
      comfyui-submit workflow.json --wait -o ./output
      .fi
      .RE
      .PP
      Generate with custom prompt, seed, and dimensions:
      .PP
      .RS
      .nf
      comfyui-submit workflow.json \\
          -p "a serene mountain landscape at sunset" \\
          -n "blurry, low quality" \\
          -s 42 -W 1024 -H 768 --wait -o ./output
      .fi
      .RE
      .PP
      Img2img with input image and denoise strength:
      .PP
      .RS
      .nf
      comfyui-submit img2img.json \\
          -i input.png -d 0.6 --wait -o ./output
      .fi
      .RE
      .PP
      Fine-tune sampling parameters:
      .PP
      .RS
      .nf
      comfyui-submit workflow.json \\
          -p "detailed portrait" --steps 30 --cfg 7.5 \\
          --sampler dpmpp_2m --scheduler karras \\
          --wait -o ./output
      .fi
      .RE
      .PP
      Submit without waiting (async), retrieve later:
      .PP
      .RS
      .nf
      comfyui-submit workflow.json -p "async job"
      # Outputs: Submitted: a1b2c3d4-e5f6-7890-abcd-ef1234567890

      # Later, retrieve results:
      comfyui-result a1b2c3d4-e5f6-7890-abcd-ef1234567890 -o ./output
      .fi
      .RE
      .PP
      Generate 5 seed variations sequentially:
      .PP
      .RS
      .nf
      comfyui-submit workflow.json \\
          -p "a cat in space" -s 100 --count 5 --wait -o ./output
      .fi
      .RE
      .PP
      Generate 10 variations in parallel:
      .PP
      .RS
      .nf
      comfyui-submit workflow.json \\
          -p "a cat in space" --count 10 --parallel --wait -o ./output
      .fi
      .RE
      .SH SEE ALSO
      .BR comfyui-queue (1),
      .BR comfyui-result (1),
      .BR comfyui-batch (1),
      .BR comfyui-cancel (1),
      .BR comfyui-status (1),
      .BR comfyui-models (1),
      .BR comfyui-info (1),
      .BR comfyui-watch (1),
      .BR comfyui-serve (1),
      .BR comfyui-client (7)
    '';
  };

  manQueue = writeTextFile {
    name = "comfyui-queue.1";
    text = ''
      .TH COMFYUI-QUEUE 1 "2026-02-23" "comfyui-client 0.9.0" "ComfyUI Client Manual"
      .SH NAME
      comfyui-queue \- show ComfyUI queue status
      .SH SYNOPSIS
      .B comfyui-queue
      .SH DESCRIPTION
      .B comfyui-queue
      displays the current status of the ComfyUI execution queue, showing the
      number of running and pending workflows.
      .SH ENVIRONMENT
      .TP
      .B COMFYUI_HOST
      Hostname of the ComfyUI server. Default: localhost
      .TP
      .B COMFYUI_PORT
      Port of the ComfyUI server. Default: 8188
      .SH OUTPUT
      The command outputs two lines:
      .PP
      .RS
      .nf
      Running: N
      Pending: M
      .fi
      .RE
      .PP
      Where N is the number of currently executing workflows and M is the number
      of workflows waiting in the queue.
      .SH EXAMPLES
      Check queue status:
      .PP
      .RS
      .nf
      comfyui-queue
      .fi
      .RE
      .PP
      Check queue on a remote server:
      .PP
      .RS
      .nf
      COMFYUI_HOST=server.local comfyui-queue
      .fi
      .RE
      .SH SEE ALSO
      .BR comfyui-submit (1),
      .BR comfyui-result (1),
      .BR comfyui-batch (1),
      .BR comfyui-cancel (1),
      .BR comfyui-status (1),
      .BR comfyui-models (1),
      .BR comfyui-info (1),
      .BR comfyui-watch (1),
      .BR comfyui-serve (1),
      .BR comfyui-client (7)
    '';
  };

  manResult = writeTextFile {
    name = "comfyui-result.1";
    text = ''
      .TH COMFYUI-RESULT 1 "2026-02-23" "comfyui-client 0.9.0" "ComfyUI Client Manual"
      .SH NAME
      comfyui-result \- retrieve results from a ComfyUI workflow
      .SH SYNOPSIS
      .B comfyui-result
      .I prompt_id
      .RI [ OPTIONS ]
      .SH DESCRIPTION
      .B comfyui-result
      retrieves and downloads the output images from a previously submitted
      ComfyUI workflow. The workflow must have completed successfully.
      .SH ARGUMENTS
      .TP
      .I prompt_id
      The prompt ID returned by
      .BR comfyui-submit (1)
      when the workflow was submitted. This is a UUID that uniquely identifies
      the workflow execution.
      .SH OPTIONS
      .TP
      .BR \-o ", " \-\-output " " \fIDIR\fR
      Output directory for downloading images. Default: current directory (.)
      .SH ENVIRONMENT
      .TP
      .B COMFYUI_HOST
      Hostname of the ComfyUI server. Default: localhost
      .TP
      .B COMFYUI_PORT
      Port of the ComfyUI server. Default: 8188
      .SH REMOTE OPERATION
      Images are downloaded from the ComfyUI server via the \fB/view\fR HTTP
      endpoint. No filesystem access to the server is required. See
      .BR comfyui-submit (1)
      for details.
      .SH EXAMPLES
      Download results to current directory:
      .PP
      .RS
      .nf
      comfyui-result a1b2c3d4-e5f6-7890-abcd-ef1234567890
      .fi
      .RE
      .PP
      Download to a specific directory:
      .PP
      .RS
      .nf
      comfyui-result a1b2c3d4-e5f6-7890-abcd-ef1234567890 -o ./images
      .fi
      .RE
      .PP
      Retrieve from a remote server:
      .PP
      .RS
      .nf
      COMFYUI_HOST=gpu-server.local \\
          comfyui-result a1b2c3d4-e5f6-7890-abcd-ef1234567890 -o ./images
      .fi
      .RE
      .SH EXIT STATUS
      .TP
      .B 0
      Success
      .TP
      .B 1
      Prompt ID not found in history
      .SH SEE ALSO
      .BR comfyui-submit (1),
      .BR comfyui-queue (1),
      .BR comfyui-batch (1),
      .BR comfyui-cancel (1),
      .BR comfyui-status (1),
      .BR comfyui-models (1),
      .BR comfyui-info (1),
      .BR comfyui-watch (1),
      .BR comfyui-serve (1),
      .BR comfyui-client (7)
    '';
  };

  manOverview = writeTextFile {
    name = "comfyui-client.7";
    text = ''
      .TH COMFYUI-CLIENT 7 "2026-02-23" "comfyui-client 0.9.0" "ComfyUI Client Manual"
      .SH NAME
      comfyui-client \- command-line interface for ComfyUI
      .SH DESCRIPTION
      The comfyui-client package provides command-line tools for interacting with
      a ComfyUI server. It includes core commands for workflow submission and
      management, as well as convenient wrapper scripts for common operations.
      .SH COMMANDS
      The package provides these core commands:
      .TP
      .BR comfyui-submit (1)
      Submit a workflow JSON file to ComfyUI with optional parameter overrides.
      Supports \fB\-\-count\fR for generating multiple seed variations.
      .TP
      .BR comfyui-batch (1)
      Run multiple jobs with different parameters from a JSON batch file.
      .TP
      .BR comfyui-queue (1)
      Display the current queue status.
      .TP
      .BR comfyui-result (1)
      Retrieve output images from a completed workflow.
      .TP
      .BR comfyui-cancel (1)
      Cancel running or pending jobs.
      .TP
      .BR comfyui-status (1)
      Show server status, versions, RAM, and GPU info.
      .TP
      .BR comfyui-models (1)
      List available models by folder type.
      .TP
      .BR comfyui-info (1)
      Display generation metadata from ComfyUI PNG images.
      .TP
      .BR comfyui-watch (1)
      Watch a folder for job files and submit them to ComfyUI automatically.
      .TP
      .BR comfyui-serve (1)
      Start the HTTP API server for workflow submission and webhook delivery.
      .SH WRAPPER SCRIPTS
      For convenience, the package includes wrapper scripts that automatically
      select the appropriate workflow file for common model/operation combinations.
      Each wrapper calls
      .B comfyui-submit
      with the \fB\-\-wait\fR flag and passes through all other arguments.
      .SS Stable Diffusion 1.5
      .TP
      .B sd15-txt2img
      Text-to-image generation using SD 1.5
      .TP
      .B sd15-img2img
      Image-to-image generation using SD 1.5
      .TP
      .B sd15-upscale
      Upscaling using SD 1.5
      .TP
      .B sd15-inpaint
      Inpainting using SD 1.5
      .SS Stable Diffusion XL
      .TP
      .B sdxl-txt2img
      Text-to-image generation using SDXL
      .TP
      .B sdxl-img2img
      Image-to-image generation using SDXL
      .TP
      .B sdxl-upscale
      Upscaling using SDXL
      .TP
      .B sdxl-inpaint
      Inpainting using SDXL
      .SS Stable Diffusion 3.5
      .TP
      .B sd35-txt2img
      Text-to-image generation using SD 3.5
      .TP
      .B sd35-img2img
      Image-to-image generation using SD 3.5
      .TP
      .B sd35-upscale
      Upscaling using SD 3.5
      .TP
      .B sd35-inpaint
      Inpainting using SD 3.5
      .SS FLUX
      .TP
      .B flux-txt2img
      Text-to-image generation using FLUX
      .TP
      .B flux-img2img
      Image-to-image generation using FLUX
      .TP
      .B flux-upscale
      Upscaling using FLUX
      .TP
      .B flux-inpaint
      Inpainting using FLUX
      .SH WORKFLOW FILES
      The package bundles workflow JSON files for all 16 model/operation combinations.
      Wrapper scripts find them automatically at:
      .PP
      .RS
      .nf
      $FLOX_ENV/share/comfyui-client/workflows/api/<model>/<model>-<operation>.json
      .fi
      .RE
      .PP
      For example,
      .B sdxl-txt2img
      loads:
      .PP
      .RS
      .nf
      $FLOX_ENV/share/comfyui-client/workflows/api/sdxl/sdxl-txt2img.json
      .fi
      .RE
      .PP
      To use custom workflows instead, set \fBCOMFYUI_WORKFLOWS\fR to a directory
      containing your own workflow tree. This overrides the bundled workflows.
      .PP
      Workflow files must be in ComfyUI API format (exported via "Save (API Format)"
      in the ComfyUI web interface or converted from standard format).
      .SH ENVIRONMENT
      .TP
      .B COMFYUI_HOST
      Hostname of the ComfyUI server. Default: localhost
      .TP
      .B COMFYUI_PORT
      Port of the ComfyUI server. Default: 8188
      .TP
      .B COMFYUI_WORKFLOWS
      Base directory containing workflow files. Overrides the bundled workflows.
      Default: $FLOX_ENV/share/comfyui-client/workflows
      .SH REMOTE OPERATION
      Commands use HTTP for workflow submission, queue queries, and image retrieval.
      The \fB\-\-wait\fR flag uses a WebSocket connection for real-time progress
      monitoring. The client can run on a different machine from the ComfyUI server.
      When downloading images with the \fB\-o\fR option, images are fetched via the
      ComfyUI \fB/view\fR endpoint and saved locally. No filesystem access to the
      server is required.
      .PP
      This architecture supports using ComfyUI as a remote image generation
      service, where the GPU server runs ComfyUI and clients submit workflows
      from separate machines.
      .SH EXAMPLES
      Generate an image with SDXL:
      .PP
      .RS
      .nf
      sdxl-txt2img -p "a beautiful sunset over mountains" -o ./output
      .fi
      .RE
      .PP
      Img2img with SD 1.5 (modify existing image):
      .PP
      .RS
      .nf
      sd15-img2img -i photo.png -p "oil painting style" -d 0.5 -o ./output
      .fi
      .RE
      .PP
      Upscale an image with FLUX:
      .PP
      .RS
      .nf
      flux-upscale -i lowres.png -o ./upscaled
      .fi
      .RE
      .PP
      Generate on a remote GPU server:
      .PP
      .RS
      .nf
      COMFYUI_HOST=gpu-server.local \\
          sdxl-txt2img -p "portrait photo" -s 12345 -o ./output
      .fi
      .RE
      .PP
      Use a custom workflow directly:
      .PP
      .RS
      .nf
      comfyui-submit ~/workflows/custom.json \\
          -p "my prompt" --wait -o ./output
      .fi
      .RE
      .SH SEE ALSO
      .BR comfyui-submit (1),
      .BR comfyui-batch (1),
      .BR comfyui-queue (1),
      .BR comfyui-result (1),
      .BR comfyui-cancel (1),
      .BR comfyui-status (1),
      .BR comfyui-models (1),
      .BR comfyui-info (1),
      .BR comfyui-watch (1),
      .BR comfyui-serve (1)
    '';
  };

  manBatch = writeTextFile {
    name = "comfyui-batch.1";
    text = ''
      .TH COMFYUI-BATCH 1 "2026-02-23" "comfyui-client 0.9.0" "ComfyUI Client Manual"
      .SH NAME
      comfyui-batch \- run multiple ComfyUI jobs from a batch file
      .SH SYNOPSIS
      .B comfyui-batch
      .I batch_file
      .B \-\-workflow
      .I workflow_path
      .RI [ OPTIONS ]
      .SH DESCRIPTION
      .B comfyui-batch
      reads a JSON file containing an array of job objects and submits each one
      to a running ComfyUI server. Each job can override prompt, seed, steps, and
      other generation parameters. All jobs share the same base workflow.
      .PP
      By default, jobs run sequentially: each is submitted, waited on, and its
      images downloaded before the next job starts. With \fB\-\-parallel\fR, all
      jobs are submitted at once and then awaited in order.
      .SH ARGUMENTS
      .TP
      .I batch_file
      Path to a JSON file containing an array of job objects. See
      .B BATCH FILE FORMAT
      below.
      .SH OPTIONS
      .TP
      .BR \-W ", " \-\-workflow " " \fIPATH\fR
      Path to the base workflow JSON file (required). This should be a workflow
      exported in API format.
      .TP
      .BR \-o ", " \-\-output " " \fIDIR\fR
      Output directory for downloading generated images.
      .TP
      .BR \-\-parallel
      Submit all jobs at once instead of sequentially.
      .TP
      .BR \-\-prefix " " \fITEXT\fR
      Prefix string prepended to output filenames.
      .SH BATCH FILE FORMAT
      The batch file must contain a JSON array of objects. Each object can have
      the following optional keys:
      .PP
      .RS
      .nf
      prompt      Positive prompt text (string)
      negative    Negative prompt text (string)
      seed        Random seed (integer)
      steps       Number of sampling steps (integer)
      cfg         CFG scale (float)
      width       Image width in pixels (integer)
      height      Image height in pixels (integer)
      denoise     Denoise strength 0.0-1.0 (float)
      sampler     Sampler algorithm name (string)
      scheduler   Scheduler type (string)
      image       Input image path for img2img (string)
      .fi
      .RE
      .PP
      Any key not present in a job object will use the workflow default.
      .SH EXAMPLES
      Simple batch file with three prompts:
      .PP
      .RS
      .nf
      [
        {"prompt": "oil painting of mountains", "seed": 42},
        {"prompt": "watercolor of ocean", "steps": 50},
        {"prompt": "digital art of forest"}
      ]
      .fi
      .RE
      .PP
      Run a batch file:
      .PP
      .RS
      .nf
      comfyui-batch jobs.json -W workflow.json -o ./output
      .fi
      .RE
      .PP
      Run in parallel:
      .PP
      .RS
      .nf
      comfyui-batch jobs.json -W workflow.json --parallel -o ./output
      .fi
      .RE
      .SH ENVIRONMENT
      .TP
      .B COMFYUI_HOST
      Hostname of the ComfyUI server. Default: localhost
      .TP
      .B COMFYUI_PORT
      Port of the ComfyUI server. Default: 8188
      .SH SEE ALSO
      .BR comfyui-submit (1),
      .BR comfyui-queue (1),
      .BR comfyui-result (1),
      .BR comfyui-cancel (1),
      .BR comfyui-status (1),
      .BR comfyui-models (1),
      .BR comfyui-info (1),
      .BR comfyui-watch (1),
      .BR comfyui-serve (1),
      .BR comfyui-client (7)
    '';
  };

  manStatus = writeTextFile {
    name = "comfyui-status.1";
    text = ''
      .TH COMFYUI-STATUS 1 "2026-02-23" "comfyui-client 0.9.0" "ComfyUI Client Manual"
      .SH NAME
      comfyui-status \- show ComfyUI server status and system info
      .SH SYNOPSIS
      .B comfyui-status
      .SH DESCRIPTION
      .B comfyui-status
      displays the ComfyUI server status including software versions, RAM usage,
      GPU devices with VRAM, and current queue counts.
      .SH ENVIRONMENT
      .TP
      .B COMFYUI_HOST
      Hostname of the ComfyUI server. Default: localhost
      .TP
      .B COMFYUI_PORT
      Port of the ComfyUI server. Default: 8188
      .SH EXAMPLES
      Check local server status:
      .PP
      .RS
      .nf
      comfyui-status
      .fi
      .RE
      .PP
      Check remote server:
      .PP
      .RS
      .nf
      COMFYUI_HOST=gpu-server.local comfyui-status
      .fi
      .RE
      .SH SEE ALSO
      .BR comfyui-submit (1),
      .BR comfyui-queue (1),
      .BR comfyui-result (1),
      .BR comfyui-batch (1),
      .BR comfyui-cancel (1),
      .BR comfyui-models (1),
      .BR comfyui-info (1),
      .BR comfyui-watch (1),
      .BR comfyui-serve (1),
      .BR comfyui-client (7)
    '';
  };

  manModels = writeTextFile {
    name = "comfyui-models.1";
    text = ''
      .TH COMFYUI-MODELS 1 "2026-02-23" "comfyui-client 0.9.0" "ComfyUI Client Manual"
      .SH NAME
      comfyui-models \- list available ComfyUI models
      .SH SYNOPSIS
      .B comfyui-models
      .RI [ FOLDER ]
      .SH DESCRIPTION
      .B comfyui-models
      lists available model types or models within a specific folder. Without
      arguments, it lists all model folder types (checkpoints, loras, vae, etc.).
      With a folder argument, it lists all models in that folder.
      .SH ARGUMENTS
      .TP
      .I FOLDER
      Model folder type to list. Common values: checkpoints, loras, vae,
      controlnet, clip, clip_vision, upscale_models, embeddings.
      .SH ENVIRONMENT
      .TP
      .B COMFYUI_HOST
      Hostname of the ComfyUI server. Default: localhost
      .TP
      .B COMFYUI_PORT
      Port of the ComfyUI server. Default: 8188
      .SH EXAMPLES
      List all model folder types:
      .PP
      .RS
      .nf
      comfyui-models
      .fi
      .RE
      .PP
      List available checkpoints:
      .PP
      .RS
      .nf
      comfyui-models checkpoints
      .fi
      .RE
      .PP
      List LoRA models on a remote server:
      .PP
      .RS
      .nf
      COMFYUI_HOST=gpu-server.local comfyui-models loras
      .fi
      .RE
      .SH SEE ALSO
      .BR comfyui-submit (1),
      .BR comfyui-queue (1),
      .BR comfyui-result (1),
      .BR comfyui-batch (1),
      .BR comfyui-cancel (1),
      .BR comfyui-status (1),
      .BR comfyui-info (1),
      .BR comfyui-watch (1),
      .BR comfyui-serve (1),
      .BR comfyui-client (7)
    '';
  };

  manInfo = writeTextFile {
    name = "comfyui-info.1";
    text = ''
      .TH COMFYUI-INFO 1 "2026-02-23" "comfyui-client 0.9.0" "ComfyUI Client Manual"
      .SH NAME
      comfyui-info \- display generation metadata from ComfyUI PNG images
      .SH SYNOPSIS
      .B comfyui-info
      .I image_path
      .RI [ OPTIONS ]
      .SH DESCRIPTION
      .B comfyui-info
      reads ComfyUI metadata embedded in PNG images and displays the generation
      parameters. ComfyUI stores workflow and prompt data as PNG text chunks.
      This is a local command and does not require a running ComfyUI server.
      .SH ARGUMENTS
      .TP
      .I image_path
      Path to a ComfyUI-generated PNG image file.
      .SH OPTIONS
      .TP
      .BR \-j ", " \-\-json
      Output the full metadata (prompt and workflow) as JSON instead of the
      human-readable summary.
      .SH OUTPUT
      By default, displays a summary of generation parameters:
      .PP
      .RS
      .nf
      Model:     sd_xl_base_1.0.safetensors
      Prompt:    a beautiful landscape
      Seed:      42
      Steps:     20
      CFG:       7.0
      Sampler:   euler
      Scheduler: normal
      Size:      1024x1024
      .fi
      .RE
      .PP
      With \fB\-\-json\fR, outputs the complete ComfyUI prompt and workflow
      metadata as a JSON object.
      .SH EXAMPLES
      Show generation info for an image:
      .PP
      .RS
      .nf
      comfyui-info output/ComfyUI_00001_.png
      .fi
      .RE
      .PP
      Dump full metadata as JSON:
      .PP
      .RS
      .nf
      comfyui-info output/ComfyUI_00001_.png --json
      .fi
      .RE
      .PP
      Pipe JSON to jq for inspection:
      .PP
      .RS
      .nf
      comfyui-info output/ComfyUI_00001_.png -j | jq '.prompt'
      .fi
      .RE
      .SH SEE ALSO
      .BR comfyui-submit (1),
      .BR comfyui-queue (1),
      .BR comfyui-result (1),
      .BR comfyui-batch (1),
      .BR comfyui-cancel (1),
      .BR comfyui-status (1),
      .BR comfyui-models (1),
      .BR comfyui-watch (1),
      .BR comfyui-serve (1),
      .BR comfyui-client (7)
    '';
  };

  manCancel = writeTextFile {
    name = "comfyui-cancel.1";
    text = ''
      .TH COMFYUI-CANCEL 1 "2026-02-23" "comfyui-client 0.9.0" "ComfyUI Client Manual"
      .SH NAME
      comfyui-cancel \- cancel running or pending ComfyUI jobs
      .SH SYNOPSIS
      .B comfyui-cancel
      .RI [ ID... ]
      .RI [ OPTIONS ]
      .SH DESCRIPTION
      .B comfyui-cancel
      cancels ComfyUI jobs. With no arguments, it interrupts the currently
      running workflow. Specific pending jobs can be removed by prompt ID.
      .SH ARGUMENTS
      .TP
      .I ID...
      One or more prompt IDs to remove from the pending queue.
      .SH OPTIONS
      .TP
      .BR \-\-clear
      Clear all pending jobs from the queue (does not interrupt the running job).
      .TP
      .BR \-\-all
      Interrupt the running job and clear all pending jobs.
      .SH ENVIRONMENT
      .TP
      .B COMFYUI_HOST
      Hostname of the ComfyUI server. Default: localhost
      .TP
      .B COMFYUI_PORT
      Port of the ComfyUI server. Default: 8188
      .SH EXAMPLES
      Interrupt the currently running job:
      .PP
      .RS
      .nf
      comfyui-cancel
      .fi
      .RE
      .PP
      Remove specific jobs from the queue:
      .PP
      .RS
      .nf
      comfyui-cancel a1b2c3d4-... e5f6a7b8-...
      .fi
      .RE
      .PP
      Clear the entire pending queue:
      .PP
      .RS
      .nf
      comfyui-cancel --clear
      .fi
      .RE
      .PP
      Stop everything:
      .PP
      .RS
      .nf
      comfyui-cancel --all
      .fi
      .RE
      .SH SEE ALSO
      .BR comfyui-submit (1),
      .BR comfyui-queue (1),
      .BR comfyui-result (1),
      .BR comfyui-batch (1),
      .BR comfyui-status (1),
      .BR comfyui-models (1),
      .BR comfyui-info (1),
      .BR comfyui-watch (1),
      .BR comfyui-serve (1),
      .BR comfyui-client (7)
    '';
  };

  manWatch = writeTextFile {
    name = "comfyui-watch.1";
    text = ''
      .TH COMFYUI-WATCH 1 "2026-03-19" "comfyui-client 0.9.0" "ComfyUI Client Manual"
      .SH NAME
      comfyui-watch \- watch a folder for ComfyUI job files
      .SH SYNOPSIS
      .B comfyui-watch
      .RI [ OPTIONS ]
      .SH DESCRIPTION
      .B comfyui-watch
      monitors a directory for JSON job files, submits them to a running ComfyUI
      server, downloads generated images, and moves job files to completed or
      failed subdirectories. Jobs are processed sequentially in FIFO order
      (oldest first by modification time).
      .PP
      On startup, the watcher creates five subdirectories under the watch
      directory:
      .TP
      .B incoming/
      Drop job JSON files here. The watcher picks them up automatically.
      .TP
      .B processing/
      Files currently being submitted and awaited. If the watcher is
      interrupted, files here are moved back to incoming/ on next startup.
      .TP
      .B completed/
      Successfully processed jobs. The job file is annotated with
      \fB_result\fR metadata containing the prompt ID, image filenames,
      and completion timestamp.
      .TP
      .B failed/
      Jobs that encountered errors. The job file is annotated with
      \fB_error\fR metadata containing the error message and timestamp.
      .TP
      .B output/
      Downloaded images from completed jobs.
      .SH OPTIONS
      .TP
      .BR \-d ", " \-\-dir " " \fIPATH\fR
      Watch directory. Default: \fB$COMFYUI_WATCH_DIR\fR
      .TP
      .BR \-w ", " \-\-workflow " " \fIPATH\fR
      Default workflow JSON file used for jobs that don't specify their own.
      Default: \fB$COMFYUI_WATCH_WORKFLOW\fR
      .TP
      .BR \-p ", " \-\-poll " " \fISECONDS\fR
      Poll interval in seconds. Default: \fB$COMFYUI_WATCH_POLL\fR or 2.0
      .SH JOB FILE FORMATS
      Job files are JSON and support three formats:
      .SS Minimal (parameters only)
      A dict of parameter overrides applied to the default workflow:
      .PP
      .RS
      .nf
      {"prompt": "a sunset over mountains", "seed": 42, "steps": 25}
      .fi
      .RE
      .SS Full (with workflow reference)
      A dict with a \fBworkflow\fR key pointing to a workflow file:
      .PP
      .RS
      .nf
      {
        "workflow": "/path/to/workflow.json",
        "prompt": "a sunset over mountains",
        "seed": 42
      }
      .fi
      .RE
      .PP
      Relative workflow paths are resolved against the watch directory.
      .SS Batch (array)
      An array of job dicts, each processed sequentially:
      .PP
      .RS
      .nf
      [
        {"prompt": "mountains at dawn", "seed": 1},
        {"prompt": "ocean at sunset", "seed": 2}
      ]
      .fi
      .RE
      .SH SUPPORTED PARAMETERS
      Job dicts accept these optional keys (same as comfyui-submit options):
      .PP
      .RS
      .nf
      prompt      Positive prompt text (string)
      negative    Negative prompt text (string)
      seed        Random seed (integer)
      steps       Sampling steps (integer)
      cfg         CFG scale (float)
      width       Image width (integer)
      height      Image height (integer)
      denoise     Denoise strength 0.0-1.0 (float)
      sampler     Sampler algorithm (string)
      scheduler   Scheduler type (string)
      image       Input image path (string)
      workflow    Workflow JSON path (string, full format only)
      .fi
      .RE
      .SH RESULT METADATA
      After processing, job files are annotated with result metadata:
      .SS Success
      .PP
      .RS
      .nf
      {
        "prompt": "a sunset",
        "_result": {
          "prompt_id": "abc-123",
          "images": ["ComfyUI_00001_.png"],
          "completed_at": "2026-03-19T12:00:00+00:00"
        }
      }
      .fi
      .RE
      .SS Failure
      .PP
      .RS
      .nf
      {
        "prompt": "a sunset",
        "_error": {
          "message": "Workflow error: node 5 failed",
          "failed_at": "2026-03-19T12:00:00+00:00"
        }
      }
      .fi
      .RE
      .SH ERROR HANDLING
      .TP
      .B JSON parse errors
      File moved to failed/ with error metadata.
      .TP
      .B Missing workflow
      File moved to failed/ with error metadata.
      .TP
      .B ComfyUI unreachable
      File left in processing/ and retried next poll cycle.
      .TP
      .B Execution errors
      File moved to failed/ with error metadata including prompt_id.
      .SH ENVIRONMENT
      .TP
      .B COMFYUI_WATCH_DIR
      Default watch directory. Used when \fB\-\-dir\fR is not specified.
      Set automatically by Flox to \fB$FLOX_ENV_CACHE/watch\fR.
      .TP
      .B COMFYUI_WATCH_WORKFLOW
      Default workflow file path. Used when \fB\-\-workflow\fR is not specified.
      .TP
      .B COMFYUI_WATCH_POLL
      Default poll interval in seconds. Used when \fB\-\-poll\fR is not specified.
      .TP
      .B COMFYUI_HOST
      Hostname of the ComfyUI server. Default: localhost
      .TP
      .B COMFYUI_PORT
      Port of the ComfyUI server. Default: 8188
      .SH EXAMPLES
      Watch a directory with a default workflow:
      .PP
      .RS
      .nf
      comfyui-watch -d ~/jobs -w ~/workflows/sdxl-txt2img.json
      .fi
      .RE
      .PP
      Use environment variables (e.g., inside Flox activate):
      .PP
      .RS
      .nf
      export COMFYUI_WATCH_DIR=~/jobs
      export COMFYUI_WATCH_WORKFLOW=~/workflows/flux-txt2img.json
      comfyui-watch
      .fi
      .RE
      .PP
      Submit a job by dropping a file:
      .PP
      .RS
      .nf
      echo '{"prompt": "a cat astronaut"}' > ~/jobs/incoming/cat.json
      .fi
      .RE
      .PP
      Run as a Flox service:
      .PP
      .RS
      .nf
      flox services start comfyui-watch
      .fi
      .RE
      .SH SEE ALSO
      .BR comfyui-submit (1),
      .BR comfyui-batch (1),
      .BR comfyui-queue (1),
      .BR comfyui-result (1),
      .BR comfyui-cancel (1),
      .BR comfyui-status (1),
      .BR comfyui-models (1),
      .BR comfyui-info (1),
      .BR comfyui-serve (1),
      .BR comfyui-client (7)
    '';
  };

  manServe = writeTextFile {
    name = "comfyui-serve.1";
    text = ''
      .TH COMFYUI-SERVE 1 "2026-03-19" "comfyui-client 0.9.0" "ComfyUI Client Manual"
      .SH NAME
      comfyui-serve \- start the comfyui-client API server
      .SH SYNOPSIS
      .B comfyui-serve
      .RI [ OPTIONS ]
      .SH DESCRIPTION
      .B comfyui-serve
      starts a FastAPI HTTP server that accepts ComfyUI workflow submissions
      and returns results synchronously or via webhook. The server provides
      health and readiness endpoints suitable for Kubernetes probes, queue
      and model inspection, and prompt cancellation.
      .PP
      Workflows are submitted via
      .B POST /prompt
      with a JSON body containing the ComfyUI API-format workflow. Without a
      webhook URL, the server waits for completion and returns images inline
      as base64-encoded data. With a webhook URL, the server returns 202
      immediately and delivers results asynchronously.
      .SH OPTIONS
      .TP
      .BR \-\-host " " \fIADDRESS\fR
      Bind address. Default: \fBCOMFYUI_SERVE_HOST\fR or 0.0.0.0
      .TP
      .BR \-\-port " " \fIPORT\fR
      Bind port. Default: \fBCOMFYUI_SERVE_PORT\fR or 3000
      .TP
      .BR \-\-log\-level " " \fILEVEL\fR
      Uvicorn log level. Values: debug, info, warning, error, critical.
      Default: info
      .SH ENDPOINTS
      .TS
      l l l.
      Method	Path	Description
      _
      GET	/health	Always returns 200 {"status":"ok"}
      GET	/ready	200 if ComfyUI reachable, 503 otherwise
      GET	/queue	Running and pending job counts
      GET	/models	List model folder types
      GET	/models/{type}	List models in a folder
      GET	/status	System stats + queue combined
      POST	/prompt	Submit workflow (sync or async)
      POST	/cancel	Cancel a specific prompt by ID
      POST	/cancel/all	Interrupt running + clear pending
      .TE
      .SH PROMPT SUBMISSION
      .B POST /prompt
      accepts a JSON body:
      .PP
      .RS
      .nf
      {
        "prompt": { ... },
        "id": "optional-request-id",
        "webhook_url": "https://example.com/hook",
        "convert_output": {
          "format": "jpeg",
          "quality": 90
        }
      }
      .fi
      .RE
      .PP
      The \fBprompt\fR field contains a ComfyUI API-format workflow dict.
      .PP
      Without \fBwebhook_url\fR: returns 200 with images as base64.
      .PP
      With \fBwebhook_url\fR: returns 202 immediately, delivers results
      via POST to the webhook URL when complete.
      .SH WEBHOOK SIGNING
      If \fBCOMFYUI_WEBHOOK_SECRET\fR is set, webhook deliveries include an
      HMAC-SHA256 signature following the Standard Webhooks specification.
      .PP
      Headers sent with each webhook POST:
      .TP
      .B webhook-id
      Unique message identifier (msg_{request_id})
      .TP
      .B webhook-timestamp
      Unix timestamp of the delivery attempt
      .TP
      .B webhook-signature
      v1,{base64-encoded HMAC-SHA256} (only when secret is configured)
      .PP
      The signature is computed over: \fB"{msg_id}.{timestamp}.{body}"\fR
      .PP
      Secrets may use the \fBwhsec_\fR prefix (base64-encoded key after
      stripping the prefix) per the Standard Webhooks convention.
      .PP
      Webhook delivery retries up to 3 times with exponential backoff
      (1s, 2s, 4s).
      .SH ENVIRONMENT
      .TP
      .B COMFYUI_SERVE_HOST
      Default bind address. Default: 0.0.0.0
      .TP
      .B COMFYUI_SERVE_PORT
      Default bind port. Default: 3000
      .TP
      .B COMFYUI_WEBHOOK_SECRET
      HMAC-SHA256 secret for signing webhook payloads. Optional.
      .TP
      .B COMFYUI_HOST
      Hostname of the ComfyUI server. Default: localhost
      .TP
      .B COMFYUI_PORT
      Port of the ComfyUI server. Default: 8188
      .SH EXAMPLES
      Start the server with defaults:
      .PP
      .RS
      .nf
      comfyui-serve
      .fi
      .RE
      .PP
      Start on a custom port:
      .PP
      .RS
      .nf
      comfyui-serve --port 8080
      .fi
      .RE
      .PP
      Submit a workflow synchronously (returns images inline):
      .PP
      .RS
      .nf
      curl -X POST http://localhost:3000/prompt \\
        -H "Content-Type: application/json" \\
        -d '{"prompt": {"1": {"class_type": "CheckpointLoaderSimple", ...}}}'
      .fi
      .RE
      .PP
      Submit with webhook (returns 202, delivers result later):
      .PP
      .RS
      .nf
      curl -X POST http://localhost:3000/prompt \\
        -H "Content-Type: application/json" \\
        -d '{
          "prompt": { ... },
          "webhook_url": "https://example.com/hook",
          "convert_output": {"format": "jpeg", "quality": 90}
        }'
      .fi
      .RE
      .PP
      Check readiness:
      .PP
      .RS
      .nf
      curl http://localhost:3000/ready
      .fi
      .RE
      .PP
      Run as a Flox service:
      .PP
      .RS
      .nf
      flox services start comfyui-api
      .fi
      .RE
      .SH WORKFLOW TEMPLATES
      On startup, the server discovers Python workflow template modules and
      registers typed \fBPOST /workflow/{model}/{operation}\fR routes. These
      provide Pydantic-validated, OpenAPI-documented endpoints as an alternative
      to submitting raw workflow JSON via \fBPOST /prompt\fR.
      .PP
      Available template routes (16 total):
      .PP
      .RS
      .nf
      POST /workflow/sd15/txt2img    SD 1.5 text-to-image
      POST /workflow/sd15/img2img    SD 1.5 image-to-image
      POST /workflow/sd15/upscale    SD 1.5 upscale
      POST /workflow/sd15/inpaint    SD 1.5 inpainting
      POST /workflow/sdxl/txt2img    SDXL text-to-image
      POST /workflow/sdxl/img2img    SDXL image-to-image
      POST /workflow/sdxl/upscale    SDXL upscale
      POST /workflow/sdxl/inpaint    SDXL inpainting
      POST /workflow/sd35/txt2img    SD 3.5 text-to-image
      POST /workflow/sd35/img2img    SD 3.5 image-to-image
      POST /workflow/sd35/upscale    SD 3.5 upscale
      POST /workflow/sd35/inpaint    SD 3.5 inpainting
      POST /workflow/flux/txt2img    FLUX text-to-image
      POST /workflow/flux/img2img    FLUX image-to-image
      POST /workflow/flux/upscale    FLUX upscale
      POST /workflow/flux/inpaint    FLUX inpainting
      .fi
      .RE
      .PP
      Each route accepts a JSON body with typed parameters (prompt, seed, steps,
      cfg, etc.) and returns a \fB{"prompt": {...}}\fR response containing the
      parameterized workflow ready for submission to ComfyUI.
      .PP
      To override the template search directory, set \fBCOMFYUI_WORKFLOW_DIR\fR.
      .PP
      Example:
      .PP
      .RS
      .nf
      curl -X POST http://localhost:3000/workflow/sdxl/txt2img \\
        -H "Content-Type: application/json" \\
        -d '{"prompt": "a sunset over mountains", "seed": 42, "steps": 25}'
      .fi
      .RE
      .SH SEE ALSO
      .BR comfyui-submit (1),
      .BR comfyui-queue (1),
      .BR comfyui-result (1),
      .BR comfyui-batch (1),
      .BR comfyui-cancel (1),
      .BR comfyui-status (1),
      .BR comfyui-models (1),
      .BR comfyui-info (1),
      .BR comfyui-watch (1),
      .BR comfyui-client (7)
    '';
  };

  # Bundle workflow files
  workflowFiles = runCommand "comfyui-workflows" {} ''
    mkdir -p $out/share/comfyui-client/workflows/api
    cp -r ${./../../workflows/api}/* $out/share/comfyui-client/workflows/api/
    mkdir -p $out/share/comfyui-client/workflows/templates
    cp -r ${./../../workflows/templates}/* $out/share/comfyui-client/workflows/templates/
  '';

  # Bundle man pages
  manPages = runCommand "comfyui-client-man" {} ''
    mkdir -p $out/share/man/man1 $out/share/man/man7
    cp ${manSubmit} $out/share/man/man1/comfyui-submit.1
    cp ${manQueue} $out/share/man/man1/comfyui-queue.1
    cp ${manResult} $out/share/man/man1/comfyui-result.1
    cp ${manBatch} $out/share/man/man1/comfyui-batch.1
    cp ${manCancel} $out/share/man/man1/comfyui-cancel.1
    cp ${manStatus} $out/share/man/man1/comfyui-status.1
    cp ${manModels} $out/share/man/man1/comfyui-models.1
    cp ${manInfo} $out/share/man/man1/comfyui-info.1
    cp ${manWatch} $out/share/man/man1/comfyui-watch.1
    cp ${manServe} $out/share/man/man1/comfyui-serve.1
    cp ${manOverview} $out/share/man/man7/comfyui-client.7
  '';

  # Setup script to install Python package into venv
  setupScript = writeShellApplication {
    name = "comfyui-client-setup";
    text = ''
      VENV="''${1:-$FLOX_ENV_CACHE/venv}"
      SOURCE_DIR="''${FLOX_ENV}/share/comfyui-client"

      # Print build recipe version marker
      flox_build_marker=$(find "$SOURCE_DIR" -maxdepth 1 -name '.flox-build-v*' -print -quit)
      if [ -n "$flox_build_marker" ]; then
        echo "=============================================="
        echo "FLOX_BUILD_RUNTIME_VERSION: $(basename "$flox_build_marker" | sed 's/.flox-build-v//')"
        echo "Source: $SOURCE_DIR"
        echo "=============================================="
      fi

      if [ ! -d "$VENV" ]; then
        echo "Error: venv not found at $VENV" >&2
        exit 1
      fi

      if [ ! -d "$SOURCE_DIR" ]; then
        echo "Error: source not found at $SOURCE_DIR" >&2
        exit 1
      fi

      # Install using uv/pip from the venv
      if command -v uv >/dev/null 2>&1; then
        uv pip install --python "$VENV/bin/python" "$SOURCE_DIR" --quiet
      else
        "$VENV/bin/pip" install "$SOURCE_DIR" --quiet
      fi
    '';
  };

  bashCompletions = writeTextFile {
    name = "comfyui-completions.bash";
    text = ''
      # Bash completions for comfyui-client commands

      _comfyui_submit_completions() {
          local cur prev opts
          COMPREPLY=()
          cur="''${COMP_WORDS[COMP_CWORD]}"
          prev="''${COMP_WORDS[COMP_CWORD-1]}"

          opts="--prompt --negative --seed --steps --cfg --width --height --denoise --sampler --scheduler --image --wait --output --count --parallel --prefix --help"

          case "$prev" in
              --sampler)
                  COMPREPLY=( $(compgen -W "euler euler_ancestral heun dpmpp_2m dpmpp_2m_sde dpmpp_3m_sde uni_pc ddim" -- "$cur") )
                  return 0
                  ;;
              --scheduler)
                  COMPREPLY=( $(compgen -W "normal karras exponential sgm_uniform simple ddim_uniform beta" -- "$cur") )
                  return 0
                  ;;
              --image|--output|-i|-o)
                  COMPREPLY=( $(compgen -f -- "$cur") )
                  return 0
                  ;;
              --prompt|--negative|--seed|--steps|--cfg|--width|--height|--denoise|--count|--prefix|-p|-n|-s|-W|-H|-d|-c)
                  return 0
                  ;;
          esac

          if [[ "$cur" == -* ]]; then
              COMPREPLY=( $(compgen -W "$opts" -- "$cur") )
          else
              COMPREPLY=( $(compgen -f -- "$cur") )
          fi
      }

      _comfyui_cancel_completions() {
          local cur opts
          COMPREPLY=()
          cur="''${COMP_WORDS[COMP_CWORD]}"
          opts="--clear --all --help"

          if [[ "$cur" == -* ]]; then
              COMPREPLY=( $(compgen -W "$opts" -- "$cur") )
          fi
      }

      _comfyui_models_completions() {
          local cur folders
          COMPREPLY=()
          cur="''${COMP_WORDS[COMP_CWORD]}"
          folders="checkpoints loras vae controlnet clip clip_vision upscale_models embeddings hypernetworks"

          if [[ "$cur" != -* ]]; then
              COMPREPLY=( $(compgen -W "$folders" -- "$cur") )
          else
              COMPREPLY=( $(compgen -W "--help" -- "$cur") )
          fi
      }

      _comfyui_info_completions() {
          local cur prev opts
          COMPREPLY=()
          cur="''${COMP_WORDS[COMP_CWORD]}"
          opts="--json --help"

          if [[ "$cur" == -* ]]; then
              COMPREPLY=( $(compgen -W "$opts" -- "$cur") )
          else
              COMPREPLY=( $(compgen -f -X '!*.png' -- "$cur") )
          fi
      }

      _comfyui_batch_completions() {
          local cur prev opts
          COMPREPLY=()
          cur="''${COMP_WORDS[COMP_CWORD]}"
          prev="''${COMP_WORDS[COMP_CWORD-1]}"
          opts="--workflow --output --parallel --prefix --help"

          case "$prev" in
              --workflow|-W|--output|-o)
                  COMPREPLY=( $(compgen -f -- "$cur") )
                  return 0
                  ;;
          esac

          if [[ "$cur" == -* ]]; then
              COMPREPLY=( $(compgen -W "$opts" -- "$cur") )
          else
              COMPREPLY=( $(compgen -f -- "$cur") )
          fi
      }

      _comfyui_simple_completions() {
          local cur
          COMPREPLY=()
          cur="''${COMP_WORDS[COMP_CWORD]}"
          if [[ "$cur" == -* ]]; then
              COMPREPLY=( $(compgen -W "--help" -- "$cur") )
          fi
      }

      _comfyui_watch_completions() {
          local cur prev opts
          COMPREPLY=()
          cur="''${COMP_WORDS[COMP_CWORD]}"
          prev="''${COMP_WORDS[COMP_CWORD-1]}"
          opts="--dir --workflow --poll --help"

          case "$prev" in
              --dir|-d|--workflow|-w)
                  COMPREPLY=( $(compgen -f -- "$cur") )
                  return 0
                  ;;
              --poll|-p)
                  return 0
                  ;;
          esac

          if [[ "$cur" == -* ]]; then
              COMPREPLY=( $(compgen -W "$opts" -- "$cur") )
          fi
      }

      _comfyui_serve_completions() {
          local cur prev opts
          COMPREPLY=()
          cur="''${COMP_WORDS[COMP_CWORD]}"
          prev="''${COMP_WORDS[COMP_CWORD-1]}"
          opts="--host --port --log-level --help"

          case "$prev" in
              --log-level)
                  COMPREPLY=( $(compgen -W "debug info warning error critical" -- "$cur") )
                  return 0
                  ;;
              --host|--port)
                  return 0
                  ;;
          esac

          if [[ "$cur" == -* ]]; then
              COMPREPLY=( $(compgen -W "$opts" -- "$cur") )
          fi
      }

      # Register completions
      complete -F _comfyui_submit_completions comfyui-submit
      complete -F _comfyui_batch_completions comfyui-batch
      complete -F _comfyui_cancel_completions comfyui-cancel
      complete -F _comfyui_models_completions comfyui-models
      complete -F _comfyui_info_completions comfyui-info
      complete -F _comfyui_simple_completions comfyui-queue
      complete -F _comfyui_simple_completions comfyui-status
      complete -F _comfyui_simple_completions comfyui-result
      complete -F _comfyui_watch_completions comfyui-watch
      complete -F _comfyui_serve_completions comfyui-serve

      # Register wrapper scripts (same options as submit)
      complete -F _comfyui_submit_completions sd15-txt2img
      complete -F _comfyui_submit_completions sd15-img2img
      complete -F _comfyui_submit_completions sd15-upscale
      complete -F _comfyui_submit_completions sd15-inpaint
      complete -F _comfyui_submit_completions sdxl-txt2img
      complete -F _comfyui_submit_completions sdxl-img2img
      complete -F _comfyui_submit_completions sdxl-upscale
      complete -F _comfyui_submit_completions sdxl-inpaint
      complete -F _comfyui_submit_completions sd35-txt2img
      complete -F _comfyui_submit_completions sd35-img2img
      complete -F _comfyui_submit_completions sd35-upscale
      complete -F _comfyui_submit_completions sd35-inpaint
      complete -F _comfyui_submit_completions flux-txt2img
      complete -F _comfyui_submit_completions flux-img2img
      complete -F _comfyui_submit_completions flux-upscale
      complete -F _comfyui_submit_completions flux-inpaint
    '';
  };

  completionFiles = runCommand "comfyui-completions" {} ''
    mkdir -p $out/share/bash-completion/completions
    cp ${bashCompletions} $out/share/bash-completion/completions/comfyui-submit
    for cmd in comfyui-batch comfyui-cancel comfyui-status comfyui-models comfyui-info \
               comfyui-queue comfyui-result comfyui-watch comfyui-serve \
               sd15-txt2img sd15-img2img sd15-upscale sd15-inpaint \
               sdxl-txt2img sdxl-img2img sdxl-upscale sdxl-inpaint \
               sd35-txt2img sd35-img2img sd35-upscale sd35-inpaint \
               flux-txt2img flux-img2img flux-upscale flux-inpaint; do
      ln -s comfyui-submit $out/share/bash-completion/completions/$cmd
    done
  '';

  joined = symlinkJoin {
    name = "comfyui-scripts-joined";
    paths = allScripts ++ [ pythonSource setupScript manPages workflowFiles completionFiles ];
  };

in
stdenv.mkDerivation {
  pname = "comfyui-scripts";
  version = "0.9.0";

  dontUnpack = true;
  dontBuild = true;

  installPhase = ''
    runHook preInstall
    cp -RP ${joined} $out
    runHook postInstall
  '';

  meta = {
    description = "CLI wrapper scripts for ComfyUI workflows (sd15, sdxl, sd35, flux)";
  };
}
