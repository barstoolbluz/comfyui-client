{ pkgs ? import <nixpkgs> {} }:

let
  # Path to installed ComfyUI workflows
  workflowsBase = "/home/daedalus/comfyui-work/user/default/workflows/api";

  # Model configurations
  models = {
    sd15 = { dir = "sd1.5"; prefix = "sd15"; };
    sdxl = { dir = "sdxl"; prefix = "sdxl"; };
    sd35 = { dir = "sd3.5"; prefix = "sd35"; };
    flux = { dir = "flux"; prefix = "flux"; };
  };

  # Operation configurations
  operations = {
    txt2img = { workflow = "txt2img"; needsImage = false; };
    img2img = { workflow = "img2img"; needsImage = true; };
    upscale = { workflow = "upscale"; needsImage = true; };
  };

  # Generate a wrapper script for a model+operation combination
  makeScript = modelKey: opKey:
    let
      model = models.${modelKey};
      op = operations.${opKey};
      scriptName = "${model.prefix}-${opKey}";
      workflowPath = "${workflowsBase}/${model.dir}/${model.prefix}-${op.workflow}.json";
    in
    pkgs.writeShellScriptBin scriptName ''
      #!/usr/bin/env bash
      # ${scriptName} - Submit ${opKey} workflow for ${modelKey}
      # Workflow: ${workflowPath}

      set -euo pipefail

      WORKFLOW="${workflowPath}"

      if [ ! -f "$WORKFLOW" ]; then
        echo "Error: Workflow not found: $WORKFLOW" >&2
        echo "Make sure ComfyUI is installed with workflows at ${workflowsBase}" >&2
        exit 1
      fi

      exec comfyui-submit "$WORKFLOW" --wait "$@"
    '';

  # Generate all script derivations
  allScripts = pkgs.lib.flatten (
    pkgs.lib.mapAttrsToList (modelKey: model:
      pkgs.lib.mapAttrsToList (opKey: op:
        makeScript modelKey opKey
      ) operations
    ) models
  );

in
pkgs.symlinkJoin {
  name = "comfyui-scripts";
  paths = allScripts;
  meta = {
    description = "CLI wrapper scripts for ComfyUI workflows";
    mainProgram = "sdxl-txt2img";
  };
}
