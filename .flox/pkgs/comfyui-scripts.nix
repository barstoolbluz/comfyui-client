{ lib, writeShellApplication, symlinkJoin }:

let
  # Model configurations: script prefix -> directory name
  models = {
    sd15 = "SD15";
    sdxl = "SDXL";
    sd35 = "SD35";
    flux = "FLUX";
  };

  operations = [ "txt2img" "img2img" "upscale" ];

  # Generate a single script
  makeScript = model: op:
    let
      dir = models.${model};
      name = "${model}-${op}";
    in
    writeShellApplication {
      inherit name;
      text = ''
        WORKFLOWS_BASE="''${COMFYUI_WORKFLOWS:-$HOME/comfyui-work/user/default/workflows}"
        WORKFLOW="$WORKFLOWS_BASE/${dir}/${model}-${op}.json"

        if [ ! -f "$WORKFLOW" ]; then
          echo "Error: Workflow not found: $WORKFLOW" >&2
          echo "Set COMFYUI_WORKFLOWS to your workflows directory." >&2
          exit 1
        fi

        exec comfyui-submit "$WORKFLOW" --wait "$@"
      '';
    };

  # Generate all scripts
  allScripts = lib.flatten (
    lib.mapAttrsToList (model: _:
      map (op: makeScript model op) operations
    ) models
  );

in
symlinkJoin {
  name = "comfyui-scripts";
  paths = allScripts;
  meta = {
    description = "CLI wrapper scripts for ComfyUI workflows (sd15, sdxl, sd35, flux)";
  };
}
