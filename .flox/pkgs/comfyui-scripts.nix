{ lib, writeShellApplication, symlinkJoin }:

let
  workflowsBase = "/home/daedalus/comfyui-work/user/default/workflows";

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
      workflowPath = "${workflowsBase}/${dir}/${model}-${op}.json";
    in
    writeShellApplication {
      inherit name;
      text = ''
        WORKFLOW="${workflowPath}"

        if [ ! -f "$WORKFLOW" ]; then
          echo "Error: Workflow not found: $WORKFLOW" >&2
          echo "Make sure ComfyUI is installed with workflows." >&2
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
