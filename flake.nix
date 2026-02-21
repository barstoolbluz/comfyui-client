{
  description = "ComfyUI CLI client and workflow scripts";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in {
        packages = {
          comfyui-scripts = pkgs.callPackage ./.flox/pkgs/comfyui-scripts.nix { };
          default = self.packages.${system}.comfyui-scripts;
        };
      }
    );
}
