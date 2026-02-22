{
  description = "PyTorch and Jupyter Dev Environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      nixpkgs,
      flake-utils,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
          };
        };

        pythonPackages = pkgs.python3Packages;
        cuda = pkgs.cudaPackages;
      in
      {
        devShells.default = pkgs.mkShell {
          venvDir = ".venv";
          buildInputs = [
            cuda.cudatoolkit
            cuda.saxpy # test cuda workable

            pythonPackages.python
            pythonPackages.venvShellHook
          ];
          postShellHook = ''
            # allow pip to install wheels
            unset SOURCE_DATE_EPOCH
            export CUDA_PATH=${cuda.cudatoolkit}
            export CUDA_HOME=${cuda.cudatoolkit}
          '';
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.zlib
            pkgs.stdenv.cc.cc
            cuda.cudatoolkit
            cuda.cudnn
            "/run/opengl-driver" # for NixOS
          ];
        };

      }
    );
}
