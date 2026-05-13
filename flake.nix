{
  description = "C and PyTorch development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    utils.url = "github:numtide/flake-utils";
  };

  outputs = { nixpkgs, utils, ... }:
    utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        cudaPkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };

        python = pkgs.python312.withPackages (ps: with ps; [
          ipython
          matplotlib
          numpy
          pandas
          scipy
          torch
        ]);

        cTools = with pkgs; [
          clang-tools
          cmake
          cppcheck
          gcc13
          gdb
          ninja
          pkg-config
        ];

        mlShell = pkgs.mkShell {
          name = "c-pytorch-env";
          nativeBuildInputs = with pkgs; [
            cmake
            gcc13
            ninja
            pkg-config
          ];

          packages = cTools ++ [
            python
          ];

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc
          ];

          NIX_ENFORCE_NO_NATIVE = "0";

          shellHook = ''
            export CC=gcc
            export CFLAGS="-std=c17 $CFLAGS"

            echo "C/PyTorch environment loaded"
            echo "GCC: $(gcc --version | head -n1)"
            python - <<'PY'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
PY
          '';
        };

        cudaPython = cudaPkgs.python312.withPackages (ps: with ps; [
          ipython
          matplotlib
          numpy
          pandas
          scipy
          torch-bin
          torchvision-bin
        ]);

        cudaShell = cudaPkgs.mkShell {
          name = "c-cuda-pytorch-env";

          nativeBuildInputs = with cudaPkgs; [
            cmake
            gcc13
            gdb
            ninja
            pkg-config
          ];

          packages = with cudaPkgs; [
            clang-tools
            cppcheck
            cudatoolkit
            linuxPackages.nvidia_x11
            cudaPython
          ];

          LD_LIBRARY_PATH = cudaPkgs.lib.makeLibraryPath [
            cudaPkgs.stdenv.cc.cc
            cudaPkgs.cudatoolkit
            cudaPkgs.linuxPackages.nvidia_x11
            "/run/opengl-driver"
          ];

          NIX_ENFORCE_NO_NATIVE = "0";

          shellHook = ''
            export CC=gcc
            export CFLAGS="-std=c17 $CFLAGS"
            export CUDA_PATH=${cudaPkgs.cudatoolkit}
            export EXTRA_LDFLAGS="-L/lib -L${cudaPkgs.linuxPackages.nvidia_x11}/lib"
            export EXTRA_CCFLAGS="-I/usr/include"

            echo "C/CUDA/PyTorch environment loaded"
            echo "GCC: $(gcc --version | head -n1)"
            python - <<'PY'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
PY
          '';
        };
      in
      {
        devShells = {
          default = mlShell;
          cuda = cudaShell;
          ml = mlShell;
        };
      });
}
