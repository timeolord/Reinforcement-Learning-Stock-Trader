{
  description = "reinforcement learning stock trader using muzero";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; config.allowUnfree = true; config.cudaSupport = true; };
        pythonEnv = pkgs.python3.withPackages (
          ps: with ps; [
            numpy
            pandas
            torchWithCuda
            tensorboard
            gymnasium
            seaborn
            optuna
            yfinance
            ray
          ]
        );
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [ pythonEnv ];
          shellHook = ''
            # nevergrad is not packaged in nixpkgs; install into a local venv
            if [ ! -d .venv ]; then
              python -m venv --system-site-packages .venv
            fi
            source .venv/bin/activate
            pip install nevergrad --quiet
          '';
        };
      }
    );
}
