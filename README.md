# ML-in-PL Counterfactuals

Hands-on material for a tutorial on counterfactual explanations prepared for the ML in PL community. The repository contains lightweight configuration files, a small sample dataset, and a set of Jupyter notebooks that walk through several families of counterfactual generation techniques (e.g. Wachter et al., Artelt & Hammer, and method comparisons).

## Install uv
Pick the approach that matches your platform:
- **macOS / Linux**
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Windows (PowerShell)**
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
- If you already have uv through Homebrew, pipx, or another package manager, make sure it is at least v0.4 and verify with:
  ```bash
  uv --version
  ```

The installer adds uv to your shell profile. Open a new terminal after installation so the command is available.

## Set up the tutorial environment
1. **Clone the repository**
   ```bash
   git clone https://github.com/LukiLenkiewicz/ML-in-PL-counterfactuals.git
   cd ML-in-PL-counterfactuals
   ```
2. **Sync dependencies (creates a `.venv` automatically)**
   ```bash
   uv sync
   ```
   The first sync compiles PyTorch and pulls the custom `counterfactuals` fork; expect a short wait.
3. **Install from scratch (to use if uv sync doesn't work)**
   ```bash
   source .venv/bin/activate
   uv pip install .
   ```

##
This tutorial is made on top of [counterfactuals](https://github.com/ofurman/counterfactuals) library.
