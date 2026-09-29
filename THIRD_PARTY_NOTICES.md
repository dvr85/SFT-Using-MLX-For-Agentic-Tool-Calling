# Third-Party Notices

This project uses the following third-party components. Their licenses apply to
the respective dependencies, model weights, dataset, and derivative artifacts as
noted, and are reproduced or referenced here as required.

## Models

### MiniCPM5-1B (openbmb/MiniCPM5-1B)

- License: Apache License 2.0
- Source: https://huggingface.co/openbmb/MiniCPM5-1B
- Used as the base model for fine-tuning; weights, architecture config, and chat
  template are downloaded via `scripts/convert_minicpm5_1b.sh`.
- The upstream repository and MiniCPM model weights are released under Apache-2.0:
  https://github.com/OpenBMB/MiniCPM/blob/main/LICENSE

## Datasets

### ToolACE (Team-ACE/ToolACE)

- License: Apache License 2.0
- Source: https://huggingface.co/datasets/Team-ACE/ToolACE
- Used for fine-tuning; normalized/normalized splits (`train.jsonl`, `valid.jsonl`,
  `test.jsonl`) are derived from it.

## Libraries

### MLX-LM (ml-explore/mlx-lm)

- License: MIT License
- Source: https://github.com/ml-explore/mlx-lm
- Copyright © 2023 Apple Inc.

### SymPy

- License: BSD 3-Clause License
- Source: https://github.com/sympy/sympy
- Used by the `solve_symbolic` harness tool.

### Matplotlib

- License: BSD-style (Matplotlib license)
- Source: https://github.com/matplotlib/matplotlib
- Used by the `plot_chart` harness tool.

## Apache License 2.0

The MiniCPM5-1B model and ToolACE dataset are licensed under the Apache License,
Version 2.0. A copy of the license text is available at:
https://www.apache.org/licenses/LICENSE-2.0

To the extent this repository distributes portions of, or works derived from,
those upstream components, the Apache-2.0 terms apply to those components.
