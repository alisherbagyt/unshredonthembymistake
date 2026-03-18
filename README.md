# unshred on them by mistake

MVP still... in process.

## Structure

- `run_pipeline.py`: CLI entry point (`argparse`)
- `configs/config.yaml`: central pipeline config
- `src/`: core pipeline modules
- `tests/test_pipeline.py`: contract tests for retrieval/solver/renderer

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q
python run_pipeline.py --config configs/config.yaml
```

## Data Layout

Drop raw scans in `data/raw/` (`.png`, `.jpg`, `.jpeg`).

Pipeline output defaults to `data/processed/reconstruction.png`.

## Interactive CLI

```powershell
python run_interactive.py
```

Prompts for an input file or folder, optional pretrained weights, output path, and a debug toggle.
When debug is enabled, progress artifacts are saved under `data/debug/` (segmentation crops,
match scores, pose graph stats, and a layout map).
