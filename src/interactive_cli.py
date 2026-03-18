import shutil
from datetime import datetime
from pathlib import Path

from src.pipeline import load_config, run


def _prompt(text: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{text}{suffix}: ").strip()
    return value or (default or "")


def _prompt_yes_no(text: str, default: bool = False) -> bool:
    d = "y" if default else "n"
    value = input(f"{text} [y/n, default {d}]: ").strip().lower()
    if not value:
        return default
    return value in {"y", "yes"}


def _resolve_input_path(input_path: str, base_dir: str = "data/interactive_inputs") -> str:
    p = Path(input_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"input path not found: {p}")

    if p.is_dir():
        return str(p)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(base_dir) / f"{p.stem}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, out_dir / p.name)
    return str(out_dir)


def run_interactive() -> str:
    print("\nshred_net v2 — interactive reconstruction")

    config_path = _prompt("Config path", "configs/config.yaml")
    cfg = load_config(config_path)

    input_path = _prompt("Input image (file or folder)")
    cfg["pipeline"]["input_dir"] = _resolve_input_path(input_path)

    output_path = _prompt("Output image", cfg["pipeline"].get("output_image", "data/reconstruction.tiff"))
    cfg["pipeline"]["output_image"] = output_path

    device = _prompt("Device (cpu/cuda)", cfg["features"].get("device", "cpu"))
    if device:
        cfg["features"]["device"] = device

    if not cfg["features"].get("weights_path"):
        default_ckpt = Path("checkpoints/eac_net_best.pt")
        if default_ckpt.exists() and _prompt_yes_no("Use pretrained weights from checkpoints/eac_net_best.pt?", True):
            cfg["features"]["weights_path"] = str(default_ckpt)

    debug = _prompt_yes_no("Save progress images (debug)?", False)
    if debug:
        debug_dir = _prompt("Debug output folder", cfg["pipeline"].get("debug_dir", "data/debug"))
        cfg["pipeline"]["debug_dir"] = debug_dir

    Path(cfg["pipeline"]["output_image"]).parent.mkdir(parents=True, exist_ok=True)

    return run(cfg, debug=debug)
