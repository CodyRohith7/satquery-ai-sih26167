#!/usr/bin/env python3
"""Run this on the REAL machine that will do the RS-VLM training/inference
(per docs/RUN_ON_WINDOWS.md) - not in a network-isolated, GPU-less test
environment, since those have already been confirmed (docs/rs_adaptation.md
section 7) to have no GPU and no internet access.

Prints and saves an honest snapshot of what's actually available: Python
version, CUDA availability, GPU model/VRAM, and installed versions of every
package the adaptation script needs. Every field is either a real measured
value or explicitly `null` / an error string - never guessed or assumed.

Usage:
    python scripts/verify_environment.py
Writes: environment_report.json (next to this script's working directory)
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys


def _try_import_version(module_name: str):
    try:
        mod = __import__(module_name)
        return getattr(mod, "__version__", "unknown_version_attr")
    except ImportError:
        return None
    except Exception as exc:  # pragma: no cover - defensive, report don't crash
        return f"import_error: {exc}"


def _cuda_info():
    info = {
        "torch_installed": False,
        "cuda_available": None,
        "cuda_version": None,
        "gpu_count": None,
        "gpu_names": [],
        "gpu_total_vram_gb": [],
    }
    try:
        import torch  # type: ignore
    except ImportError:
        return info
    info["torch_installed"] = True
    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = torch.version.cuda
        if info["cuda_available"]:
            n = torch.cuda.device_count()
            info["gpu_count"] = n
            for i in range(n):
                props = torch.cuda.get_device_properties(i)
                info["gpu_names"].append(props.name)
                info["gpu_total_vram_gb"].append(round(props.total_memory / (1024 ** 3), 2))
    except Exception as exc:  # pragma: no cover
        info["error"] = str(exc)
    return info


def _nvidia_smi_raw():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return f"nvidia-smi exited {result.returncode}: {result.stderr.strip()}"
    except FileNotFoundError:
        return "nvidia-smi not found on PATH"
    except Exception as exc:  # pragma: no cover
        return f"error running nvidia-smi: {exc}"


def main() -> int:
    report = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": {
            "torch": _try_import_version("torch"),
            "transformers": _try_import_version("transformers"),
            "peft": _try_import_version("peft"),
            "accelerate": _try_import_version("accelerate"),
            "datasets": _try_import_version("datasets"),
            "rasterio": _try_import_version("rasterio"),
            "huggingface_hub": _try_import_version("huggingface_hub"),
        },
        "cuda": _cuda_info(),
        "nvidia_smi_raw": _nvidia_smi_raw(),
    }

    print(json.dumps(report, indent=2))
    with open("environment_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nWrote environment_report.json in the current directory.")
    print("Bring this file back (it's already inside the connected "
          "PycharmProjects/satquery-ai folder if you run this from there) "
          "so it can be reported accurately rather than assumed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
