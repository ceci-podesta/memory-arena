"""
memory_arena.evaluation.run_metadata
-------------------------------------
Captura metadata de cada corrida de benchmark: timestamps, duración,
hardware, git commit. Se persiste como JSON en `results/runs/<run_id>.json`.

Permite:
- Estimar tiempos para planear re-corridas.
- Detectar regresiones de performance (si una corrida tarda mucho más
  que antes, algo cambió).
- Trazabilidad: saber en qué commit se corrió cada experimento.
"""

import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class RunMetadata:
    run_id: str
    strategy: str
    benchmark: str
    model: str
    num_samples: int
    started_at: str  # ISO 8601 UTC
    ended_at: str | None = None
    duration_seconds: float | None = None
    hardware: dict = field(default_factory=dict)
    git_commit: str | None = None


def start_run(
    strategy: str,
    benchmark: str,
    model: str,
    num_samples: int,
) -> RunMetadata:
    """Inicializa la metadata al arrancar una corrida."""
    now = datetime.now(timezone.utc)
    run_id = f"{now.strftime('%Y%m%d_%H%M%S')}_{strategy}_{benchmark}"
    return RunMetadata(
        run_id=run_id,
        strategy=strategy,
        benchmark=benchmark,
        model=model,
        num_samples=num_samples,
        started_at=now.isoformat(),
        hardware=_capture_hardware(),
        git_commit=_git_short_sha(),
    )


def finalize_run(metadata: RunMetadata, output_path: Path) -> None:
    """Completa metadata con timestamps finales y escribe a disco."""
    ended = datetime.now(timezone.utc)
    started = datetime.fromisoformat(metadata.started_at)
    metadata.ended_at = ended.isoformat()
    metadata.duration_seconds = round((ended - started).total_seconds(), 3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(asdict(metadata), f, indent=2, ensure_ascii=False)


def _capture_hardware() -> dict:
    info = {
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    gpu = _query_gpu()
    if gpu is not None:
        info["gpu"] = gpu
    return info


def _query_gpu() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().splitlines()[0]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _git_short_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None
