import subprocess
from pathlib import Path
from typing import Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_cmd(cmd: Sequence[str], cwd: Optional[Path] = None) -> None:
    cmd_str = " ".join(cmd)
    print(f"[RUN] {cmd_str}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)
