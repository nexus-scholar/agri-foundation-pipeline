"""Filesystem utilities with Windows long-path safety."""
from __future__ import annotations

import os
import shutil
import stat
import time
from pathlib import Path


def _win_path(path: Path) -> str:
    """Return extended-length Windows path when needed."""
    if os.name != "nt":
        return str(path)
    try:
        resolved = Path(path).resolve(strict=False)
    except Exception:
        resolved = Path(path).absolute()
    as_str = str(resolved)
    return as_str if as_str.startswith("\\\\?\\") else f"\\\\?\\{as_str}"


def ensure_dir(path: Path):
    """Create directory if missing, handling Windows path limits."""
    path = Path(path)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        os.makedirs(_win_path(path), exist_ok=True)


def safe_rmtree(path: Path, attempts: int = 3):
    """Remove a directory tree even if Windows keeps handles open."""
    path = Path(path)
    for attempt in range(attempts):
        if not path.exists():
            return
        shutil.rmtree(path, ignore_errors=True)
        if not path.exists():
            return
        time.sleep(0.5 * (attempt + 1))
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                file_path = Path(root) / name
                try:
                    os.chmod(file_path, stat.S_IWRITE)
                    file_path.unlink(missing_ok=True)
                except Exception:
                    pass
            for name in dirs:
                dir_path = Path(root) / name
                try:
                    dir_path.rmdir()
                except Exception:
                    pass
        try:
            path.rmdir()
        except Exception:
            continue


def copy_file(src: Path, dst: Path):
    """Copy a file preserving metadata while supporting long paths."""
    src = Path(src)
    dst = Path(dst)
    ensure_dir(dst.parent)
    try:
        shutil.copy2(src, dst)
        return
    except Exception:
        pass
    with open(_win_path(src), "rb") as rf, open(_win_path(dst), "wb") as wf:
        shutil.copyfileobj(rf, wf, length=1024 * 1024)
    try:
        shutil.copystat(src, dst)
    except Exception:
        pass


def copytree(src: Path, dst: Path):
    """Copy directory contents while collecting failures."""
    src = Path(src)
    dst = Path(dst)
    ensure_dir(dst)
    failures: list[tuple[str, str, str]] = []
    for root, _, files in os.walk(src):
        rel = Path(root).relative_to(src)
        target_dir = dst / rel
        ensure_dir(target_dir)
        for name in files:
            src_file = Path(root) / name
            dst_file = target_dir / name
            try:
                with open(_win_path(src_file), "rb") as rf, open(_win_path(dst_file), "wb") as wf:
                    shutil.copyfileobj(rf, wf, length=1024 * 1024)
            except Exception as exc:  # pragma: no cover - logging handles details
                failures.append((str(src_file), str(dst_file), str(exc)))
    if failures:
        raise RuntimeError(f"Failed to copy {len(failures)} files; see log for details")


def count_files(path: Path) -> int:
    """Return total number of files contained under the directory."""
    path = Path(path)
    return sum(1 for _ in path.rglob("*") if _.is_file())

