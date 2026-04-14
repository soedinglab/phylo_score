from __future__ import annotations

import hashlib
import os
import struct
from pathlib import Path

import numpy as np

from . import _phylo_score

_CACHE_NAMESPACE = b"phylo_score.optimize_models.v1"


def _default_cache_dir() -> Path:
    return Path("./phylo_score_cache")


def _update_hash_with_bytes(sha1: "hashlib._Hash", label: bytes, value: bytes) -> None:
    sha1.update(label)
    sha1.update(len(value).to_bytes(8, "big"))
    sha1.update(value)


def _optimize_models_cache_key(
    tree_path: str | os.PathLike[str],
    alignment_path: str | os.PathLike[str],
    reg_strength: float,
    mi_strength: float,
) -> str:
    tree_bytes = Path(tree_path).read_bytes()
    alignment_bytes = Path(alignment_path).read_bytes()

    sha1 = hashlib.sha1()
    sha1.update(_CACHE_NAMESPACE)
    _update_hash_with_bytes(sha1, b"tree", tree_bytes)
    _update_hash_with_bytes(sha1, b"alignment", alignment_bytes)
    sha1.update(b"reg_strength")
    sha1.update(struct.pack(">d", float(reg_strength)))
    sha1.update(b"mi_strength")
    sha1.update(struct.pack(">d", float(mi_strength)))
    return sha1.hexdigest()


def _write_cache_file(cache_path: Path, arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        ind_ll=arrays[0],
        pair_ll=arrays[1],
        ind_reg=arrays[2],
        pair_reg=arrays[3],
    )


def optimize_models(
    tree_path: str | os.PathLike[str],
    alignment_path: str | os.PathLike[str],
    reg_strength: float,
    mi_strength: float,
    *,
    cache_dir: str | os.PathLike[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache_root = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    cache_key = _optimize_models_cache_key(tree_path, alignment_path, reg_strength, mi_strength)
    cache_path = cache_root / f"{cache_key}.npz"

    if cache_path.exists():
        cached = np.load(cache_path)
        return (
                    np.array(cached["ind_ll"], copy=True),
                    np.array(cached["pair_ll"], copy=True),
                    np.array(cached["ind_reg"], copy=True),
                    np.array(cached["pair_reg"], copy=True),
                )

    arrays = _phylo_score.optimize_models(str(tree_path), str(alignment_path), reg_strength, mi_strength)
    _write_cache_file(cache_path, arrays)
    return arrays


optimize_models_uncached = _phylo_score.optimize_models

__all__ = ["optimize_models", "optimize_models_uncached"]