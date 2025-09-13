# Thin wrapper to make ``import preprocess_py`` work. The real implementation
# lives in the sibling file without the ".py" suffix. This file dynamically
# loads it and re-exports every public symbol.

from __future__ import annotations

import importlib.util as _iu
import sys as _sys
from pathlib import Path as _Path

_src_path = _Path(__file__).with_suffix("")
if not _src_path.exists():
    raise ImportError("Missing source file 'preprocess_py' next to wrapper.")

_spec = _iu.spec_from_file_location("_forecoast_preprocess_impl", _src_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not create import spec for {_src_path}")

_mod = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore[arg-type]

globals().update(_mod.__dict__)
_sys.modules.setdefault("preprocess_py", _mod)
_sys.modules.setdefault("src.preprocess_py", _mod)
