# NOTE: Thin wrapper to expose the existing *train_py* module (stored as a
# file without the canonical ".py" extension) under the fully-qualified Python
# module name ``train_py`` so that a plain ``import train_py`` succeeds.  This
# avoids the complex fallback logic in *src/main.py* and makes the codebase
# compatible with standard import mechanisms, packaging tools and static
# analysers.

from __future__ import annotations

import importlib.util as _iu
import sys as _sys
from pathlib import Path as _Path

# -----------------------------------------------------------------------------
# Locate sibling file ``train_py`` (no extension) that contains the actual
# implementation.  ``__file__`` points to *train_py.py* (this wrapper).  By
# stripping the suffix we get the path of the original source file.
# -----------------------------------------------------------------------------
_src_path = _Path(__file__).with_suffix("")  # same directory, no ".py"

if not _src_path.exists():
    raise ImportError(
        "The underlying FoReCoast-CL implementation file 'train_py' was not "
        "found next to the wrapper. Ensure that the project structure is "
        "intact and that the file has not been deleted."
    )

_spec = _iu.spec_from_file_location("_forecoast_train_impl", _src_path)
if _spec is None or _spec.loader is None:  # pragma: no cover – extreme edge case
    raise ImportError(f"Failed to create import spec for {_src_path}")

_mod = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore[arg-type]

# -----------------------------------------------------------------------------
# Re-export everything so that *train_py.py* behaves exactly like the original
# file.  We insert the loaded implementation in *sys.modules* twice so that
# both ``import train_py`` and ``import src.train_py`` resolve to the same
# object, preventing duplicate copies of singletons.
# -----------------------------------------------------------------------------
globals().update(_mod.__dict__)  # make symbols available at top-level
_sys.modules.setdefault("train_py", _mod)
_sys.modules.setdefault("src.train_py", _mod)
