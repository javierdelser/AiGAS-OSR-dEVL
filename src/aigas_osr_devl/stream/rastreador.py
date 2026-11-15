"""Load ``ClassTracker`` from ``legacy/aigas_osr_devl/estimar_cardinalidad_sin_etiquetas.py``."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_LEGACY = Path(__file__).resolve().parents[3] / "legacy" / "aigas_osr_devl" / "estimar_cardinalidad_sin_etiquetas.py"
_spec = importlib.util.spec_from_file_location("estimar_cardinalidad_sin_etiquetas", _LEGACY)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

ClassTracker = _mod.ClassTracker
RastreadorClases = ClassTracker
tamano_prefijo = _mod.tamano_prefijo

__all__ = ["ClassTracker", "RastreadorClases", "tamano_prefijo"]