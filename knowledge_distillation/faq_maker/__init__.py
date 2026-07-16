"""Compatibility package for legacy ``faq_maker.*`` imports.

New code should import from ``knowledge_distillation`` directly. This package
keeps older commands that add ``knowledge_distillation`` to ``PYTHONPATH``
working without changing behavior.
"""

from __future__ import annotations

from pathlib import Path


_PACKAGE_DIR = Path(__file__).resolve().parent.parent
__path__ = [str(_PACKAGE_DIR)]
