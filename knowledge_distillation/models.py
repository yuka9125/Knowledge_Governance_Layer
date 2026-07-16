from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List


def utc_now_iso() -> str:
    """UTC現在時刻をISO8601形式で返す。"""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class NormalizedItem:
    """入力データの共通形式。"""

    question: str
    source_text: str
    answer: str = ""
    category: str = ""
    source: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FAQCandidate:
    """FAQ候補の共通形式。"""

    question: str
    answer: str
    category: str
    confidence: float
    sources: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

