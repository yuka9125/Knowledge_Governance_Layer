from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence


def canonicalize_header(name: str) -> str:
    """ヘッダー名を比較用に正規化する。"""
    text = str(name).strip().lower()
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", "", text)
    text = text.replace("_", "").replace("-", "")
    return text


def build_header_index(headers: Iterable[str]) -> Dict[str, str]:
    """正規化キーから実ヘッダー名への逆引き辞書を作る。"""
    index: Dict[str, str] = {}
    for header in headers:
        key = canonicalize_header(header)
        if key and key not in index:
            index[key] = header
    return index


def resolve_first_header(
    headers: Iterable[str], aliases: Sequence[str]
) -> Optional[str]:
    """優先順のエイリアスで最初に一致したヘッダーを返す。"""
    index = build_header_index(headers)
    for alias in aliases:
        key = canonicalize_header(alias)
        if key in index:
            return index[key]
    return None


def resolve_all_headers(headers: Iterable[str], aliases: Sequence[str]) -> List[str]:
    """優先順のエイリアスに一致したヘッダーを重複なく返す。"""
    index = build_header_index(headers)
    resolved: List[str] = []
    seen = set()
    for alias in aliases:
        key = canonicalize_header(alias)
        if key in index:
            actual = index[key]
            if actual not in seen:
                seen.add(actual)
                resolved.append(actual)
    return resolved


def resolve_explicit_headers(
    headers: Iterable[str], specified_cols: Sequence[str] | None
) -> List[str]:
    """CLI明示指定列をヘッダー解決し、存在する列のみを順序維持で返す。"""
    if not specified_cols:
        return []
    index = build_header_index(headers)
    resolved: List[str] = []
    seen = set()
    for raw in specified_cols:
        key = canonicalize_header(raw)
        if key in index:
            actual = index[key]
            if actual not in seen:
                seen.add(actual)
                resolved.append(actual)
    return resolved


def join_labeled_non_empty_values(row: Dict[str, object], cols: Sequence[str]) -> str:
    """空欄を除外し、列名ラベル付きで連結する。"""
    blocks: List[str] = []
    for col in cols:
        value = str(row.get(col, "") or "").strip()
        if not value:
            continue
        blocks.append(f"【{col}】\n{value}")
    return "\n\n".join(blocks).strip()

