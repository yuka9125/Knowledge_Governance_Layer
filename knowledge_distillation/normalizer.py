from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Sequence

from knowledge_distillation.adapters.utils import (
    build_header_index,
    canonicalize_header,
    join_labeled_non_empty_values,
)
from knowledge_distillation.models import NormalizedItem

TITLE_FALLBACK_MAX_LEN = 60

TITLE_ALIASES = [
    "コールタイトル",
    "件名",
    "タイトル",
    "subject",
    "title",
]

CATEGORY_ALIASES = [
    "種別",
    "カテゴリ名",
    "分類",
    "カテゴリ",
    "category",
]

ANSWER_ALIASES = [
    "対応内容",
    "対応内容(一次)",
    "対応内容(二次)",
    "第1対応",
    "第2対応",
    "最終結果",
    "一次回答",
    "二次回答",
    "最終回答",
    "回答",
    "解決内容",
    "対応結果",
    "response",
    "resolution",
    "close notes",
    "close_notes",
]

LINK_ALIASES = [
    "リンク名",
    "関連リンク",
    "link",
    "link_name",
    "url",
]


def _resolve_one(row: Dict[str, object], explicit_col: str | None, aliases: Sequence[str]) -> str:
    headers = [str(k) for k in row.keys()]
    index = build_header_index(headers)

    if explicit_col:
        key = canonicalize_header(explicit_col)
        actual = index.get(key)
        if actual:
            return str(row.get(actual, "") or "").strip()

    for alias in aliases:
        key = canonicalize_header(alias)
        actual = index.get(key)
        if not actual:
            continue
        value = str(row.get(actual, "") or "").strip()
        if value:
            return value
    return ""


def _resolve_many(
    row: Dict[str, object],
    explicit_cols: Sequence[str] | None,
    aliases: Sequence[str],
) -> List[str]:
    headers = [str(k) for k in row.keys()]
    index = build_header_index(headers)
    resolved: List[str] = []
    seen = set()

    def add_col(raw_col: str) -> None:
        key = canonicalize_header(raw_col)
        actual = index.get(key)
        if not actual or actual in seen:
            return
        seen.add(actual)
        resolved.append(actual)

    if explicit_cols:
        for col in explicit_cols:
            add_col(col)
        return resolved

    for alias in aliases:
        add_col(alias)
    return resolved


def _fallback_title(overview: str) -> str:
    text = (overview or "").strip()
    if not text:
        return ""
    if len(text) <= TITLE_FALLBACK_MAX_LEN:
        return text
    return text[:TITLE_FALLBACK_MAX_LEN]


def _build_snow_row(
    item: NormalizedItem,
    question_col: str | None,
    answer_cols: Sequence[str] | None,
    title_col: str | None,
    category_col: str | None,
    link_col: str | None,
) -> Dict[str, str]:
    raw_row = item.metadata.get("raw", {})
    row = raw_row if isinstance(raw_row, dict) else {}

    overview = _resolve_one(row, question_col, aliases=[])
    if not overview:
        overview = item.question.strip() or item.source_text.strip()

    response_cols = _resolve_many(row, explicit_cols=answer_cols, aliases=ANSWER_ALIASES)
    response = join_labeled_non_empty_values(row, response_cols) if response_cols else ""
    if not response:
        response = item.answer.strip()

    title = _resolve_one(row, explicit_col=title_col, aliases=TITLE_ALIASES)
    if not title:
        title = _fallback_title(overview)

    category = _resolve_one(row, explicit_col=category_col, aliases=CATEGORY_ALIASES)
    if not category:
        category = item.category.strip()

    link_name = _resolve_one(row, explicit_col=link_col, aliases=LINK_ALIASES)

    return {
        "件名": title,
        "カテゴリ": category,
        "概要": overview,
        "対応結果": response,
        "リンク名": link_name,
    }


def normalize_items_to_snow_csv(
    items: Sequence[NormalizedItem],
    out_path: str | Path,
    question_col: str | None = None,
    answer_cols: Sequence[str] | None = None,
    title_col: str | None = None,
    category_col: str | None = None,
    link_col: str | None = None,
) -> Dict[str, object]:
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []
    for item in items:
        rows.append(
            _build_snow_row(
                item=item,
                question_col=question_col,
                answer_cols=answer_cols,
                title_col=title_col,
                category_col=category_col,
                link_col=link_col,
            )
        )

    headers = ["件名", "カテゴリ", "概要", "対応結果", "リンク名"]
    with target.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    return {"out_path": str(target), "count": len(rows)}
