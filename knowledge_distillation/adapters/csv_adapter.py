from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from knowledge_distillation.adapters.base import InputAdapter
from knowledge_distillation.adapters.utils import (
    join_labeled_non_empty_values,
    resolve_all_headers,
    resolve_explicit_headers,
    resolve_first_header,
)
from knowledge_distillation.models import NormalizedItem

QUESTION_ALIASES = [
    "質問",
    "問い合わせ",
    "問い合わせ内容",
    "問合せ内容",
    "件名",
    "タイトル",
    "subject",
    "title",
    "summary",
    "概要",
    "description",
    "body",
    "本文",
]

ANSWER_ALIASES = [
    "一次回答",
    "対応内容",
    "最終回答",
    "完了コメント",
    "第1対応",
    "第2対応",
    "最終結果",
    "回答",
    "対応結果",
    "解決方法",
    "解決内容",
    "answer",
    "response",
    "resolution",
    "close_notes",
    "close notes",
]

CATEGORY_ALIASES = [
    "カテゴリ",
    "category",
    "分類",
    "種別",
]

SOURCE_TEXT_ALIASES = [
    "詳細",
    "説明",
    "概要",
    "description",
    "body",
    "本文",
    "問い合わせ内容",
    "問合せ内容",
    "summary",
]


class CSVAdapter(InputAdapter):
    """デモCSV向けの汎用CSVアダプタ。"""

    def __init__(
        self,
        path: str | Path,
        source_name: str = "csv",
        question_col: str | None = None,
        answer_cols: Sequence[str] | None = None,
        source_text_cols: Sequence[str] | None = None,
    ):
        self.path = Path(path)
        self.source_name = source_name
        self.question_col = question_col
        self.answer_cols = list(answer_cols) if answer_cols else None
        self.source_text_cols = (
            list(source_text_cols) if source_text_cols else None
        )

    def fetch_items(self) -> Iterable[Dict[str, Any]]:
        if not self.path.exists():
            raise FileNotFoundError(f"CSVが見つかりません: {self.path}")

        encodings = ["utf-8-sig", "utf-8", "cp932", "shift-jis"]
        last_error: Exception | None = None
        for encoding in encodings:
            try:
                with self.path.open("r", encoding=encoding, newline="") as f:
                    reader = csv.DictReader(f)
                    return [dict(row) for row in reader]
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        raise ValueError(
            f"CSV読み込みに失敗しました: {self.path}"
        ) from last_error

    def normalize(self, items: Iterable[Dict[str, Any]]) -> List[NormalizedItem]:
        rows = list(items)
        if not rows:
            return []

        headers = list(rows[0].keys())
        question_col = self._resolve_question_col(headers)
        if not question_col:
            raise ValueError(
                "質問系の列が見つかりません。エイリアス候補を確認してください。"
            )

        answer_cols = self._resolve_answer_cols(headers)
        category_col = resolve_first_header(headers, CATEGORY_ALIASES)
        source_text_cols = self._resolve_source_text_cols(
            headers=headers,
            question_col=question_col,
            answer_cols=answer_cols,
        )

        normalized: List[NormalizedItem] = []
        for i, row in enumerate(rows):
            question = str(row.get(question_col, "") or "").strip()
            if not question:
                continue

            answer = join_labeled_non_empty_values(row, answer_cols)
            source_text = join_labeled_non_empty_values(row, source_text_cols)
            category = (
                str(row.get(category_col, "") or "").strip() if category_col else ""
            )

            metadata = {
                "row_index": i,
                "path": str(self.path),
                "mapped_columns": {
                    "question": question_col,
                    "source_text": source_text_cols,
                    "answer": answer_cols,
                    "category": category_col,
                },
                "raw": row,
            }
            normalized.append(
                NormalizedItem(
                    question=question,
                    source_text=source_text or question,
                    answer=answer,
                    category=category,
                    source=self.source_name,
                    metadata=metadata,
                )
            )
        return normalized

    def _resolve_question_col(self, headers: List[str]) -> str | None:
        if self.question_col:
            resolved = resolve_explicit_headers(headers, [self.question_col])
            if not resolved:
                raise ValueError(
                    f"指定された質問列が見つかりません: {self.question_col}"
                )
            return resolved[0]
        return resolve_first_header(headers, QUESTION_ALIASES)

    def _resolve_answer_cols(self, headers: List[str]) -> List[str]:
        if self.answer_cols is not None:
            resolved = resolve_explicit_headers(headers, self.answer_cols)
            if not resolved:
                raise ValueError(
                    "指定された回答列が見つかりません: "
                    + ", ".join(self.answer_cols)
                )
            return resolved
        return resolve_all_headers(headers, ANSWER_ALIASES)

    def _resolve_source_text_cols(
        self,
        headers: List[str],
        question_col: str,
        answer_cols: Sequence[str],
    ) -> List[str]:
        if self.source_text_cols is not None:
            resolved = resolve_explicit_headers(headers, self.source_text_cols)
            if not resolved:
                raise ValueError(
                    "指定されたsource_text列が見つかりません: "
                    + ", ".join(self.source_text_cols)
                )
            return resolved

        # 自動解決時は「本文系 + 詳細/説明系 + 回答系」を連結対象にする。
        cols: List[str] = [question_col]
        cols.extend(resolve_all_headers(headers, SOURCE_TEXT_ALIASES))
        cols.extend(answer_cols)
        unique: List[str] = []
        seen = set()
        for col in cols:
            if col and col not in seen:
                seen.add(col)
                unique.append(col)
        return unique
