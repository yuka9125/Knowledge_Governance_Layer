from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from knowledge_distillation.adapters.base import InputAdapter
from knowledge_distillation.adapters.csv_adapter import CSVAdapter
from knowledge_distillation.adapters.utils import (
    join_labeled_non_empty_values,
    resolve_all_headers,
    resolve_explicit_headers,
    resolve_first_header,
)
from knowledge_distillation.models import NormalizedItem

SNOW_SUBJECT_ALIASES = ["件名", "subject", "title", "タイトル"]
SNOW_CATEGORY_ALIASES = ["カテゴリ", "category", "分類", "種別"]
SNOW_OVERVIEW_ALIASES = [
    "概要",
    "description",
    "説明",
    "問い合わせ内容",
    "問合せ内容",
    "本文",
]
SNOW_RESPONSE_ALIASES = [
    "対応結果",
    "resolution",
    "close_notes",
    "解決内容",
    "対応内容",
    "回答",
]


class ServiceNowAdapter(InputAdapter):
    """ServiceNowエクスポートCSV向けアダプタ。"""

    def __init__(
        self,
        path: str | Path,
        question_col: str | None = None,
        answer_cols: Sequence[str] | None = None,
        source_text_cols: Sequence[str] | None = None,
    ):
        self.path = Path(path)
        self.question_col = question_col
        self.answer_cols = list(answer_cols) if answer_cols else None
        self.source_text_cols = (
            list(source_text_cols) if source_text_cols else None
        )
        self._csv_adapter = CSVAdapter(path=self.path, source_name="servicenow")

    def fetch_items(self) -> Iterable[Dict[str, Any]]:
        return self._csv_adapter.fetch_items()

    def normalize(self, items: Iterable[Dict[str, Any]]) -> List[NormalizedItem]:
        rows = list(items)
        if not rows:
            return []

        headers = list(rows[0].keys())
        subject_col = self._resolve_subject_col(headers)
        category_col = resolve_first_header(headers, SNOW_CATEGORY_ALIASES)
        overview_col = resolve_first_header(headers, SNOW_OVERVIEW_ALIASES)
        answer_cols = self._resolve_answer_cols(headers)

        # ServiceNowは件名優先。件名がなければ概要で補完する。
        question_col = self._resolve_question_col(
            headers=headers, subject_col=subject_col, overview_col=overview_col
        )
        if not question_col:
            raise ValueError("ServiceNow入力の件名/概要系列が見つかりません。")
        source_text_cols = self._resolve_source_text_cols(
            headers=headers,
            question_col=question_col,
            overview_col=overview_col,
            answer_cols=answer_cols,
        )

        normalized: List[NormalizedItem] = []
        for i, row in enumerate(rows):
            question = str(row.get(question_col, "") or "").strip()
            if not question:
                continue

            answer = join_labeled_non_empty_values(row, answer_cols)
            category = (
                str(row.get(category_col, "") or "").strip() if category_col else ""
            )
            source_text = join_labeled_non_empty_values(row, source_text_cols)

            metadata = {
                "row_index": i,
                "path": str(self.path),
                "mapped_columns": {
                    "question": question_col,
                    "answer": answer_cols,
                    "source_text": source_text_cols,
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
                    source="servicenow",
                    metadata=metadata,
                )
            )
        return normalized

    def _resolve_subject_col(self, headers: List[str]) -> str | None:
        return resolve_first_header(headers, SNOW_SUBJECT_ALIASES)

    def _resolve_question_col(
        self,
        headers: List[str],
        subject_col: str | None,
        overview_col: str | None,
    ) -> str | None:
        if self.question_col:
            resolved = resolve_explicit_headers(headers, [self.question_col])
            if not resolved:
                raise ValueError(
                    f"指定された質問列が見つかりません: {self.question_col}"
                )
            return resolved[0]
        return subject_col or overview_col

    def _resolve_answer_cols(self, headers: List[str]) -> List[str]:
        if self.answer_cols is not None:
            resolved = resolve_explicit_headers(headers, self.answer_cols)
            if not resolved:
                raise ValueError(
                    "指定された回答列が見つかりません: "
                    + ", ".join(self.answer_cols)
                )
            return resolved
        return resolve_all_headers(headers, SNOW_RESPONSE_ALIASES)

    def _resolve_source_text_cols(
        self,
        headers: List[str],
        question_col: str,
        overview_col: str | None,
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

        cols: List[str] = [question_col]
        if overview_col:
            cols.append(overview_col)
        cols.extend(resolve_all_headers(headers, SNOW_OVERVIEW_ALIASES))
        cols.extend(answer_cols)

        unique: List[str] = []
        seen = set()
        for col in cols:
            if col and col not in seen:
                seen.add(col)
                unique.append(col)
        return unique
