#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
承認済みKnowledgeの入出力（JSON / CSV 両対応）。

- JSONは従来どおりの契約ファイル（serving が参照する形式）で維持する。
- CSVはExcelでそのまま開ける同内容のファイルとして併せて出力する。
- 重複判定（Phase 3-2の既存FAQ照合）は JSON / CSV どちらからでも読み込める。

CSVは英語ヘッダー（JSONのキー名）で出力するが、読み込み時は日本語ヘッダーの
手作りCSVも受け付ける（CSV_HEADER_ALIASES 参照）。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

# approved_knowledge の標準項目（JSON/CSVで共通・この順で出力する）
APPROVED_KNOWLEDGE_FIELDS: Sequence[str] = (
    "knowledge_id",
    "question",
    "answer",
    "category",
    "link_names",
    "approved_status",
    "approved_at",
)

JSON_FORMAT = "json"
CSV_FORMAT = "csv"

# CSVヘッダーの別名。手作業で用意した既存FAQ CSVも重複判定に使えるようにする。
CSV_HEADER_ALIASES: Dict[str, Sequence[str]] = {
    "knowledge_id": (
        "knowledge_id",
        "ナレッジID",
        "ナレッジid",
        "FAQ_ID",
        "faq_id",
        "FAQID",
        "id",
        "ID",
    ),
    "question": ("question", "質問", "候補_質問", "Q"),
    "answer": ("answer", "回答", "候補_回答", "A"),
    "category": ("category", "カテゴリ"),
    "link_names": ("link_names", "参考リンク・資料名", "リンク名"),
    "approved_status": ("approved_status", "承認ステータス", "承認状態"),
    "approved_at": ("approved_at", "承認日時", "承認日"),
}

# Phase 3-2（既存FAQ照合）が期待する列名
FAQ_MATCHING_COLUMNS: Sequence[str] = (
    "質問",
    "回答",
    "カテゴリ",
    "knowledge_id",
    "approved_at",
)

APPROVED_STATUS_VALUE = "approved"


def detect_format(path: str | Path) -> str:
    """拡張子から入出力フォーマットを判定する（既定はJSON）。"""
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return CSV_FORMAT
    return JSON_FORMAT


def companion_path(path: str | Path, fmt: str | None = None) -> Path:
    """同じ内容を別フォーマットで置くときのパスを返す。

    fmt 未指定なら「もう一方のフォーマット」を返す（json ⇄ csv）。
    """
    target = Path(path)
    if fmt is None:
        fmt = CSV_FORMAT if detect_format(target) == JSON_FORMAT else JSON_FORMAT
    suffix = ".csv" if fmt == CSV_FORMAT else ".json"
    return target.with_suffix(suffix)


def resolve_existing_path(path: str | Path) -> Optional[Path]:
    """指定パスが無ければ同名のもう一方のフォーマットを探して返す。"""
    target = Path(path)
    if target.exists():
        return target
    alternative = companion_path(target)
    if alternative.exists():
        return alternative
    return None


def _normalize_header(header: Any) -> str:
    """CSVヘッダーを比較用に整える。"""
    return str(header or "").replace("﻿", "").strip()


def _build_csv_field_map(fieldnames: Iterable[Any]) -> Dict[str, str]:
    """CSVの実ヘッダー名 → 標準キー名のマップを作る。"""
    alias_to_field: Dict[str, str] = {}
    for field, aliases in CSV_HEADER_ALIASES.items():
        for alias in aliases:
            alias_to_field[alias.lower()] = field

    field_map: Dict[str, str] = {}
    used_fields: set[str] = set()
    for header in fieldnames or []:
        normalized = _normalize_header(header)
        if not normalized:
            continue
        field = alias_to_field.get(normalized.lower())
        if field is not None and field not in used_fields:
            field_map[header] = field
            used_fields.add(field)
        else:
            # 未知の列は列名のまま保持する（情報を落とさない）
            field_map[header] = normalized
    return field_map


def read_approved_knowledge(path: str | Path) -> List[Dict[str, str]]:
    """approved_knowledge を JSON / CSV から読む（無い・壊れていれば空）。"""
    target = Path(path)
    if not target.exists():
        return []
    if detect_format(target) == CSV_FORMAT:
        return _read_csv(target)
    return _read_json(target)


def _read_json(path: Path) -> List[Dict[str, str]]:
    """JSON配列を読む。"""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def _read_csv(path: Path) -> List[Dict[str, str]]:
    """CSV（BOM付きUTF-8 / UTF-8 / Shift-JIS）を読む。"""
    for encoding in ("utf-8-sig", "utf-8", "cp932"):
        try:
            with path.open("r", encoding=encoding, newline="") as f:
                reader = csv.DictReader(f)
                field_map = _build_csv_field_map(reader.fieldnames or [])
                items: List[Dict[str, str]] = []
                for raw_row in reader:
                    row: Dict[str, str] = {}
                    for header, value in raw_row.items():
                        key = field_map.get(header)
                        if not key:
                            continue
                        row[key] = "" if value is None else str(value).strip()
                    if any(row.values()):
                        items.append(row)
                return items
        except UnicodeDecodeError:
            continue
        except OSError:
            return []
    return []


def _csv_columns(items: Sequence[Dict[str, Any]]) -> List[str]:
    """標準項目を先頭に、追加項目を出現順で足した列順を作る。"""
    columns = list(APPROVED_KNOWLEDGE_FIELDS)
    for item in items:
        for key in item.keys():
            if key not in columns:
                columns.append(key)
    return columns


def write_approved_knowledge_json(
    items: Sequence[Dict[str, Any]], path: str | Path
) -> Path:
    """approved_knowledge をJSONで書き出す。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(list(items), f, ensure_ascii=False, indent=2)
    return target


def write_approved_knowledge_csv(
    items: Sequence[Dict[str, Any]], path: str | Path
) -> Path:
    """approved_knowledge をCSVで書き出す（Excel向けBOM付きUTF-8）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    columns = _csv_columns(items)
    with target.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    column: "" if item.get(column) is None else str(item.get(column))
                    for column in columns
                }
            )
    return target


def write_approved_knowledge(
    items: Sequence[Dict[str, Any]], path: str | Path
) -> Path:
    """拡張子に応じてJSON/CSVのどちらかで書き出す。"""
    if detect_format(path) == CSV_FORMAT:
        return write_approved_knowledge_csv(items, path)
    return write_approved_knowledge_json(items, path)


def write_approved_knowledge_both(
    items: Sequence[Dict[str, Any]],
    path: str | Path,
    csv_path: str | Path | None = None,
    json_path: str | Path | None = None,
) -> Dict[str, Path]:
    """JSONとCSVの両方を書き出し、{"json": Path, "csv": Path} を返す。

    path はどちらの拡張子でもよい。指定が無いもう一方は同名の別拡張子に出力する。
    """
    primary = Path(path)
    if detect_format(primary) == CSV_FORMAT:
        csv_target = Path(csv_path) if csv_path is not None else primary
        json_target = (
            Path(json_path)
            if json_path is not None
            else companion_path(primary, JSON_FORMAT)
        )
    else:
        json_target = Path(json_path) if json_path is not None else primary
        csv_target = (
            Path(csv_path)
            if csv_path is not None
            else companion_path(primary, CSV_FORMAT)
        )

    return {
        JSON_FORMAT: write_approved_knowledge_json(items, json_target),
        CSV_FORMAT: write_approved_knowledge_csv(items, csv_target),
    }


def filter_approved(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """承認済みの行だけ残す。

    approved_status 列（キー）がどの行にも無いソースは「承認済みの一覧」と
    みなして全件通す（手作りの既存FAQ CSVをそのまま照合に使えるようにする）。
    """
    has_status = any("approved_status" in item for item in items)
    if not has_status:
        return list(items)
    return [
        item
        for item in items
        if str(item.get("approved_status", "")).strip().lower()
        == APPROVED_STATUS_VALUE
    ]


def to_faq_matching_rows(
    items: Sequence[Dict[str, Any]]
) -> List[Dict[str, str]]:
    """承認済みKnowledgeをPhase 3-2の照合用行（日本語列名）へ変換する。"""
    rows: List[Dict[str, str]] = []
    for item in filter_approved(items):
        question = str(item.get("question", "") or "").strip()
        answer = str(item.get("answer", "") or "").strip()
        if not question or not answer:
            continue
        rows.append(
            {
                "質問": question,
                "回答": answer,
                "カテゴリ": str(item.get("category", "") or "").strip(),
                "knowledge_id": str(item.get("knowledge_id", "") or "").strip(),
                "approved_at": str(item.get("approved_at", "") or "").strip(),
            }
        )
    return rows


def load_faq_matching_rows(path: str | Path) -> List[Dict[str, str]]:
    """JSON/CSVのどちらからでもPhase 3-2照合用の行を読み込む。

    指定パスが存在しなければ同名のもう一方のフォーマットを自動で探す。
    """
    resolved = resolve_existing_path(path)
    if resolved is None:
        return []
    return to_faq_matching_rows(read_approved_knowledge(resolved))
