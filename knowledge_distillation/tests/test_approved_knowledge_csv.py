#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
承認済みKnowledgeのCSV出力・CSV読込（重複判定用）の単体テスト。
"""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from openpyxl import Workbook

from knowledge_distillation.approved_knowledge_exporter import (
    approved_knowledge_companion_path,
    export_approved_knowledge_files,
    export_approved_knowledge_from_excel,
)
from knowledge_distillation.approved_knowledge_io import (
    APPROVED_KNOWLEDGE_FIELDS,
    CSV_FORMAT,
    JSON_FORMAT,
    load_faq_matching_rows,
    read_approved_knowledge,
    resolve_existing_path,
    write_approved_knowledge_both,
    write_approved_knowledge_csv,
)


HEADERS = [
    "ナレッジID",
    "グループID",
    "候補_質問",
    "候補_回答",
    "カテゴリ",
    "参考リンク・資料名",
    "統合件数",
    "既存FAQ_ID",
    "既存FAQ_質問",
    "既存FAQ_回答",
    "既存FAQ_比較",
    "リスクレベル",
    "信頼度",
    "推奨アクション",
    "判定根拠",
    "レビュー結果",
]


def _row(
    knowledge_id="k-001",
    question="質問",
    answer="回答",
    category="IT",
    link_names="",
    existing_faq_id="",
    review="採用",
):
    return [
        knowledge_id, 1, question, answer, category, link_names, 1,
        existing_faq_id, "", "", "既存FAQなし/未照合",
        "low", 0.9, "新規FAQ作成", "", review,
    ]


def _create_reviewed_excel(rows) -> Path:
    tmp_dir = Path(tempfile.mkdtemp(prefix="approved_knowledge_csv_test_"))
    xlsx_path = tmp_dir / "FAQ_final_result.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.title = "最終ナレッジ候補一覧"
    ws.append(HEADERS)
    for row in rows:
        ws.append(row)
    wb.save(xlsx_path)
    return xlsx_path


class ApprovedKnowledgeCsvOutputTest(unittest.TestCase):
    """JSONを残したままCSVも出力されることを検証。"""

    def test_export_writes_json_and_csv_with_same_content(self):
        xlsx_path = _create_reviewed_excel(
            [
                _row(
                    knowledge_id="k-001",
                    question="VPNにつながりません",
                    answer="最新版へ更新してください。",
                    link_names="VPN手順書",
                ),
                _row(
                    knowledge_id="k-002",
                    question="対象外の質問",
                    answer="対象外の回答",
                    review="保留",
                ),
            ]
        )
        output_path = xlsx_path.parent / "approved_knowledge.json"

        outputs = export_approved_knowledge_files(xlsx_path, output_path)

        self.assertTrue(outputs[JSON_FORMAT].exists())
        self.assertTrue(outputs[CSV_FORMAT].exists())
        self.assertEqual(outputs[CSV_FORMAT].suffix, ".csv")

        json_items = read_approved_knowledge(outputs[JSON_FORMAT])
        csv_items = read_approved_knowledge(outputs[CSV_FORMAT])

        # 採用行だけがJSON/CSV双方に同じ内容で出る
        self.assertEqual(len(json_items), 1)
        self.assertEqual(len(csv_items), 1)
        for field in APPROVED_KNOWLEDGE_FIELDS:
            self.assertEqual(
                str(json_items[0].get(field, "")),
                str(csv_items[0].get(field, "")),
                msg=f"{field} がJSONとCSVで一致しません",
            )
        self.assertEqual(json_items[0]["question"], "VPNにつながりません")
        self.assertEqual(json_items[0]["link_names"], "VPN手順書")

    def test_csv_has_bom_and_standard_header_order(self):
        items = [
            {
                "knowledge_id": "k-001",
                "question": "質問",
                "answer": "回答",
                "category": "IT",
                "link_names": "",
                "approved_status": "approved",
                "approved_at": "2026-01-01T00:00:00+09:00",
            }
        ]
        tmp_dir = Path(tempfile.mkdtemp(prefix="approved_knowledge_csv_header_"))
        csv_path = write_approved_knowledge_csv(items, tmp_dir / "out.csv")

        raw = csv_path.read_bytes()
        self.assertTrue(raw.startswith(b"\xef\xbb\xbf"), "Excel向けBOMがありません")

        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            header = next(csv.reader(f))
        self.assertEqual(header, list(APPROVED_KNOWLEDGE_FIELDS))

    def test_csv_output_path_can_be_specified(self):
        xlsx_path = _create_reviewed_excel([_row()])
        json_path = xlsx_path.parent / "approved_knowledge.json"
        csv_path = xlsx_path.parent / "custom" / "knowledge.csv"

        outputs = export_approved_knowledge_files(
            xlsx_path, json_path, csv_output_path=csv_path
        )

        self.assertEqual(outputs[CSV_FORMAT], csv_path)
        self.assertTrue(csv_path.exists())

    def test_export_with_csv_output_path_writes_json_companion(self):
        xlsx_path = _create_reviewed_excel([_row()])
        csv_path = xlsx_path.parent / "approved_knowledge.csv"

        primary = export_approved_knowledge_from_excel(xlsx_path, csv_path)

        self.assertEqual(primary, csv_path)
        json_path = approved_knowledge_companion_path(csv_path)
        self.assertTrue(json_path.exists())
        self.assertEqual(len(read_approved_knowledge(json_path)), 1)

    def test_merge_base_can_be_csv(self):
        tmp_dir = Path(tempfile.mkdtemp(prefix="approved_knowledge_csv_base_"))
        base_csv = write_approved_knowledge_csv(
            [
                {
                    "knowledge_id": "k-20260101-1",
                    "question": "既存の質問",
                    "answer": "既存の回答",
                    "category": "IT",
                    "link_names": "",
                    "approved_status": "approved",
                    "approved_at": "2026-01-01T00:00:00+09:00",
                }
            ],
            tmp_dir / "approved_knowledge.csv",
        )

        xlsx_path = _create_reviewed_excel(
            [_row(knowledge_id="k-001", question="新しい質問", answer="新しい回答")]
        )
        outputs = export_approved_knowledge_files(
            xlsx_path,
            tmp_dir / "approved_knowledge.json",
            base_path=base_csv,
        )

        merged = read_approved_knowledge(outputs[JSON_FORMAT])
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["knowledge_id"], "k-20260101-1")
        self.assertEqual(merged[1]["question"], "新しい質問")


class ApprovedKnowledgeCsvMatchingTest(unittest.TestCase):
    """重複判定（Phase 3-2照合）の入力としてCSVを使えることを検証。"""

    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="approved_knowledge_matching_"))

    def test_matching_rows_from_csv_equal_rows_from_json(self):
        items = [
            {
                "knowledge_id": "k-001",
                "question": "VPNにつながりません",
                "answer": "最新版へ更新してください。",
                "category": "IT",
                "link_names": "",
                "approved_status": "approved",
                "approved_at": "2026-01-01T00:00:00+09:00",
            },
            {
                "knowledge_id": "k-002",
                "question": "未承認の質問",
                "answer": "未承認の回答",
                "category": "IT",
                "link_names": "",
                "approved_status": "draft",
                "approved_at": "",
            },
        ]
        tmp_dir = self._tmp_dir()
        outputs = write_approved_knowledge_both(
            items, tmp_dir / "approved_knowledge.json"
        )

        json_rows = load_faq_matching_rows(outputs[JSON_FORMAT])
        csv_rows = load_faq_matching_rows(outputs[CSV_FORMAT])

        self.assertEqual(json_rows, csv_rows)
        # approved 以外は照合対象から外れる
        self.assertEqual(len(csv_rows), 1)
        self.assertEqual(csv_rows[0]["質問"], "VPNにつながりません")
        self.assertEqual(csv_rows[0]["回答"], "最新版へ更新してください。")

    def test_csv_with_japanese_headers_is_accepted(self):
        tmp_dir = self._tmp_dir()
        csv_path = tmp_dir / "existing_faq.csv"
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ナレッジID", "質問", "回答", "カテゴリ"])
            writer.writerow(["FAQ-1", "経費精算の締日は？", "毎月末日です。", "経理"])

        rows = load_faq_matching_rows(csv_path)

        # approved_status 列が無い手作りCSVは全件を承認済みとして扱う
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["knowledge_id"], "FAQ-1")
        self.assertEqual(rows[0]["質問"], "経費精算の締日は？")
        self.assertEqual(rows[0]["カテゴリ"], "経理")

    def test_csv_is_used_when_json_is_missing(self):
        tmp_dir = self._tmp_dir()
        write_approved_knowledge_csv(
            [
                {
                    "knowledge_id": "k-001",
                    "question": "質問",
                    "answer": "回答",
                    "category": "IT",
                    "link_names": "",
                    "approved_status": "approved",
                    "approved_at": "",
                }
            ],
            tmp_dir / "approved_knowledge.csv",
        )
        json_path = tmp_dir / "approved_knowledge.json"

        self.assertFalse(json_path.exists())
        self.assertEqual(
            resolve_existing_path(json_path), tmp_dir / "approved_knowledge.csv"
        )
        self.assertEqual(len(load_faq_matching_rows(json_path)), 1)

    def test_shift_jis_csv_is_readable(self):
        tmp_dir = self._tmp_dir()
        csv_path = tmp_dir / "approved_knowledge.csv"
        with csv_path.open("w", encoding="cp932", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["質問", "回答"])
            writer.writerow(["パスワードを忘れました", "管理者へ連絡してください。"])

        rows = load_faq_matching_rows(csv_path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["質問"], "パスワードを忘れました")

    def test_missing_file_returns_empty(self):
        tmp_dir = self._tmp_dir()
        self.assertEqual(load_faq_matching_rows(tmp_dir / "nope.json"), [])

    def test_broken_json_returns_empty(self):
        tmp_dir = self._tmp_dir()
        json_path = tmp_dir / "approved_knowledge.json"
        json_path.write_text("{ broken", encoding="utf-8")
        self.assertEqual(read_approved_knowledge(json_path), [])

    def test_json_round_trip_keeps_extra_fields_in_csv(self):
        tmp_dir = self._tmp_dir()
        items = [
            {
                "knowledge_id": "k-001",
                "question": "質問",
                "answer": "回答",
                "category": "IT",
                "link_names": "",
                "approved_status": "approved",
                "approved_at": "",
                "memo": "補足メモ",
            }
        ]
        outputs = write_approved_knowledge_both(
            items, tmp_dir / "approved_knowledge.json"
        )
        csv_items = read_approved_knowledge(outputs[CSV_FORMAT])
        self.assertEqual(csv_items[0]["memo"], "補足メモ")
        # JSON側は従来どおりそのまま保持される
        json_items = json.loads(
            outputs[JSON_FORMAT].read_text(encoding="utf-8")
        )
        self.assertEqual(json_items[0]["memo"], "補足メモ")


if __name__ == "__main__":
    unittest.main()
