#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase3の最終候補出力ロジックの単体テスト。"""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from knowledge_distillation.deduplication_system import Phase3Processor
from knowledge_distillation.verification_excel import (
    DuplicateGroup,
    GroupCandidate,
    ProcessingRecord,
)


class Phase3OutputTest(unittest.TestCase):
    """統合件数の算出を検証する。"""

    def test_similar_logs_count_uses_final_gid_counts(self):
        processor = Phase3Processor()
        processor.faq_checked = False
        processor.source_df = pd.DataFrame(
            [
                {
                    "件名": "VPN接続不可",
                    "質問": "VPNに接続できない場合は？",
                    "回答": "最新版へ更新してください。",
                    "カテゴリ": "IT",
                    "信頼度": 0.95,
                }
            ],
            index=[0],
        )
        processor.processing_records = {
            0: ProcessingRecord(
                original_idx=0,
                final_gid=10,
                final_result="◯採用",
                question="VPNに接続できない場合は？",
                answer="最新版へ更新してください。",
            ),
            1: ProcessingRecord(
                original_idx=1,
                final_gid=10,
                final_result="P2削除（類似）",
                question="VPNにつながらない場合は？",
                answer="VPNクライアントを更新してください。",
                raw_overview="VPNにつながらない",
            ),
        }
        processor.final_groups = [
            DuplicateGroup(
                group_id=0,
                candidates=[
                    GroupCandidate(
                        original_idx=0,
                        rank=1,
                        is_adopted=True,
                        similarity=None,
                        question="VPNに接続できない場合は？",
                        answer="最新版へ更新してください。",
                        category="IT",
                        confidence_score=0.95,
                    )
                ],
            )
        ]

        candidates = processor._build_knowledge_candidates(processor.source_df)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["similar_logs_count"], 2)
        self.assertEqual(
            candidates[0]["answer_candidate_2"],
            "VPNクライアントを更新してください。",
        )
        self.assertEqual(candidates[0]["source_logs"][1], "VPNにつながらない")


class Phase3FinalResultFileTest(unittest.TestCase):
    """最終ナレッジ候補をJSONとCSVの両方で出力することを検証。"""

    def _candidates(self):
        return [
            {
                "knowledge_id": "k-001",
                "cluster_id": "c-001",
                "group_id": 1,
                "question": "VPNに接続できない場合は？",
                "answer": "最新版へ更新してください。",
                "category": "IT",
                "link_names": "VPN手順書",
                "source_logs": ["VPN接続不可", "VPNにつながらない"],
                "similar_logs_count": 2,
                "matched_faq_id": "",
                "matched_faq_question": "",
                "matched_faq_answer": "",
                "matched_faq_similarity": None,
                "existing_faq_comparison": "既存FAQなし/未照合",
                "final_result": "◯採用",
                "recommended_action": "新規FAQ作成",
                "judgement_reason": "信頼度理由: 0.95",
                "existing_faq_diff_reason": "FAQ未指定",
                "risk_level": "low",
                "review_status": "draft",
                "review_result": "",
                "confidence": 0.95,
            }
        ]

    def test_export_final_results_writes_json_and_csv(self):
        out_dir = Path(tempfile.mkdtemp(prefix="phase3_final_result_"))
        processor = Phase3Processor(output_dir=str(out_dir))

        json_path, csv_path = processor._export_final_results(self._candidates())

        self.assertTrue(Path(json_path).exists())
        self.assertTrue(Path(csv_path).exists())
        self.assertEqual(processor.final_json_path, json_path)
        self.assertEqual(processor.final_csv_path, csv_path)

        with open(json_path, encoding="utf-8") as f:
            json_rows = json.load(f)
        csv_df = pd.read_csv(csv_path, encoding="utf-8-sig")

        # JSONとCSVは同じ項目・同じ列順・同じ件数
        self.assertEqual(len(json_rows), 1)
        self.assertEqual(len(csv_df), 1)
        self.assertEqual(list(csv_df.columns), list(json_rows[0].keys()))
        self.assertEqual(csv_df.iloc[0]["knowledge_id"], "k-001")
        self.assertEqual(
            csv_df.iloc[0]["question"], "VPNに接続できない場合は？"
        )
        # source_logs は配列なのでCSVではJSON文字列として保持する
        self.assertEqual(
            json.loads(csv_df.iloc[0]["source_logs"]),
            ["VPN接続不可", "VPNにつながらない"],
        )


if __name__ == "__main__":
    unittest.main()
