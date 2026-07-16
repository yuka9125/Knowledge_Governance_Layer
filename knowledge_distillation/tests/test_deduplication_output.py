#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase3の最終候補出力ロジックの単体テスト。"""

import unittest

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


if __name__ == "__main__":
    unittest.main()
