#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Streamlit app summary helpersの単体テスト。"""

import contextlib
import importlib
import io
import unittest

from knowledge_distillation.verification_excel import ProcessingRecord


class AppSummaryTest(unittest.TestCase):
    """app表示用の件数集計を検証する。"""

    def test_p32_similar_is_counted_as_faq_excluded(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            app = importlib.import_module("knowledge_distillation.app")

        records = {
            1: ProcessingRecord(
                original_idx=1,
                final_result="◯採用",
                answer="answer",
            ),
            2: ProcessingRecord(
                original_idx=2,
                final_result="P3-2確認（既存FAQ完全一致）",
                answer="answer",
            ),
            3: ProcessingRecord(
                original_idx=3,
                final_result="P3-2確認（既存FAQ類似）",
                answer="answer",
            ),
            4: ProcessingRecord(
                original_idx=4,
                final_result="P3-2確認（既存FAQ更新候補）",
                answer="answer",
            ),
        }

        review_count, excluded_count = app.summarize_review_targets(records)

        self.assertEqual(review_count, 2)
        self.assertEqual(excluded_count, 2)


if __name__ == "__main__":
    unittest.main()
