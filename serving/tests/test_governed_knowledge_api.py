#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Governed Knowledge API（Phase C）の単体テスト。

実キー無し（テキスト類似度フォールバック）で動くことを確認する。
"""

import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from serving.governed_knowledge_api import (
    GovernedKnowledgeService,
    create_app,
)


SAMPLE_KNOWLEDGE = [
    {
        "knowledge_id": "k-007",
        "question": "飲食費・ゴルフ関係費申請の方法を教えてください",
        "answer": "イントラマートのWorkflowから申請一覧を開き、該当フローを選択してください。",
        "category": "質問",
        "approved_status": "approved",
        "approved_at": "2026-05-29T10:00:00+09:00",
    },
    {
        "knowledge_id": "k-010",
        "question": "VPNにつながりません",
        "answer": "最新版VPNクライアントへ更新してください。",
        "category": "質問",
        "approved_status": "approved",
        "approved_at": "2026-05-29T10:00:00+09:00",
    },
]


class _FakeEmbeddingBackend:
    """常に利用不可（None）を返し、テキストフォールバックを強制する。"""

    def embed(self, texts):
        return None


def _write_knowledge(items):
    tmp_dir = tempfile.mkdtemp()
    path = Path(tmp_dir) / "approved_knowledge.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False)
    return path


def _build_service(items):
    return GovernedKnowledgeService(
        approved_path=_write_knowledge(items),
        embedding_backend=_FakeEmbeddingBackend(),
    )


class GovernedKnowledgeServiceTest(unittest.TestCase):
    def test_exact_match_is_answerable(self):
        service = _build_service(SAMPLE_KNOWLEDGE)
        result = service.search("飲食費・ゴルフ関係費申請の方法を教えてください")
        self.assertTrue(result["answerable"])
        self.assertEqual(result["knowledge_id"], "k-007")
        self.assertEqual(result["source"], "approved_knowledge")
        self.assertIn("Workflow", result["answer"])

    def test_partial_match_is_answerable(self):
        service = _build_service(SAMPLE_KNOWLEDGE)
        result = service.search("VPNにつながりません。在宅勤務中です")
        self.assertTrue(result["answerable"])
        self.assertEqual(result["knowledge_id"], "k-010")

    def test_short_partial_match_is_not_answerable(self):
        service = _build_service(SAMPLE_KNOWLEDGE)
        result = service.search("申請方法")
        self.assertFalse(result["answerable"])
        self.assertEqual(result["fallback"], "human_review")

    def test_unknown_question_is_not_answerable(self):
        service = _build_service(SAMPLE_KNOWLEDGE)
        result = service.search("経費精算の締め日はいつですか")
        self.assertFalse(result["answerable"])
        self.assertEqual(result["reason"], "No approved knowledge found")
        self.assertEqual(result["fallback"], "human_review")

    def test_unapproved_knowledge_is_not_answerable(self):
        service = _build_service(
            [
                *SAMPLE_KNOWLEDGE,
                {
                    "knowledge_id": "k-draft",
                    "question": "未承認の質問です",
                    "answer": "この回答は返してはいけません。",
                    "category": "質問",
                    "approved_status": "draft",
                    "approved_at": "",
                },
            ]
        )
        result = service.search("未承認の質問です")
        self.assertFalse(result["answerable"])
        self.assertEqual(result["fallback"], "human_review")

    def test_empty_query_is_not_answerable(self):
        service = _build_service(SAMPLE_KNOWLEDGE)
        result = service.search("   ")
        self.assertFalse(result["answerable"])

    def test_missing_file_is_not_answerable(self):
        service = GovernedKnowledgeService(
            approved_path=Path(tempfile.mkdtemp()) / "missing.json",
            embedding_backend=_FakeEmbeddingBackend(),
        )
        result = service.search("VPNにつながりません")
        self.assertFalse(result["answerable"])

    def test_reload_on_file_update(self):
        path = _write_knowledge([])
        service = GovernedKnowledgeService(
            approved_path=path,
            embedding_backend=_FakeEmbeddingBackend(),
        )
        self.assertFalse(service.search("VPNにつながりません")["answerable"])

        with path.open("w", encoding="utf-8") as f:
            json.dump(SAMPLE_KNOWLEDGE, f, ensure_ascii=False)

        self.assertTrue(service.search("VPNにつながりません")["answerable"])


class GovernedKnowledgeApiEndpointTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app(_build_service(SAMPLE_KNOWLEDGE)))

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_search_answerable(self):
        response = self.client.get(
            "/knowledge/search", params={"q": "VPNにつながりません"}
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["answerable"])
        self.assertEqual(body["knowledge_id"], "k-010")

    def test_search_not_answerable(self):
        response = self.client.get(
            "/knowledge/search", params={"q": "存在しない質問です"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["answerable"])

    def test_search_requires_q(self):
        response = self.client.get("/knowledge/search")
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
