#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web アプリ（Phase E）の単体テスト。

/ ・/health ・/chat を、fake エージェント注入で実キー・実Azure無しに検証する。
"""

import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from serving.governed_knowledge_api import GovernedKnowledgeService
from serving.web_app import create_web_app


SAMPLE_KNOWLEDGE = [
    {
        "knowledge_id": "k-010",
        "question": "VPNに接続できません",
        "answer": "最新版VPNクライアントへ更新してください。",
        "category": "質問",
        "approved_status": "approved",
        "approved_at": "2026-05-29T10:00:00+09:00",
    },
]


class _FakeEmbeddingBackend:
    def embed(self, texts):
        return None


class _FakeAgent:
    """answer を持つだけのダミーエージェント。"""

    def __init__(
        self,
        reply="（テスト応答）",
        raise_runtime=False,
        raise_generic=False,
    ):
        self.reply = reply
        self.raise_runtime = raise_runtime
        self.raise_generic = raise_generic
        self.last_question = None

    async def answer(self, question: str) -> str:
        self.last_question = question
        if self.raise_runtime:
            raise RuntimeError("Azure OpenAI(chat) の環境変数が未設定です: AZURE_OPENAI_API_KEY")
        if self.raise_generic:
            raise Exception("Azure OpenAI service failed")
        return self.reply


def _build_service():
    tmp_dir = tempfile.mkdtemp()
    path = Path(tmp_dir) / "approved_knowledge.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(SAMPLE_KNOWLEDGE, f, ensure_ascii=False)
    return GovernedKnowledgeService(
        approved_path=path, embedding_backend=_FakeEmbeddingBackend()
    )


class WebAppTest(unittest.TestCase):
    def _client(self, agent=None):
        app = create_web_app(
            knowledge_service=_build_service(),
            support_agent=agent or _FakeAgent(),
        )
        return TestClient(app)

    def test_health_still_works(self):
        response = self._client().get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_index_serves_html(self):
        response = self._client().get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("Knowledge Governance Chat", response.text)

    def test_knowledge_search_still_works(self):
        response = self._client().get(
            "/knowledge/search", params={"q": "VPNに接続できません"}
        )
        self.assertTrue(response.json()["answerable"])

    def test_chat_returns_agent_answer(self):
        agent = _FakeAgent(reply="最新版VPNクライアントへ更新してください。")
        response = self._client(agent).post(
            "/chat", json={"question": "VPNに接続できません"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["answer"], "最新版VPNクライアントへ更新してください。"
        )
        self.assertEqual(agent.last_question, "VPNに接続できません")

    def test_chat_empty_question_is_400(self):
        response = self._client().post("/chat", json={"question": "   "})
        self.assertEqual(response.status_code, 400)

    def test_chat_missing_question_is_422(self):
        response = self._client().post("/chat", json={})
        self.assertEqual(response.status_code, 422)

    def test_chat_agent_unavailable_is_503(self):
        agent = _FakeAgent(raise_runtime=True)
        response = self._client(agent).post(
            "/chat", json={"question": "VPNに接続できません"}
        )
        self.assertEqual(response.status_code, 503)
        self.assertIn("error", response.json())

    def test_chat_agent_service_error_is_503(self):
        agent = _FakeAgent(raise_generic=True)
        response = self._client(agent).post(
            "/chat", json={"question": "VPNに接続できません"}
        )
        self.assertEqual(response.status_code, 503)
        self.assertIn("error", response.json())
        self.assertNotIn("Azure OpenAI service failed", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
