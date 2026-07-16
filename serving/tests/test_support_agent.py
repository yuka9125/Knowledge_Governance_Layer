#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回答エージェント / Knowledge Plugin（Phase D）の単体テスト。

- Plugin: FastAPI TestClient を注入し、実キー・実サーバ無しで検証する。
- SupportAgent: fake エージェントを注入し、配線（ツール内蔵→応答抽出）を検証する。
"""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from serving.governed_knowledge_api import GovernedKnowledgeService, create_app
from serving.governed_knowledge_plugin import GovernedKnowledgePlugin
from serving.support_agent import SupportAgent, SYSTEM_INSTRUCTION


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


def _build_test_client():
    tmp_dir = tempfile.mkdtemp()
    path = Path(tmp_dir) / "approved_knowledge.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(SAMPLE_KNOWLEDGE, f, ensure_ascii=False)
    service = GovernedKnowledgeService(
        approved_path=path,
        embedding_backend=_FakeEmbeddingBackend(),
    )
    return TestClient(create_app(service))


class GovernedKnowledgePluginTest(unittest.TestCase):
    def setUp(self):
        # api_base="" でTestClientのルート相対URLに合わせる
        self.plugin = GovernedKnowledgePlugin(
            api_base="", http_client=_build_test_client()
        )

    def test_search_hit_returns_answerable_true(self):
        raw = self.plugin.search_approved_knowledge("VPNに接続できません")
        body = json.loads(raw)
        self.assertTrue(body["answerable"])
        self.assertEqual(body["knowledge_id"], "k-010")
        self.assertIn("VPNクライアント", body["answer"])

    def test_search_miss_returns_answerable_false(self):
        raw = self.plugin.search_approved_knowledge("経費精算の締め日は？")
        body = json.loads(raw)
        self.assertFalse(body["answerable"])
        self.assertEqual(body["fallback"], "human_review")

    def test_api_error_is_handled_as_not_answerable(self):
        # 到達不能なベースURLでもエラーを握り、推測させない応答を返す
        plugin = GovernedKnowledgePlugin(
            api_base="http://127.0.0.1:1", timeout=0.2
        )
        body = json.loads(plugin.search_approved_knowledge("VPNに接続できません"))
        self.assertFalse(body["answerable"])
        self.assertEqual(body["fallback"], "human_review")


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content


class _FakeAgent:
    """get_response を持つだけのダミーエージェント。"""

    def __init__(self):
        self.received_messages = None

    async def get_response(self, messages=None, **kwargs):
        self.received_messages = messages
        return _FakeResponse("（承認済み回答に基づく応答）")


class _FakePlugin:
    def __init__(self, raw_result: str):
        self.raw_result = raw_result
        self.last_question = None

    def search_approved_knowledge(self, question: str) -> str:
        self.last_question = question
        return self.raw_result


class SupportAgentWiringTest(unittest.TestCase):
    def test_answer_returns_approved_knowledge_without_guessing(self):
        plugin = _FakePlugin(
            json.dumps(
                {
                    "answerable": True,
                    "knowledge_id": "k-010",
                    "answer": "最新版VPNクライアントへ更新してください。",
                },
                ensure_ascii=False,
            )
        )
        fake_agent = _FakeAgent()
        agent = SupportAgent(plugin=plugin, agent=fake_agent)
        result = asyncio.run(agent.answer("VPNに接続できません"))
        self.assertIn("最新版VPNクライアントへ更新してください。", result)
        self.assertIn("knowledge_id: k-010", result)
        self.assertEqual(plugin.last_question, "VPNに接続できません")
        self.assertIsNone(fake_agent.received_messages)

    def test_answer_escalates_when_not_answerable(self):
        plugin = _FakePlugin(
            json.dumps(
                {
                    "answerable": False,
                    "reason": "No approved knowledge found",
                    "fallback": "human_review",
                },
                ensure_ascii=False,
            )
        )
        agent = SupportAgent(plugin=plugin, agent=_FakeAgent())
        result = asyncio.run(agent.answer("経費精算の締め日は？"))
        self.assertIn("承認済みナレッジに該当がありません", result)
        self.assertIn("担当部門", result)

    def test_system_instruction_enforces_governance(self):
        # 承認済みのみ・推測禁止・エスカレーションの方針が指示に含まれること
        self.assertIn("search_approved_knowledge", SYSTEM_INSTRUCTION)
        self.assertIn("answerable=false", SYSTEM_INSTRUCTION)
        self.assertIn("担当部門", SYSTEM_INSTRUCTION)
        self.assertIn("禁止", SYSTEM_INSTRUCTION)


if __name__ == "__main__":
    unittest.main()
