#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回答エージェント（Phase D / Semantic Kernel）。

承認済みKnowledgeのみに基づいて回答する社内サポートエージェント。
1. ユーザーの質問を受ける
2. ツール search_approved_knowledge（Governed Knowledge API）を呼ぶ
3. answerable=true なら承認済み answer を根拠に回答
4. answerable=false なら推測せずエスカレーション（担当部門確認）を案内

Azure OpenAI(chat) が必要なのは実応答生成のみ。chatサービス/エージェントは
注入できるようにしてあり、実キー無しでも配線テストが可能。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any, Optional

from serving.governed_knowledge_plugin import GovernedKnowledgePlugin


SYSTEM_INSTRUCTION = """あなたは社内サポートのエージェントです。
回答は必ず search_approved_knowledge ツールの結果のみに基づいてください。

- まず必ず search_approved_knowledge ツールを呼び、承認済みKnowledgeを検索する。
- answerable=true のときだけ、その answer を根拠に回答する。回答は answer の内容に忠実に行う。
- answerable=false のときは推測で回答せず、
  「承認済みナレッジに該当がありませんでした。お手数ですが担当部門へご確認ください。」と案内し、
  エスカレーションを促す。
- 未承認Knowledgeや一般知識・推測からの回答は禁止。
- 日本語で、簡潔かつ丁寧に回答する。
"""

DEFAULT_API_VERSION = "2024-10-21"
AGENT_NAME = "GovernedSupportAgent"


def build_azure_chat_service():
    """環境変数から Azure OpenAI(chat) サービスを構築する。

    未設定の場合は実行時に分かるよう例外を投げる（importは失敗させない）。
    """
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", DEFAULT_API_VERSION)

    missing = [
        name
        for name, value in (
            ("AZURE_OPENAI_ENDPOINT", endpoint),
            ("AZURE_OPENAI_API_KEY", api_key),
            ("AZURE_OPENAI_CHAT_DEPLOYMENT", deployment),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Azure OpenAI(chat) の環境変数が未設定です: " + ", ".join(missing)
        )

    # importはここで行う（未インストール環境でのモジュール読み込み失敗を避ける）
    from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

    return AzureChatCompletion(
        deployment_name=deployment,
        endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )


def build_agent(chat_service: Any, plugin: GovernedKnowledgePlugin):
    """Semantic Kernel の ChatCompletionAgent を構築する。"""
    from semantic_kernel.agents import ChatCompletionAgent
    from semantic_kernel.connectors.ai import FunctionChoiceBehavior

    return ChatCompletionAgent(
        service=chat_service,
        name=AGENT_NAME,
        instructions=SYSTEM_INSTRUCTION,
        plugins=[plugin],
        function_choice_behavior=FunctionChoiceBehavior.Auto(),
    )


def _extract_text(response: Any) -> str:
    """エージェント応答からテキストを取り出す。"""
    content = getattr(response, "content", None)
    if content is not None:
        return str(content)
    return str(response)


class SupportAgent:
    """承認済みKnowledgeに基づく回答エージェント。"""

    def __init__(
        self,
        chat_service: Any | None = None,
        plugin: GovernedKnowledgePlugin | None = None,
        agent: Any | None = None,
    ) -> None:
        self._chat_service = chat_service
        self._plugin = plugin or GovernedKnowledgePlugin()
        self._agent = agent

    def _ensure_agent(self) -> Any:
        if self._agent is None:
            service = self._chat_service or build_azure_chat_service()
            self._agent = build_agent(service, self._plugin)
        return self._agent

    async def answer(self, question: str) -> str:
        """質問に対し、承認済みKnowledge検索→answer/escalate分岐で返す。"""
        raw_result = self._plugin.search_approved_knowledge(question)
        try:
            result = json.loads(raw_result)
        except json.JSONDecodeError:
            result = {
                "answerable": False,
                "reason": "Invalid knowledge search result",
                "fallback": "human_review",
            }

        if not result.get("answerable"):
            return "承認済みナレッジに該当がありませんでした。お手数ですが担当部門へご確認ください。"

        answer = str(result.get("answer", "")).strip()
        knowledge_id = str(result.get("knowledge_id", "")).strip()
        if not answer:
            return "承認済みナレッジに該当がありませんでした。お手数ですが担当部門へご確認ください。"

        if knowledge_id:
            return f"{answer}\n\n根拠: 承認済みナレッジ（knowledge_id: {knowledge_id}）"
        return f"{answer}\n\n根拠: 承認済みナレッジ"


async def _amain(question: str) -> None:
    agent = SupportAgent()
    print(await agent.answer(question))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="承認済みKnowledgeに基づく回答エージェント（要 Azure OpenAI / 起動中のGoverned Knowledge API）。"
    )
    parser.add_argument("question", help="ユーザーの質問")
    args = parser.parse_args()
    asyncio.run(_amain(args.question))


if __name__ == "__main__":
    main()
